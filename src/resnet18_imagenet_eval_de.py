import argparse
import time
import math
import os, sys
from unittest import result
from meta_module import MetaModule
from opt import NAG, RMSProp, Adam

import torch
import torch.nn as nn
import torch.autograd as autograd

import torchvision

from networks import RNNOptimizer

import wandb

from resnet_imagenet import resnet18
torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import add_gpu_params, parse_gpu, add_other_params
from optimizer import add_optimizer_params, create_adam_optimizer_from_args


from data_utils import FT_Dataset # BinCorpus, BinLMOrderedIterator
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir, set_seed, AverageMeter, rgetattr, rsetattr, evaluate_cifar10, print_args, hutch_loop

from utils import setup_model_dataset
import itertools

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch GPT2 evaluation script.')
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--meta_train_eval_epoch", type=int, default=5)


add_gpu_params(parser)
add_optimizer_params(parser)
add_other_params(parser)

# influence model, calculate the influence score between two samples
#gamma = args.gamma
#alpha = args.alpha
grad_res_momentum = 0

# Store the last gradient on basis variables for P_plus_BFGS update
gk_last = None

# Variables for BFGS and backtracking line search
rho = 0.55
rho = 0.4
sigma = 0.4
#Bk = torch.eye(args.n_components).cuda()
sk = None

# Store the backtracking line search times
search_times = []

def update_grad(model, grad_vec):
	idx = 0
	for name,param in model.named_parameters():
		arr_shape = param.grad.shape
		size = 1
		for i in range(len(list(arr_shape))):
			size *= arr_shape[i]
		param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
		idx += size


def get_model_grad_vec(model):
	# Return the model grad as a vector

	vec = []
	for name,param in model.named_parameters():
		vec.append(param.grad.detach().reshape(-1))
	return torch.cat(vec, 0)

def get_model_param_vec(model):
	"""
	Return model parameters as a vector
	"""
	vec = []
	for name,param in model.named_parameters():
		vec.append(param.detach().cpu().numpy().reshape(-1))
	return np.concatenate(vec, 0)

def to_use(name):
	return 'theta' in name
	
def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, P, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_accuracy=None):
	training_steps = [100, 200, 300, 400, 500, 1000, 2000, 5000]
	model.train()
	opt_net.train()
	meta_optimizer.zero_grad()
	avg_lm_loss = AverageMeter()
	print('start to train the model................', epoch)
	log_start_time = time.time()
	all_losses = []
	mt = []
	vt = []
	hidden_states = []
	cell_states = []
	params_to_calc_grad = []
	param_names = []

	model = create_model().cuda()
	if args.use_second_layer:
		nlayer=2
	else:
		nlayer=1

	mt.append(torch.zeros(P.shape[0], 1, requires_grad=False).cuda())
	vt.append(torch.zeros(P.shape[0], 1, requires_grad=False).cuda())
	hidden_states.append([torch.zeros(P.shape[0], opt_net.hidden_sz, requires_grad=True).to(args.local_rank) for _ in range(nlayer)])
	cell_states.append([torch.zeros(P.shape[0], opt_net.hidden_sz, requires_grad=True).to(args.local_rank) for _ in range(nlayer)])
	#param_names.append(name)
	#params_to_calc_grad.append(p)

	train_iter = iter(train_loader)
	current_idx = 0

	for epoch in range(args.max_epoch):

		for _input, _label in train_loader:
			_input = _input.cuda()
			_label = _label.cuda()
			current_idx += 1
			with torch.autograd.set_detect_anomaly(True):
				logits = model(_input)
				_loss = torch.nn.functional.cross_entropy(logits, _label)
				
				wandb.log({"meta_eval/train_loss": _loss.data.cpu().numpy()}, step=train_step)
				train_step += 1
				avg_lm_loss.update(_loss.item())

				
				result_params = {}
				#all_params_gradients = torch.autograd.grad(_loss, model.parameters(), only_inputs=False, create_graph=True)
				_loss.backward(retain_graph=True)
				flatten = []
				for name, p in model.named_parameters():
					#print(name, p.abs().mean())
					flatten.append(p.grad.data.reshape(-1,1))
				#if current_idx == 2: assert False
				flatten = torch.cat(flatten, 0)
				all_losses.append(_loss)
				result_h = []
				result_c = []

				grad = torch.mm(P, flatten)

				#for name, p, grad, m, v, h, c in zip(param_names, model.parameters(), all_params_gradients, mt, vt, hidden_states, cell_states):
				mt[0].mul_(beta1).add_((1 - beta1) * grad)
				mt_hat = mt[0] / (1-beta1**(current_idx + 1))
				vt[0].mul_(beta2).add_((1 - beta2) * (grad ** 2))
				vt_hat = vt[0] / (1-beta2**(current_idx + 1))
				mt_tilde = mt_hat / (torch.sqrt(vt_hat) + 1e-8)
				gt_tilde = grad / (torch.sqrt(vt_hat) + 1e-8)

				mt_tilde = mt_tilde.view(-1, 1)
				gt_tilde = gt_tilde.view(-1, 1)

				updates, nh, nc = opt_net(
					torch.cat([mt_tilde, gt_tilde], 1),
					hidden_states[0], cell_states[0]
				)
				result_h.append(nh)
				result_c.append(nc)

				#all_losses.append(((updates.view(*p.size()) - grad) ** 2).sum())
				up_grad = torch.mm(P.transpose(0, 1), updates)
				idx = 0
				for name, p in model.named_parameters():
					size = p.nelement()
					result_params[name] = p - up_grad[idx:idx+size].view(*p.size()) #updates.view(*p.size()) * args.scale
					result_params[name].retain_grad()        
					idx = idx + size

				hidden_states = result_h
				cell_states   = result_c

			if True:
				meta_loss = sum(all_losses)
				all_losses = []

				for name in result_params:
					param = result_params[name].detach()
					param.requires_grad_()
					rsetattr(model, name, param)

				for h in hidden_states:
					for hi in h:
						hi.detach_()
						hi.requires_grad = True

				for c in cell_states:
					for ci in c:
						ci.detach_()
						ci.requires_grad = True
					
			if train_step % args.log_interval == 0:
				elapsed = time.time() - log_start_time

				log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
									'| ms/batch {:5.2f} | meta loss {:5.2f}'.format(
									epoch, train_step, current_idx + 1, 
									elapsed * 1000 / args.log_interval, meta_loss) 
				wandb.log({"meta_eval/meta_loss": meta_loss}, step=train_step)
				print(log_str)
				log_start_time = time.time()
				avg_lm_loss.reset()
		
		
				
		eval_start_time = time.time()
		with torch.no_grad():
			valid_loss, valid_accuracy = evaluate_cifar10(model, valid_loader, args, train_step)
		wandb.log({"meta_eval/avg_val_loss": valid_loss}, step=train_step)
		wandb.log({"meta_eval/avg_accuracy": valid_accuracy}, step=train_step)

		if best_val_accuracy is None or valid_accuracy > best_val_accuracy:
			best_val_accuracy = valid_accuracy
			model_path = os.path.join(args.work_dir, 'best.pt')
			#torch.save({'optimizer_state_dict': opt_net.state_dict()}, model_path)
		elif args.cl:
			args.training_steps = training_steps[training_steps.index(args.training_steps) + 1]
		log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
							'| valid loss {:5.2f} | valid accuracy {:5.2f} | best accuracy {:5.2f} '.format(
							train_step // args.eval_interval, train_step,
							(time.time() - eval_start_time), valid_loss, valid_accuracy, best_val_accuracy)
		print(log_str)
		#model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
		#torch.save({'optimizer_state_dict': opt_net.state_dict()}, model_path)
	
	return train_step, best_val_accuracy

if __name__ == '__main__':
	args = parser.parse_args()

	set_seed(args.random_seed)
	parse_gpu(args)
	print_args(args)

	from torchvision import transforms, datasets
	traindir = os.path.join('/ssd1/xinyu/dataset/imagenet2012', 'train')
	valdir = os.path.join('/ssd1/xinyu/dataset/imagenet2012', 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])

	train_dataset = datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=128, shuffle=True,
		num_workers=32, pin_memory=True)

	test_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=128, shuffle=False,
		num_workers=32)
	#train_loader, test_loader = setup_model_dataset(args)
	
	
	args.name = f"eval_opt_resnet18_imagenet_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	if args.use_second_layer:
		args.name = args.name + "_second"
	args.optimizer_checkpoint = f"./trained_models/opt_resnet18_imagenet_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}/best.pt"
	args.work_dir = f"./trained_models/eval_opt_resnet18_imagenet_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	
	if args.use_second_layer:
		args.work_dir = args.work_dir + "_second"
	args.logging = create_exp_dir(args.work_dir)

	
	wandb.init(project=f"l2o_lora", entity="xxchen", name=args.name)
	wandb.config.update({'hidden_sz': args.hidden_sz, 'training_steps': args.training_steps, 'unroll_length': args.unroll_length})
	trainable = 0
	optimizer = None

	args.max_step = 1000000

	#scheduler = create_optimizer_scheduler(optimizer, args)
	scheduler = None

	opt_net = RNNOptimizer(preproc=False, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
	opt_net.load_state_dict(torch.load(args.optimizer_checkpoint, map_location="cpu")['optimizer_state_dict'])
	opt_net = opt_net.to(args.local_rank)

	if args.meta_optimizer == 'Adam':
		meta_optimizer = torch.optim.Adam(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'AdamW':
		meta_optimizer = torch.optim.AdamW(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'RMSprop':
		meta_optimizer = torch.optim.RMSprop(opt_net.parameters(), lr=args.meta_lr)

	train_step = 0
	best_val_accuracy = None

	#P = calculate_basis(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, test_loader, args, create_model, init_weight, train_step=train_step, epoch = 0, unroll=args.unroll_length, best_val_accuracy=best_val_accuracy)
	#assert False
	P = torch.load("resnet18_ImageNet_P.pth.tar")
	model = resnet18(num_classes=1000)
	state_dict= torch.load("resnet18_ImageNet_0.pt")
	new_state_dict = {}
	for key in state_dict:
		if key.startswith("module."):
			new_state_dict[key[7:]] = state_dict[key]
		else:
			new_state_dict[key] = state_dict[key]
	def create_model():
		resnet = resnet18(num_classes=1000)
		resnet.load_state_dict(new_state_dict)
		return resnet
	for epoch in itertools.count(start=1):
				
			#def train_validate(model, optimizer, scheduler, train_data_iter, train_corpus, valid_data_iter, valid_corpus, args, train_step = 0, epoch = 0):
		train_step, best_val_accuracy = train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, test_loader, args, create_model, None, P, train_step=train_step, epoch = epoch, unroll=args.unroll_length, best_val_accuracy=best_val_accuracy)