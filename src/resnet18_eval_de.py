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

from resneto import resnet18
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
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--dataset", default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])

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

def P_SGD(model, optimizer, grad, oldf, X, y):
	# P_plus_BFGS algorithm

	global rho, sigma, Bk, sk, gk_last, grad_res_momentum, gamma, alpha, search_times

	gk = torch.mm(P, grad.reshape(-1,1))

	grad_proj = torch.mm(P.transpose(0, 1), gk)
	grad_res = grad - grad_proj.reshape(-1)

	# Update the model grad and do a step
	update_grad(model, grad_proj)
	optimizer.step()



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
	
def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, P, mt, vt, hidden_states, cell_states, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_accuracy=None):
	training_steps = [100, 200, 300, 400, 500, 1000, 2000, 5000]
	model.train()
	opt_net.eval()
	meta_optimizer.zero_grad()
	avg_lm_loss = AverageMeter()
	print('start to train the model................', epoch)
	log_start_time = time.time()
	all_losses = []

	for idx, data in enumerate(train_loader):
		_input, _label = data
		_input = _input.cuda()
		_label = _label.cuda()
		with torch.autograd.set_detect_anomaly(True):
			logits = model(_input) 
			_loss = torch.nn.functional.cross_entropy(logits, _label)
			_loss = _loss.mean() 

			wandb.log({"meta_eval/train_loss": _loss.data.cpu().numpy()}, step=train_step)
			train_step += 1
			avg_lm_loss.update(_loss.item())

			
			result_params = {}
			_loss.backward()
			flatten = []
			for name, p in model.named_parameters():
				#print(name, p.abs().mean())
				flatten.append(p.grad.data.reshape(-1,1))

			flatten = torch.cat(flatten, 0)
			all_losses.append(_loss)
			result_h = []
			result_c = []

			grad = torch.mm(P, flatten)

			#for name, p, grad, m, v, h, c in zip(param_names, model.parameters(), all_params_gradients, mt, vt, hidden_states, cell_states):
			mt[0].mul_(beta1).add_((1 - beta1) * grad)
			mt_hat = mt[0] / (1-beta1**(train_step + 1))
			vt[0].mul_(beta2).add_((1 - beta2) * (grad ** 2))
			vt_hat = vt[0] / (1-beta2**(train_step + 1))
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
			sgd_grad = torch.mm(P.transpose(0, 1), mt[0])
			idx = 0
			for name, p in model.named_parameters():
				size = p.nelement()

				result_params[name] = p - up_grad[idx:idx+size].view(*p.size())
						
				result_params[name].retain_grad()        
				idx = idx + size

			hidden_states = result_h
			cell_states   = result_c

		if True:
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
				 

		else:
			for name in result_params:
				rsetattr(model, name, result_params[name])
		
		if train_step % args.log_interval == 0:
			elapsed = time.time() - log_start_time

			log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
								'| ms/batch {:5.2f} | loss {:5.2f} | avg loss {:5.2f} | ppl {:5.2f}'.format(
								epoch, train_step, idx + 1, 
								elapsed * 1000 / args.log_interval, avg_lm_loss.val, avg_lm_loss.avg, math.exp(avg_lm_loss.avg)) 
			wandb.log({"meta_eval/avg_train_loss": avg_lm_loss.avg}, step=train_step)
			print(log_str)
			log_start_time = time.time()
			avg_lm_loss.reset()
	# evaluation interval
	wandb.log({"training_steps": args.training_steps}, step=train_step)
	if True:
		eval_start_time = time.time()
		with torch.no_grad():
			valid_loss, valid_accuracy = evaluate_cifar10(model, valid_loader, args, train_step)
		wandb.log({"meta_eval/avg_val_loss": valid_loss}, step=train_step)
		wandb.log({"meta_eval/avg_accuracy": valid_accuracy}, step=train_step)

		if best_val_accuracy is None or valid_accuracy > best_val_accuracy:
			best_val_accuracy = valid_accuracy
		elif args.cl:
			args.training_steps = training_steps[training_steps.index(args.training_steps) + 1]
		log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
							'| valid loss {:5.2f} | valid accuracy {:5.2f} | best accuracy {:5.2f} '.format(
							train_step // args.eval_interval, train_step,
							(time.time() - eval_start_time), valid_loss, valid_accuracy, best_val_accuracy)
		print(log_str)
	
	return train_step, best_val_accuracy, model, mt, vt, hidden_states, cell_states




if __name__ == '__main__':
	args = parser.parse_args()

	set_seed(args.random_seed)
	parse_gpu(args)
	print_args(args)


	train_loader, test_loader = setup_model_dataset(args, dataset=args.dataset)
	
	args.name = f"opt_eval_resnet18_{args.dataset}_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_sc{args.scale}_mlr{args.meta_lr}_bs{args.batch_size}"
	if args.use_second_layer:
		args.name = args.name + "_second"
	args.work_dir = f"./trained_models/eval_opt_resnet18_{args.dataset}_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_sc{args.scale}_mlr{args.meta_lr}_bs{args.batch_size}"
	args.optimizer_checkpoint = f"./trained_models/opt_resnet18_{args.dataset}_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_sc{args.scale}_mlr{args.meta_lr}_bs{args.batch_size}/best.pt"
	if args.use_second_layer:
		args.work_dir = args.work_dir + "_second"
	

	
	wandb.init(project=f"l2o_lora", entity="xxchen", name=args.name)
	wandb.config.update({'hidden_sz': args.hidden_sz, 'training_steps': args.training_steps, 'unroll_length': args.unroll_length})
	if args.dataset == 'CIFAR10':
		num_classes = 10
	else:
		num_classes = 100

	model = resnet18(num_classes=num_classes)
	if args.use_second_layer:
		nlayer=2
	else:
		nlayer=1
	
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

	P = torch.load(f"resnet18_{args.dataset}_P.pth.tar")
	state_dict= torch.load(f"resnet18_{args.dataset}_0.pt")

	new_state_dict = {}
	for key in state_dict:
		if key.startswith("module."):
			new_state_dict[key[7:]] = state_dict[key]
	def create_model():
		resnet = resnet18(num_classes=num_classes)
		resnet.load_state_dict(new_state_dict)
		return resnet
	model = create_model().cuda()
	mt = []
	vt = []
	hidden_states = []
	cell_states = []
	mt.append(torch.zeros(P.shape[0], 1, requires_grad=False).cuda())
	vt.append(torch.zeros(P.shape[0], 1, requires_grad=False).cuda())
	hidden_states.append([torch.zeros(P.shape[0], opt_net.hidden_sz, requires_grad=True).to(args.local_rank) for _ in range(nlayer)])
	cell_states.append([torch.zeros(P.shape[0], opt_net.hidden_sz, requires_grad=True).to(args.local_rank) for _ in range(nlayer)])
	for epoch in itertools.count(start=1):
				
			#def train_validate(model, optimizer, scheduler, train_data_iter, train_corpus, valid_data_iter, valid_corpus, args, train_step = 0, epoch = 0):
		train_step, best_val_accuracy, model, mt, vt, hidden_states, cell_state = train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, test_loader, args, create_model, None, P, mt, vt, hidden_states, cell_states, train_step=train_step, epoch = epoch, unroll=args.unroll_length, best_val_accuracy=best_val_accuracy)
			
		if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
			break