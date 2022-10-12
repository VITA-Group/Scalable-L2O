import argparse
import time
import math
import os, sys
from opt import NAG, RMSProp, Adam

import torch
import torch.nn as nn
import torch.autograd as autograd

import torchvision

from networks import RNNOptimizer

import wandb

from resnet import resnet8
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
add_gpu_params(parser)
add_optimizer_params(parser)
add_other_params(parser)

# influence model, calculate the influence score between two samples

def to_use(name):
	return 'adapter' in name or 'fc' in name
	
def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_accuracy=None):
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
	for name, p in model.named_parameters():
		if to_use(name):
			print(name)
			mt.append(torch.zeros_like(p, requires_grad=False))
			vt.append(torch.zeros_like(p, requires_grad=False))
			hidden_states.append([torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank) for _ in range(nlayer)])
			cell_states.append([torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank) for _ in range(nlayer)])
			param_names.append(name)
			params_to_calc_grad.append(p)
	train_iter = iter(enumerate(train_loader))
	current_idx = 0
	while True:
		current_idx += 1
		try:
			idx, data = next(train_iter)
		except:
			train_iter = iter(enumerate(train_loader))
			idx, data = next(train_iter)
		if current_idx >= args.training_steps:
			break
		_input, _label = data
		_input = _input.cuda()
		_label = _label.cuda()
		with torch.autograd.set_detect_anomaly(True):
			logits = model(_input) 
			_loss = torch.nn.functional.cross_entropy(logits, _label)
			_loss = _loss.mean() 

			wandb.log({"meta_train/train_loss": _loss.data.cpu().numpy()}, step=train_step)
			train_step += 1
			avg_lm_loss.update(_loss.item())

			all_losses.append(_loss)
			result_params = {}
			all_params_gradients = torch.autograd.grad(_loss, params_to_calc_grad, retain_graph=True)

			result_h = []
			result_c = []

			for name, p, grad, m, v, h, c in zip(param_names, params_to_calc_grad, all_params_gradients, mt, vt, hidden_states, cell_states):
					m.mul_(beta1).add_((1 - beta1) * grad)
					mt_hat = m / (1-beta1**(idx + 1))
					v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
					vt_hat = v / (1-beta2**(idx + 1))
					mt_tilde = mt_hat / (torch.sqrt(vt_hat) + 1e-8)
					gt_tilde = grad / (torch.sqrt(vt_hat) + 1e-8)

					mt_tilde = mt_tilde.view(-1, 1)
					gt_tilde = gt_tilde.view(-1, 1)

					updates, nh, nc = opt_net(
						torch.cat([mt_tilde, gt_tilde], 1),
						h, c
					)

					result_h.append(nh)
					result_c.append(nc)


					result_params[name] = p - updates.view(*p.size()) * args.scale
					result_params[name].retain_grad()        
			hidden_states = result_h
			cell_states   = result_c

		if (current_idx) % unroll == 0:
			meta_loss = sum(all_losses)
			meta_loss.backward()
			meta_optimizer.step()
			meta_optimizer.zero_grad()
			#meta_scheduler.step()

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
				 

			elapsed = time.time() - log_start_time

			log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
								'| ms/batch {:5.2f} | meta loss {:5.2f}'.format(
								epoch, train_step, current_idx + 1, 
								elapsed * 1000 / args.log_interval, meta_loss) 
			wandb.log({"meta_train/meta_loss": meta_loss}, step=train_step)
			print(log_str)
			log_start_time = time.time()
			avg_lm_loss.reset()
			
				
		else:
			for name in result_params:
				rsetattr(model, name, result_params[name])
		
		if (not args.late_update) or ((current_idx) % args.unroll_length == 0):
			params_to_calc_grad = []
			for name, p in model.named_parameters():
				if to_use(name):
					params_to_calc_grad.append(p)
			
		
	# evaluation interval
	wandb.log({"training_steps": args.training_steps}, step=train_step)
	if epoch % 5 == 0:
		eval_start_time = time.time()
		with torch.no_grad():
			valid_loss, valid_accuracy = evaluate_cifar10(model, valid_loader, args, train_step)
		wandb.log({"meta_train/avg_val_loss": valid_loss}, step=train_step)
		wandb.log({"meta_train/avg_accuracy": valid_accuracy}, step=train_step)

		if best_val_accuracy is None or valid_accuracy > best_val_accuracy:
			best_val_accuracy = valid_accuracy
			model_path = os.path.join(args.work_dir, 'best.pt')
			torch.save({'optimizer_state_dict': opt_net.state_dict()}, model_path)
		elif args.cl:
			args.training_steps = training_steps[training_steps.index(args.training_steps) + 1]
		log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
							'| valid loss {:5.2f} | valid accuracy {:5.2f} | best accuracy {:5.2f} '.format(
							train_step // args.eval_interval, train_step,
							(time.time() - eval_start_time), valid_loss, valid_accuracy, best_val_accuracy)
		print(log_str)
		model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
		torch.save({'optimizer_state_dict': opt_net.state_dict()}, model_path)
	
	return train_step, best_val_accuracy

if __name__ == '__main__':
	args = parser.parse_args()

	set_seed(args.random_seed)
	parse_gpu(args)
	print_args(args)

	args.logging = create_exp_dir(args.work_dir)

	train_loader, test_loader = setup_model_dataset(args)
	
	
	args.name = f"opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	if args.use_second_layer:
		args.name = args.name + "_second"
	args.work_dir = f"./trained_models/opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	if args.use_second_layer:
		args.work_dir = args.work_dir + "_second"
	args.logging = create_exp_dir(args.work_dir)

	
	wandb.init(project=f"l2o_lora", entity="xxchen", name=args.name)
	wandb.config.update({'hidden_sz': args.hidden_sz, 'training_steps': args.training_steps, 'unroll_length': args.unroll_length})
	model = resnet8(num_classes=10, lora_dim=args.lora_dim)
	create_model = lambda: resnet8(num_classes=10, lora_dim=args.lora_dim)
	
	init_weight = model.state_dict()
	model = model.cuda()
	trainable = 0
	optimizer = None

	args.max_step = 1000000

	#scheduler = create_optimizer_scheduler(optimizer, args)
	scheduler = None

	opt_net = RNNOptimizer(preproc=True, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
	opt_net = opt_net.to(args.local_rank)

	if args.meta_optimizer == 'Adam':
		meta_optimizer = torch.optim.Adam(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'AdamW':
		meta_optimizer = torch.optim.AdamW(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'RMSprop':
		meta_optimizer = torch.optim.RMSprop(opt_net.parameters(), lr=args.meta_lr)

	train_step = 0
	best_val_accuracy = None
	for epoch in itertools.count(start=1):
				
			#def train_validate(model, optimizer, scheduler, train_data_iter, train_corpus, valid_data_iter, valid_corpus, args, train_step = 0, epoch = 0):
		train_step, best_val_accuracy = train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, test_loader, args, create_model, init_weight, train_step=train_step, epoch = epoch, unroll=args.unroll_length, best_val_accuracy=best_val_accuracy)
			
		if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
			break