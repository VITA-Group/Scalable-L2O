import argparse
import time
import math
import os, sys
from opt import NAG, RMSProp, Adam

import torch
import torch.nn as nn
import torch.autograd as autograd


from resnet import resnet8
from utils import setup_model_dataset

from networks import RNNOptimizer

import wandb

torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import add_gpu_params, parse_gpu, add_other_params
from optimizer import add_optimizer_params, create_adam_optimizer_from_args

from exp_utils import create_exp_dir, set_seed, AverageMeter, rgetattr, rsetattr, evaluate, print_args, hutch_loop, evaluate_cifar10

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

def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, mt, vt, hidden_states, cell_states, train_epoch = 5, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_ppl=None):
	model.train()
	opt_net.eval()
	meta_optimizer.zero_grad()
	avg_lm_loss = AverageMeter()
	print('start to train the model................', epoch)
	log_start_time = time.time()
	all_losses = []

	params_to_calc_grad = []
	param_names = []

	for name, p in model.named_parameters():
		if to_use(name):
			param_names.append(name)
			params_to_calc_grad.append(p)
	print(f"Total param: {len(param_names)}")
	unroll_losses = []

	for epoch in range(train_epoch):
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
				all_params_gradients = torch.autograd.grad(_loss, params_to_calc_grad, retain_graph=True)

				result_h = []
				result_c = []

				for name, p, grad, m, v, h, c in zip(param_names, params_to_calc_grad, all_params_gradients, mt, vt, hidden_states, cell_states):

						m.mul_(beta1).add_((1 - beta1) * grad)
						mt_hat = m / (1-beta1**(train_step + 1))
						v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
						vt_hat = v / (1-beta2**(train_step + 1))
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
						result_params[name] = p - updates.view(*p.size()).detach() * 1e-3
						result_params[name].retain_grad()

				hidden_states = result_h
				cell_states   = result_c

			if (idx + 1) % unroll == 0:
				for name in result_params:
					param = result_params[name].detach()
					param.requires_grad_()
					rsetattr(model, name, param)

				for h in hidden_states:
					for hi in h:
						hi.detach_()

				for c in cell_states:
					for ci in c:
						ci.detach_() 
			else:
				for name in result_params:
					rsetattr(model, name, result_params[name])
			
			if (not args.late_update) or ((idx + 1) % args.unroll_length == 0):
				params_to_calc_grad = []
				for name, p in model.named_parameters():
					if to_use(name):
						params_to_calc_grad.append(p)
			
					
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
		if True:
			eval_start_time = time.time()
			with torch.no_grad():
				valid_loss, valid_accuracy = evaluate_cifar10(model, valid_loader, args, train_step)
			wandb.log({"meta_eval/val_loss": valid_loss}, step=train_step)
			wandb.log({"meta_eval/avg_accuracy": valid_accuracy}, step=train_step)


			if best_val_ppl is None or valid_accuracy < best_val_ppl:
				best_val_ppl = valid_accuracy

			log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
								'| valid loss {:5.2f} | valid accuracy {:5.2f} | best accuracy {:5.2f} '.format(
								train_step // args.eval_interval, train_step,
								(time.time() - eval_start_time), valid_loss, valid_accuracy, best_val_ppl)
			print(log_str)
	
	return train_step, best_val_ppl, mt, vt, hidden_states, cell_states

if __name__ == '__main__':
	args = parser.parse_args()
	set_seed(args.random_seed)
	parse_gpu(args)
	print_args(args)

	args.logging = create_exp_dir(args.work_dir)

	train_loader, valid_loader = setup_model_dataset(args)
	args.name = f"eval_opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	args.work_dir = f"./trained_models/opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	if args.use_second_layer:
		args.name = args.name + "_second"
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
	#lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc, find_unused_parameters=True)
	

	opt_net = RNNOptimizer(preproc=True, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
	if args.use_second_layer:
		args.optimizer_checkpoint = f"./trained_models/opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}_second/best.pt"
	else:
		args.optimizer_checkpoint = f"./trained_models/opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}/best.pt"
		
	opt_net.load_state_dict(torch.load(args.optimizer_checkpoint, map_location="cpu")['optimizer_state_dict'])
	opt_net = opt_net.to(args.local_rank)

	if args.meta_optimizer == 'Adam':
		meta_optimizer = torch.optim.Adam(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'AdamW':
		meta_optimizer = torch.optim.AdamW(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'RMSprop':
		meta_optimizer = torch.optim.RMSprop(opt_net.parameters(), lr=args.meta_lr)

	train_step = 0
	best_val_ppl = None
	mt = []
	vt = []
	hidden_states = []
	cell_states = []
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

	train_step, best_val_ppl, mt, vt, hidden_states, cell_states = train_validate(model, opt_net, None, scheduler, meta_optimizer, train_loader,valid_loader, args, create_model, init_weight, mt, vt, hidden_states, cell_states, train_step=train_step, train_epoch=args.max_epoch, unroll=args.unroll_length, best_val_ppl=best_val_ppl)