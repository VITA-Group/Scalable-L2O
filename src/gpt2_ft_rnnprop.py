import argparse
import time
import math
import os, sys

from psutil import net_connections
from opt import NAG, RMSProp, Adam

import torch
import torch.nn as nn
import torch.autograd as autograd

from networks import RNNOptimizer

import wandb

torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import add_gpu_params, parse_gpu, add_other_params
from optimizer import add_optimizer_params, create_adam_optimizer_from_args


from data_utils import FT_Dataset # BinCorpus, BinLMOrderedIterator
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir, set_seed, AverageMeter, rgetattr, rsetattr, evaluate, print_args, hutch_loop

import itertools

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch GPT2 evaluation script.')

add_gpu_params(parser)
add_optimizer_params(parser)
add_other_params(parser)

# influence model, calculate the influence score between two samples

def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, meta_scheduler, train_loader, valid_loader, args, create_model, init_weight, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_ppl=None):
	training_steps = [100, 200, 300, 400, 500, 1000, 2000, 5000]
	model.train()
	opt_net.train()
	meta_optimizer.zero_grad()
	avg_lm_loss = AverageMeter()
	print('start to train the model................', epoch)
	log_start_time = time.time()
	all_losses = []
	mt = {}
	vt = {}
	hidden_states = {}
	cell_states = {}
	for name, p in model.named_parameters():
		if 'adapter' in name:
			mt[name] = torch.zeros_like(p, requires_grad=False)
			vt[name] = torch.zeros_like(p, requires_grad=False)
			hidden_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank)]
			cell_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank)]
	model = create_model().cuda()
	model.load_weight(init_weight)

	for idx, data in enumerate(train_loader):
		data = {key: value for key, value in data.items()}

		_input = data['input'].to(args.device)
		_target = data['target'].to(args.device)
		_msk = data['mask'].to(args.device)
		with torch.autograd.set_detect_anomaly(True):
			_lm_logits, _lm_loss = model(_input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth) 

			_lm_loss = _lm_loss.mean() 

			wandb.log({"meta_train/train_loss": _lm_loss.data.cpu().numpy()}, step=train_step)
			train_step += 1
			_lm_loss.backward(retain_graph=True)
			avg_lm_loss.update(_lm_loss.item())
			_loss = _lm_loss

			all_losses.append(_loss)
			result_params = {}

			result_h = {}
			result_c = {}
			for name, p in model.named_parameters():
				if 'adapter' in name:
					m, v, h, c =  mt[name], vt[name], hidden_states[name], cell_states[name]
					grad = p.grad.detach()
					m.mul_(beta1).add_((1 - beta1) * grad)
					mt[name] = m
					mt_hat = m / (1-beta1**(idx + 1))
					v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
					vt[name] = v
					vt_hat = v / (1-beta2**(idx + 1))
					mt_tilde = mt_hat / (torch.sqrt(vt_hat) + 1e-8)
					gt_tilde = grad / (torch.sqrt(vt_hat) + 1e-8)

					mt_tilde = mt_tilde.view(-1, 1)
					gt_tilde = gt_tilde.view(-1, 1)

					updates, nh, nc = opt_net(
						torch.cat([mt_tilde, gt_tilde], 1),
						h, c
					)

					result_h[name] = nh
					result_c[name] = nc


					result_params[name] = p - updates.view(*p.size()) * args.scale
					result_params[name].retain_grad()        
			hidden_states = result_h
			cell_states   = result_c

		if (idx + 1) % unroll == 0:
			meta_loss = sum(all_losses)
			meta_optimizer.step()
			meta_optimizer.zero_grad()
			#meta_scheduler.step()

			all_losses = []

			for name in result_params:
				param = result_params[name].detach()
				param.requires_grad_()
				rsetattr(model, name, param)

			for h in hidden_states:
				for hi in hidden_states[h]:
					hi.detach_()
					hi.requires_grad = True

			for c in cell_states:
				for ci in cell_states[c]:
					ci.detach_()
					ci.requires_grad = True
				
				 

			elapsed = time.time() - log_start_time

			log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
								'| ms/batch {:5.2f} | meta loss {:5.2f}'.format(
								epoch, train_step, idx + 1, 
								elapsed * 1000 / args.log_interval, meta_loss) 
			wandb.log({"meta_train/meta_loss": meta_loss}, step=train_step)
			print(log_str)
			log_start_time = time.time()
			avg_lm_loss.reset()
			
		else:
			for name in result_params:
				rsetattr(model, name, result_params[name])

		if idx > args.training_steps:
			break

	# evaluation interval
	wandb.log({"training_steps": args.training_steps}, step=train_step)
	if True:
		eval_start_time = time.time()
		with torch.no_grad():
			valid_loss, valid_ppl = evaluate(model, valid_loader, args, train_step)
		wandb.log({"meta_train/avg_val_loss": valid_loss}, step=train_step)

		if best_val_ppl is None or valid_ppl < best_val_ppl:
			best_val_ppl = valid_ppl
			model_path = os.path.join(args.work_dir, 'best.pt')
			torch.save({'optimizer_state_dict': opt_net.state_dict()}, model_path)
		elif args.cl:
			args.training_steps = training_steps[training_steps.index(args.training_steps) + 1]
		log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
							'| valid loss {:5.2f} | valid ppl {:5.2f} | best ppl {:5.2f} '.format(
							train_step // args.eval_interval, train_step,
							(time.time() - eval_start_time), valid_loss, valid_ppl, best_val_ppl)
		print(log_str)
		model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
		torch.save({'optimizer_state_dict': opt_net.state_dict()}, model_path)
	
	return train_step, best_val_ppl

if __name__ == '__main__':
	args = parser.parse_args()
	set_seed(args.random_seed)
	parse_gpu(args)
	print_args(args)

	if args.rank == 0:
		args.logging = create_exp_dir(args.work_dir)

	train_data =  FT_Dataset(args.train_data, args.train_batch_size, args.seq_len, joint_lm=args.obj=='jlm', prefix_len=args.prefix_len, infix_len=args.infix_len)   
	
	valid_data = FT_Dataset(args.valid_data, args.valid_batch_size, args.seq_len, prefix_len=args.prefix_len, infix_len=args.infix_len)
	train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed) if args.platform != 'single' else None
	valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed) if args.platform != 'single' else None

	train_loader = DataLoader(train_data, batch_size=args.train_batch_size, num_workers=0, shuffle=True, pin_memory=False, drop_last=True,
														sampler=train_sampler)
	
	valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=False,
													 sampler=valid_sampler)

	if args.model_card == 'gpt2.sm':
		config = GPT2Config(n_embd=768, n_layer=12, n_head=12, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
												prefix_len=args.prefix_len, infix_len=args.infix_len)
	elif args.model_card == 'gpt2.md':
		config = GPT2Config(n_embd=1024, n_layer=24, n_head=16, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
												prefix_len=args.prefix_len, infix_len=args.infix_len)
	elif args.model_card == 'gpt2.lg':
		config = GPT2Config(n_embd=1280, n_layer=36, n_head=20, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
												prefix_len=args.prefix_len, infix_len=args.infix_len)

	args.name = f"opt_gpt2_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
	if args.hessian:
		args.name = args.hessian + "_hessian"
	args.work_dir = f"./trained_models/opt_gpt2_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
	os.makedirs(args.work_dir, exist_ok=True)
	wandb.init(project=f"l2o_lora", entity="xxchen", name=args.name)
	wandb.config.update({'hidden_sz': args.hidden_sz, 'training_steps': args.training_steps, 'unroll_length': args.unroll_length})
	lm_net = GPT2LMModel(config)
	create_model = lambda: GPT2LMModel(config)
	if args.init_checkpoint is not None:
		print('loading model pretrained weight.')
		lm_net.load_weight(torch.load(args.init_checkpoint))  
	
	init_weight = torch.load(args.init_checkpoint)
	lm_net = lm_net.cuda()
	trainable = 0
	if args.lora_dim == 0:
		optimizer = create_adam_optimizer_from_args(lm_net, args)
		# create_adam_optimizer(lm_net, args.lr, args.weight_decay, correct_bias=True, adam_epislon=1.0e-6, no_decay_bias=args.no_decay_bias)
	else:
		optimizer = None
		for n, p in lm_net.named_parameters():
			if 'adapter' in n:
				trainable += p.numel()
				p.requires_grad = True
			else:
				p.requires_grad = False
		
		#None, args.lr, args.weight_decay, optimizer_grouped_parameters=optimizer_grouped_parameters, correct_bias=True, adam_epislon=1.0e-6)

	if args.max_step is None:
		args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
		print('set max_step:', args.max_step)

	#scheduler = create_optimizer_scheduler(optimizer, args)
	scheduler = None
	#lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc, find_unused_parameters=True)


	opt_net = RNNOptimizer(preproc=True, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
	opt_net = opt_net.to(args.local_rank)
	n_params = 0
	for name, p in lm_net.named_parameters():
		if p.requires_grad:
			n_params += int(np.prod(p.size()))

	print(n_params)
	
	if args.meta_optimizer == 'Adam':
		meta_optimizer = torch.optim.Adam(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'AdamW':
		meta_optimizer = torch.optim.AdamW(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'RMSprop':
		meta_optimizer = torch.optim.RMSprop(opt_net.parameters(), lr=args.meta_lr)

	meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, 100000, 1e-4)
	train_step = 0
	best_val_ppl = None
	for epoch in itertools.count(start=1):
			#def train_validate(model, optimizer, scheduler, train_data_iter, train_corpus, valid_data_iter, valid_corpus, args, train_step = 0, epoch = 0):
		train_step, best_val_ppl = train_validate(lm_net, opt_net, optimizer, scheduler, meta_optimizer, meta_scheduler, train_loader, valid_loader, args, create_model, init_weight, train_step=train_step, epoch = epoch, unroll=args.unroll_length, best_val_ppl=best_val_ppl)
			
		if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
			break