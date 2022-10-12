import argparse
from genericpath import exists
import time
import math
import os, sys
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

def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, mt, vt, hidden_states, cell_states, train_epoch = 5, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_ppl=None):
	model.train()
	opt_net.eval()
	meta_optimizer.zero_grad()
	avg_lm_loss = AverageMeter()
	print('start to train the model................', epoch)
	log_start_time = time.time()
	
	unroll_losses = []

	for epoch in range(train_epoch):
		for idx, data in enumerate(train_loader):
			data = {key: value for key, value in data.items()}

			_input = data['input'].to(args.device)
			_target = data['target'].to(args.device)
			_msk = data['mask'].to(args.device)
			with torch.autograd.set_detect_anomaly(True):
				_lm_logits, _lm_loss = model(_input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth) 

				_lm_loss = _lm_loss.mean() 
				wandb.log({"meta_eval/train_loss": _lm_loss.data.cpu().numpy()}, step=train_step)
				train_step += 1
				avg_lm_loss.update(_lm_loss.item())
				_loss = _lm_loss/(args.grad_acc)
				_loss.backward(retain_graph=True)
				result_params = {}
				
				result_h = {}
				result_c = {}

				result_params = {}
				count = 0
				for name, p in model.named_parameters():
					if 'adapter' in name:
						grad = p.grad.detach()
						m = mt[name] 
						v = vt[name]
						h = hidden_states[name]
						c = cell_states[name]
						m.mul_(beta1).add_((1 - beta1) * grad)
						mt[name] = m
						mt_hat = m / (1-beta1**(train_step + 1))
						v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
						vt[name] = v
						vt_hat = v / (1-beta2**(train_step + 1))
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
						result_params[name] = p - updates.view(*p.size()).detach() * args.scale
						result_params[name].retain_grad()

				hidden_states = result_h
				cell_states   = result_c

			if True:
				
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
					
			if train_step % args.log_interval == 0:
				elapsed = time.time() - log_start_time

				log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
									'| ms/batch {:5.2f} | loss {:5.2f} | avg loss {:5.2f} | ppl {:5.2f}'.format(
									epoch, train_step, idx + 1, 
									elapsed * 1000 / args.log_interval, avg_lm_loss.val, avg_lm_loss.avg, math.exp(avg_lm_loss.avg)) 
				wandb.log({"meta_train/avg_train_loss": avg_lm_loss.avg}, step=train_step)
				print(log_str)
				log_start_time = time.time()
				avg_lm_loss.reset()
				
		# evaluation interval
			if train_step % args.eval_interval  == 0:
				model_path = os.path.join(args.work_dir, f'{train_step}.pt')
				torch.save({'model_state_dict': model.state_dict()}, model_path)
				eval_start_time = time.time()
				with torch.no_grad():
					valid_loss, valid_ppl = evaluate(model, valid_loader, args, train_step)
				wandb.log({"meta_eval/avg_val_loss": valid_loss}, step=train_step)

				if best_val_ppl is None or valid_ppl < best_val_ppl:
					best_val_ppl = valid_ppl
					model_path = os.path.join(args.work_dir, 'best.pt')
					torch.save({'model_state_dict': model.state_dict()}, model_path)

				log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
									'| valid loss {:5.2f} | valid ppl {:5.2f} | best ppl {:5.2f} '.format(
									train_step // args.eval_interval, train_step,
									(time.time() - eval_start_time), valid_loss, valid_ppl, best_val_ppl)
				print(log_str)
				
	
	return train_step, best_val_ppl, mt, vt, hidden_states, cell_states

if __name__ == '__main__':
	args = parser.parse_args()
	set_seed(args.random_seed)
	parse_gpu(args)
	print_args(args)

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
	args.name = f"eval_opt_gpt2_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
	args.work_dir = f"./trained_models/eval_opt_gpt2_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
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
		
		#None, args.lr, args.weight_decay, optimizer_grouped_parameters=optimizer_grouped_parameters, correct_bias=True, adam_epislon=1.0e-6)

	if args.max_step is None:
		args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
		print('set max_step:', args.max_step)

	#scheduler = create_optimizer_scheduler(optimizer, args)
	scheduler = None
	#lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc, find_unused_parameters=True)


	opt_net = RNNOptimizer(preproc=True, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
	args.optimizer_checkpoint = f"./trained_models/opt_gpt2_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}/best.pt"
	opt_net.load_state_dict(torch.load(args.optimizer_checkpoint, map_location="cpu")['optimizer_state_dict'])
	opt_net = opt_net.to(args.local_rank)
	n_params = 0
	for name, p in lm_net.named_parameters():
		if 'adapter' in name:
			n_params += int(np.prod(p.size()))

	print(n_params)
	if args.meta_optimizer == 'Adam':
		meta_optimizer = torch.optim.Adam(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'AdamW':
		meta_optimizer = torch.optim.AdamW(opt_net.parameters(), lr=args.meta_lr)
	elif args.meta_optimizer == 'RMSprop':
		meta_optimizer = torch.optim.RMSprop(opt_net.parameters(), lr=args.meta_lr)

	train_step = 0
	best_val_ppl = None
	mt = {}
	vt = {}
	hidden_states = {}
	cell_states = {}
	for name, p in lm_net.named_parameters():
		if 'adapter' in name:
			mt[name] = torch.zeros_like(p, requires_grad=False)
			vt[name] = torch.zeros_like(p, requires_grad=False)
			hidden_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz).to(args.local_rank)]
			cell_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz).to(args.local_rank)]

	lm_net.load_weight(init_weight)
	train_step, best_val_ppl, mt, vt, hidden_states, cell_states = train_validate(lm_net, opt_net, None, scheduler, meta_optimizer, train_loader,valid_loader, args, create_model, init_weight, mt, vt, hidden_states, cell_states, train_step=train_step, train_epoch=args.max_epoch, unroll=args.unroll_length, best_val_ppl=best_val_ppl)