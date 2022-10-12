import argparse
import time
import math
import os, sys
from opt import NAG, RMSProp, Adam

import torch
import torch.nn as nn
import torch.autograd as autograd


from resnet_origin import resnet8
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

add_gpu_params(parser)
add_optimizer_params(parser)
add_other_params(parser)

# influence model, calculate the influence score between two samples
def to_use(name):
	return True

def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, mt, vt, train_epoch = 5, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_ppl=None):
	model.train()
	avg_lm_loss = AverageMeter()
	print('start to train the model................', epoch)
	log_start_time = time.time()
	all_losses = []


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
				_loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				
					
			if train_step % args.log_interval == 0:
				elapsed = time.time() - log_start_time

				log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
									'| ms/batch {:5.2f} | loss {:5.2f} | avg loss {:5.2f} '.format(
									epoch, train_step, idx + 1, 
									elapsed * 1000 / args.log_interval, avg_lm_loss.val, avg_lm_loss.avg) 
				wandb.log({"meta_eval/avg_train_loss": avg_lm_loss.avg}, step=train_step)
				print(log_str)
				log_start_time = time.time()
				avg_lm_loss.reset()
				
			# evaluation interval
			if train_step % args.eval_interval  == 0:
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
	
	return train_step, best_val_ppl, mt, vt

if __name__ == '__main__':
	args = parser.parse_args()
	set_seed(args.random_seed)
	parse_gpu(args)
	print_args(args)

	args.logging = create_exp_dir(args.work_dir)

	train_loader, valid_loader = setup_model_dataset(args)
	args.name = f"eval_opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
	args.work_dir = f"./trained_models/opt_resnet8_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}"
	args.logging = create_exp_dir(args.work_dir)

	wandb.init(project=f"l2o_lora", entity="xxchen", name='resnet8_baseline')
	model = resnet8(num_classes=10)
	create_model = lambda: resnet8(num_classes=10)

	init_weight = model.state_dict()
	model = model.cuda()
	trainable = 0

	args.max_step = 1000000

	#scheduler = create_optimizer_scheduler(optimizer, args)
	scheduler = None
	#lm_net, optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc, find_unused_parameters=True)
	

	optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)

	train_step = 0
	best_val_ppl = None
	mt = []
	vt = []


	train_step, best_val_ppl, mt, vt = train_validate(model, None, optimizer, scheduler, None, train_loader,valid_loader, args, create_model, init_weight, mt, vt, train_step=train_step, train_epoch=args.max_epoch, unroll=args.unroll_length, best_val_ppl=best_val_ppl)