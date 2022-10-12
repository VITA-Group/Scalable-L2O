#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import functools
import os, shutil
import numpy as np
import random
import torch


def logging(s, log_path, print_=True, log_=True):
		if print_:
				print(s)
		if log_:
				with open(log_path, 'a+') as f_log:
						f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
		return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
		if debug:
				print('Debug Mode : no experiment dir created')
				return functools.partial(logging, log_path=None, log_=False)

		if not os.path.exists(dir_path):
				os.makedirs(dir_path)

		print('Experiment dir : {}'.format(dir_path))
		if scripts_to_save is not None:
				script_path = os.path.join(dir_path, 'scripts')
				if not os.path.exists(script_path):
						os.makedirs(script_path)
				for script in scripts_to_save:
						dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
						shutil.copyfile(script, dst_file)

		return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

def save_checkpoint(model, optimizer, path, epoch):
		torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
		torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))

def set_seed(seed):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(seed)
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)

class AverageMeter(object):
	"""Computes and stores the average and current value
		 Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

import functools

def rsetattr(obj, attr, val):
		pre, _, post = attr.rpartition('.')
		return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
		def _getattr(obj, attr):
				return getattr(obj, attr, *args)
		return functools.reduce(_getattr, [obj] + attr.split('.'))


def print_args(args):
	if args.rank == 0:
		print('=' * 100)
		for k, v in args.__dict__.items():
			print('    - {} : {}'.format(k, v))
		print('=' * 100)


import time
import math
def accuracy(output, target, topk=(1,)):
		"""Computes the precision@k for the specified values of k"""
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
				correct_k = correct[:k].view(-1).float().sum(0)
				res.append(correct_k.mul_(100.0 / batch_size))
		return res

def evaluate(model, valid_loader, args, step):
	model.eval()
	total_loss = 0.
	start_time = time.time()

	avg_lm_loss = AverageMeter()

	with torch.no_grad():
		for idx, data in enumerate(valid_loader):
			data = {key: value for key, value in data.items()}

			_input = data['input'].to(args.device)
			_target = data['target'].to(args.device)
			_msk = data['mask'].to(args.device)

			_lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
			loss = _loss.mean() 
			
			avg_lm_loss.update(loss.item())
			#wandb.log({"meta_train/val_loss": loss.item()})
			if idx % 100 == 0:
				print('eval samples:', idx, 'loss:', loss.float())

		total_time = time.time() - start_time
		print('average loss', avg_lm_loss.avg)
	return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)

def evaluate_cifar10(model, valid_loader, args, step):
	model.eval()
	total_loss = 0.
	start_time = time.time()

	avg_lm_loss = AverageMeter()
	avg_accuracy = AverageMeter()
	with torch.no_grad():
		for idx, data in enumerate(valid_loader):
			_input, _target = data
			_input = _input.cuda()
			_target = _target.cuda()
			try:
				_lm_logits = model(_input) 
				loss = torch.nn.functional.cross_entropy(_lm_logits, _target) 
			except:
				loss = torch.nn.functional.cross_entropy(_lm_logits[0], _target) 
			accuracy_batch = accuracy(_lm_logits, _target)
			avg_accuracy.update(accuracy_batch[0].item())
			avg_lm_loss.update(loss.item())
			#wandb.log({"meta_train/val_loss": loss.item()})
			print('eval samples:', idx, 'loss:', loss.float())

		total_time = time.time() - start_time
		print('average loss', avg_lm_loss.avg)
		print('average accuracy', avg_accuracy.avg)

	return avg_lm_loss.avg, avg_accuracy.avg

from torch import autograd
def hutch_loop(loss, params, niter):
		g = autograd.grad(loss, params, create_graph=True)
		out = 0.0
		for i in range(niter):
				vsplit = [torch.randn_like(p) for p in params]
				Hv = autograd.grad(g, params, vsplit, retain_graph=True)
				out += group_product(Hv, vsplit)
		return out/niter

def group_product(xs, ys):
		return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])