# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import wandb
from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

from vit_models.modeling import VisionTransformer, CONFIGS
from vit_utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from vit_utils.data_utils import get_loader
from vit_utils.dist_util import get_world_size

from gpu import add_gpu_params, parse_gpu, add_other_params
from optimizer import add_optimizer_params

from exp_utils import create_exp_dir, set_seed, AverageMeter, rgetattr, rsetattr, print_args, hutch_loop

import itertools

logger = logging.getLogger(__name__)

from networks import RNNOptimizer

import time

class AverageMeter(object):
	"""Computes and stores the average and current value"""
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


def simple_accuracy(preds, labels):
	return (preds == labels).mean()


def save_model(args, model):
	model_to_save = model.module if hasattr(model, 'module') else model
	model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
	torch.save(model_to_save.state_dict(), model_checkpoint)
	logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
	# Prepare model
	config = CONFIGS[args.model_type]

	model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=1000)
	model.load_from(np.load(args.pretrained_dir))
	model.to(args.device)
	num_params = count_parameters(model)

	logger.info("{}".format(config))
	logger.info("Training parameters %s", args)
	logger.info("Total Parameter: \t%2.1fM" % num_params)
	print(num_params)
	return args, model


def count_parameters(model):
	params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return params/1000000


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def to_use(name):
	return (not ('lora' not in name and (name.startswith('transformer')))) 

def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, P, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_accuracy=None):
	training_steps = [100, 200, 300, 400, 500, 1000, 2000, 5000]
	model.train()
	opt_net.eval()
	avg_lm_loss = AverageMeter()
	print('start to train the model................', epoch)
	log_start_time = time.time()
	all_losses = []
	mt = {}
	vt = {}
	hidden_states = {}
	cell_states = {}

	model = create_model(args).cuda()
	if args.use_second_layer:
		nlayer = 2
	else:
		nlayer = 1

	for name, p in model.named_parameters():
		if to_use(name):
			mt[name] = torch.zeros_like(p, requires_grad=False)
			vt[name] = torch.zeros_like(p, requires_grad=False)
			hidden_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank)]
			cell_states[name] = [torch.zeros(p.nelement(), opt_net.hidden_sz, requires_grad=True).to(args.local_rank)]

	train_iter = iter(train_loader)
	current_idx = 0
	epoch = 0
	is_done = False
	while epoch < args.max_epoch:
		current_idx += 1
		#for name, p in model.named_parameters():
		#	print(name, p.abs().mean())
		with torch.autograd.set_detect_anomaly(True):
			accu_loss = 0
			roll = int(128 / args.batch_size)
			for i in range(roll):
				try:
					_input, _label = next(train_iter)
				except:
					train_iter = iter(train_loader)
					_input, _label = next(train_iter)
					epoch += 1

				_input = _input.cuda()
				_label = _label.cuda()
				#print(_input)
				_loss = model(_input, _label)		
				_loss.backward(retain_graph=True)	
				accu_loss += _loss.item() / roll
			train_step += 1
			result_params = {}
			#all_params_gradients = torch.autograd.grad(_loss, model.parameters(), only_inputs=False, create_graph=True)
			wandb.log({"meta_eval/train_loss": accu_loss}, step=train_step)
			all_losses.append(accu_loss)
			avg_lm_loss.update(accu_loss)
			result_h = {}
			result_c = {}

			
			for (name, p) in model.named_parameters():
				if to_use(name):
					m, v, h, c = mt[name], vt[name], hidden_states[name], cell_states[name]
					grad = p.grad.data
					m.mul_(beta1).add_((1 - beta1) * grad)
					mt[name] = m
					mt_hat = m / (1-beta1**(current_idx + 1))
					v.mul_(beta2).add_((1 - beta2) * (grad ** 2))
					vt[name] = v
					vt_hat = v / (1-beta2**(current_idx + 1))
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

					if epoch > 5:
						result_params[name] = p - updates.view(*p.size()) * args.scale * 0.1
					else:
						result_params[name] = p - updates.view(*p.size()) * args.scale

					#p.sub_(updates.view(*p.size()).detach() * args.scale)
					result_params[name].retain_grad()        
			hidden_states = result_h
			cell_states   = result_c

		
		meta_loss = sum(all_losses)
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
			
				
		if train_step % args.eval_interval == 0:
			eval_start_time = time.time()
			with torch.no_grad():
				valid_loss, valid_accuracy = valid(args, model, valid_loader, train_step)
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
	
	return train_step, best_val_accuracy


def valid(args, model, test_loader, global_step):
	# Validation!
	eval_losses = AverageMeter()

	logger.info("***** Running Validation *****")
	logger.info("  Num steps = %d", len(test_loader))
	logger.info("  Batch size = %d", args.batch_size)

	model.eval()
	all_preds, all_label = [], []
	loss_fct = torch.nn.CrossEntropyLoss()
	for step, batch in enumerate(test_loader):
		batch = tuple(t.to(args.device) for t in batch)
		x, y = batch
		with torch.no_grad():
			logits = model(x)[0]

			eval_loss = loss_fct(logits, y)
			eval_losses.update(eval_loss.item())

			preds = torch.argmax(logits, dim=-1)

		if len(all_preds) == 0:
			all_preds.append(preds.detach().cpu().numpy())
			all_label.append(y.detach().cpu().numpy())
		else:
			all_preds[0] = np.append(
				all_preds[0], preds.detach().cpu().numpy(), axis=0
			)
			all_label[0] = np.append(
				all_label[0], y.detach().cpu().numpy(), axis=0
			)

	all_preds, all_label = all_preds[0], all_label[0]
	accuracy = simple_accuracy(all_preds, all_label)

	logger.info("\n")
	logger.info("Validation Results")
	logger.info("Global Steps: %d" % global_step)
	logger.info("Valid Loss: %2.5f" % eval_losses.avg)
	logger.info("Valid Accuracy: %2.5f" % accuracy)

	return eval_losses.avg, accuracy


def main():
	parser = argparse.ArgumentParser()
	# Required parameters
	parser.add_argument("--batch-size", default=128, type=int)

	parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
						help="Which downstream task.")
	parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
												 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
						default="ViT-B_16",
						help="Which variant to use.")
	parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
						help="Where to search for pretrained ViT models.")
	parser.add_argument("--output_dir", default="output", type=str,
						help="The output directory where checkpoints will be written.")

	parser.add_argument("--img_size", default=224, type=int,
						help="Resolution size")
	parser.add_argument("--eval_every", default=100, type=int,
						help="Run prediction on validation set every so many steps."
							 "Will always run one evaluation at the end of training.")

	parser.add_argument("--learning_rate", default=3e-2, type=float,
						help="The initial learning rate for SGD.")
	parser.add_argument("--num_steps", default=10000, type=int,
						help="Total number of training epochs to perform.")
	parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
						help="How to decay the learning rate.")
	parser.add_argument("--warmup_steps", default=500, type=int,
						help="Step of training to perform learning rate warmup for.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float,
						help="Max gradient norm.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16_opt_level', type=str, default='O2',
						help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
							 "See details at https://nvidia.github.io/apex/amp.html")
	parser.add_argument('--loss_scale', type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	add_gpu_params(parser)
	add_optimizer_params(parser)
	add_other_params(parser)
	parser.add_argument("--meta_eval_eval_epoch", default=5, type=int)
	args = parser.parse_args()

	args.name = f"opt_vit_16_b_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	if args.use_second_layer:
		args.name = args.name + "_second"
	args.optimizer_checkpoint = f"trained_models/opt_vit_16_b_ul10_ts1000_hz_16_dim16_sc0.0002_al32_mlr0.01_bs64/best.pt"
	args.work_dir = f"./trained_models/eval_opt_vit_16_b_ul{args.unroll_length}_ts{args.training_steps}_hz_{args.hidden_sz}_dim{args.lora_dim}_sc{args.scale}_al{args.lora_alpha}_mlr{args.meta_lr}_bs{args.batch_size}"
	args.logging = create_exp_dir(args.work_dir)
	# Setup CUDA, GPU & distributed training
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.n_gpu = torch.cuda.device_count()

	wandb.init(project=f"l2o_lora", entity="xxchen", name=args.name)
	wandb.config.update({'hidden_sz': args.hidden_sz, 'training_steps': args.training_steps, 'unroll_length': args.unroll_length})
	
	args.device = device

	logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
						datefmt='%m/%d/%Y %H:%M:%S',
						level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
	logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
				   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

	# Model & Tokenizer Setup
	args, model = setup(args)

	def create_model(args):
		args, model = setup(args)
		return model
	opt_net = RNNOptimizer(preproc=False, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
	opt_net.load_state_dict(torch.load(args.optimizer_checkpoint, map_location="cpu")['optimizer_state_dict'])
	opt_net = opt_net.to(args.local_rank)
	from torchvision import transforms, datasets
	meta_optimizer = None
	train_step = 0
	best_val_accuracy = None
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
		train_dataset, batch_size=64, shuffle=True,
		num_workers=32, pin_memory=True)

	test_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=64, shuffle=False,
		num_workers=32)
	train_step, best_val_accuracy = train_validate(model, opt_net, None, None, meta_optimizer, train_loader, test_loader, args, create_model, None, None, train_step=train_step, epoch = 0, unroll=args.unroll_length, best_val_accuracy=best_val_accuracy)


if __name__ == "__main__":
	main()
