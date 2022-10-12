import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from collections import OrderedDict 
import argparse

from torch.optim.lr_scheduler import _LRScheduler

def add_optimizer_params(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate.')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay rate.')
    parser.add_argument('--correct_bias', action='store_true', help='correct adam bias term.')
    parser.add_argument('--adam_epislon', default=1e-6, type=float, help='adam epsilon.')
    parser.add_argument('--no_decay_bias', action='store_true', help='no weight decay on bias weigh.')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1 term.')
    parser.add_argument('--adam_beta2', default=0.98, type=float, help='adam beta2 term.')
    
    parser.add_argument('--scheduler', default='linear', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'linear', 'cycle'],
                        help='lr scheduler to use.')

    parser.add_argument('--max_step', type=int, default=None, help='upper epoch limit')

    parser.add_argument('--max_epoch', type=int, default=None, help='max epoch of training.')

    parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')

    parser.add_argument('--i_steps', type=str, default='0', help='interval_steps')
    parser.add_argument('--i_lrs', type=str, default='0.00025', help='interval_lrs')
    parser.add_argument('--meta-optimizer', type=str, default='Adam')




class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 max_lr : float = 0.1,
                 min_lr : float = 0.0,
                 warmup_steps : int = 0,
                 max_steps : int = 1,
                 alpha : float = 0.,
                 last_epoch : int = -1
        ):
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        
        self.alpha = alpha # decrease rate of max learning rate by cycle
        self.max_steps = max_steps
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.max_lr * self.last_epoch / self.warmup_steps
            return curr_lr
        else:
            _step = min(self.last_epoch, self.max_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * _step / self.max_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.max_lr * decayed # learning_rate * decayed

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: 
            param_group['lr'] = _lr


class CyclicScheduler(_LRScheduler):
    def __init__(self,
                 optimizer, # : torch.optim.Optimizer,
                 interval_steps = [],
                 interval_lrs = [],
                 last_epoch = -1,
        ):        
        self.optimizer = optimizer

        self.interval_steps = interval_steps
        self.interval_lrs = interval_lrs

        self.last_epoch = last_epoch

        super(CyclicScheduler, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.interval_lrs[0]
    
    def get_lr(self):
        for _i in range(0, len(self.interval_steps)-1):
            if self.last_epoch >= self.interval_steps[_i] and self.last_epoch < self.interval_steps[_i + 1]:
                _alpha = (self.last_epoch - self.interval_steps[_i]) / (self.interval_steps[_i+1] - self.interval_steps[_i] + 1e-6)
                if _alpha < 0:
                    _alpha = 0
                if _alpha >= 1:
                    _alpha = 1
                curr_lr = _alpha * self.interval_lrs[_i + 1] + (1.0 - _alpha) * self.interval_lrs[_i]             
                return curr_lr
        return self.interval_lrs[-1]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        #self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: #, self.get_lr()):
            param_group['lr'] = _lr



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def create_grouped_parameters(model, no_decay_bias): # args):
    if not no_decay_bias:
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()], # if not any(nd in n for nd in no_decay)],
        }]
    else:
        no_decay = ["bias", "layer_norm.weight"]

        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0,
        }]
    return optimizer_grouped_parameters


def create_adam_optimizer(model, lr, weight_decay, optimizer_grouped_parameters=None, beta1=0.9, beta2=0.98, correct_bias=True, adam_epislon=1e-6, no_decay_bias=False):
    #no_decay = ["bias"]
    if optimizer_grouped_parameters is None:
        optimizer_grouped_parameters = create_grouped_parameters(model, no_decay_bias)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(beta1, beta2), eps=adam_epislon, weight_decay=weight_decay, correct_bias=correct_bias)
    return optimizer

def create_sgd_optimizer(model, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    return optimizer
    
#def create_parameter_optimizer(, args):

def create_adam_optimizer_from_args(model, args, grouped_parameters=None):
    if grouped_parameters is None:
        grouped_parameters = create_grouped_parameters(model, args.no_decay_bias)

    optimizer = AdamW(grouped_parameters, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epislon, 
                      weight_decay=args.weight_decay, correct_bias=args.correct_bias)
    return optimizer


def create_sfw_optimizer_from_args(model, args, grouped_parameters=None):
    if grouped_parameters is None:
        grouped_parameters = create_grouped_parameters(model, args.no_decay_bias)

    optimizer = SFW(grouped_parameters, lr=args.lr)
    return optimizer

def create_adagradsfw_optimizer_from_args(model, args, grouped_parameters=None):
    if grouped_parameters is None:
        grouped_parameters = create_grouped_parameters(model, args.no_decay_bias)

    optimizer = AdaGradSFW(grouped_parameters, lr=args.lr)
    return optimizer


def create_optimizer_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=args.lr, min_lr=0.0, 
                                                  warmup_steps=args.warmup_step, max_steps=args.max_step, alpha=0)
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_step, args.max_step, last_epoch=-1)
    elif args.scheduler == 'cycle':
        if args.i_steps is not None:
            args.i_steps = [int(_i) for _i in args.i_steps.split(',')]
            args.i_lrs = [float(_i) for _i in args.i_lrs.split(',')]
        args.max_step = args.i_steps[-1]
        print('max_step is rest to', args.max_step)
        scheduler = CyclicScheduler(optimizer, interval_steps=args.i_steps, interval_lrs=args.i_lrs)
    elif args.scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_step, args.max_step, last_epoch=-1)
    else:
        # constant leanring rate.
        scheduler = None
    return scheduler

