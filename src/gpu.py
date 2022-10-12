import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

def add_gpu_params(parser: argparse.ArgumentParser):
  parser.add_argument("--platform", default='single', type=str, help='platform cloud')
  parser.add_argument("--local_rank", default=0, type=int, help='local rank')
  parser.add_argument("--rank", default=0, type=int, help='rank')
  parser.add_argument("--device", default=0, type=int, help='device')
  parser.add_argument("--world_size", default=0, type=int, help='world size')
  parser.add_argument("--random_seed", default=10, type=int, help='random seed')

def add_other_params(parser):
  
  parser.add_argument('--train_data', type=str, default='../data/wikitext-103',
                    help='location of training data corpus')
  parser.add_argument('--valid_data', type=str, default='../data/wikitext-103',
                    help='location of validation data corpus')
  parser.add_argument('--train_batch_size', type=int, default=2, help='training batch size')
  parser.add_argument('--valid_batch_size', type=int, default=1, help='validation batch size')

  parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumlate')

  parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

  parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

  parser.add_argument('--model_card', default='gpt2.sm', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], help='model names.')

  parser.add_argument('--init_checkpoint', default=None, type=str, help='initial checkpoint.')

  parser.add_argument('--fp16', action='store_true', help='model fp16.')

  parser.add_argument('--log_interval', type=int, default=100, help='log interval.')

  parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval.')

  parser.add_argument('--save_interval', type=int, default=500, help='save interval.')

  parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), help='working folder.')

  parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension.')

  parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha.')

  parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], help='language model training objective.')

  parser.add_argument('--lora_dropout', default=0.0, type=float, help='dropout probability for lora layers.')

  parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing.')

  parser.add_argument('--prefix_len', default=0, type=int, help='prefix length.')

  parser.add_argument('--infix_len', default=0, type=int, help='infix length.')

  parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval.')

  parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate.')

  parser.add_argument('--roll_step', type=int, default=100, help='rolling step.')

  parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs.')

  parser.add_argument('--unroll_length', type=int, default=5, help='num_unroll')

  parser.add_argument('--training_steps', type=int, default=100, help='training_steps')

  parser.add_argument('--hidden_sz', type=int, default=20, help='hidden_size')

  parser.add_argument('--il_weight', type=float, default=1e-3, help='il')

  parser.add_argument('--cl', action="store_true")

  parser.add_argument('--il', action="store_true")

  parser.add_argument('--stronger-il', action="store_true")

  parser.add_argument('--meta-lr', default=1e-2, type=float)

  parser.add_argument('--random_scaling', action="store_true")

  parser.add_argument('--name', type=str)

  parser.add_argument('--use-second-layer', action="store_true")

  parser.add_argument('--hessian', action="store_true")

  parser.add_argument('--scale', default=1e-4, type=float)

  parser.add_argument('--optimizer_checkpoint', default=None, type=str)

  parser.add_argument('--late-update', action="store_true")




def distributed_opt(args, model, opt, grad_acc=1, find_unused_parameters=False):
  if args.platform == 'azure':
    args.hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    opt = args.hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters(), backward_passes_per_step=grad_acc)
  elif args.platform == 'philly' or args.platform == 'k8s' or args.platform == 'local':
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, 
                                                      find_unused_parameters=find_unused_parameters, broadcast_buffers=False)
  return model, opt


def distributed_gather(args, tensor):
  g_y = [torch.zeros_like(tensor) for _ in range(args.world_size)]
  torch.distributed.all_gather(g_y, tensor, async_op=False)
  return torch.stack(g_y)

def distributed_sync(args):
  if args.platform == 'azure':
    args.hvd.allreduce(torch.tensor(0), name='barrier')
  else:
    args.dist.barrier()

def parse_gpu(args):
  if args.platform == 'single':
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.rank = local_rank
    args.device = device
    args.world_size = 1
    args.dist = None
  if args.platform == 'local':
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.rank = local_rank
    args.device = device
    args.world_size = torch.distributed.get_world_size()
    args.dist = dist
    
  elif args.platform == 'azure':
    import horovod.torch as hvd
    hvd.init()
    print('azure hvd rank', hvd.rank(), 'local rank', hvd.local_rank())
    local_rank = hvd.local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)                         
    rank = hvd.rank()
    world_size = hvd.size()
    
    args.local_rank = local_rank
    args.rank = rank
    args.device = device
    args.world_size = world_size
    args.hvd = hvd

  elif args.platform == 'philly':
    local_rank = args.local_rank 
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda", local_rank)   

    args.rank = rank
    args.device = device
    args.world_size = world_size
    args.dist = dist
  elif args.platform == 'k8s':
    master_uri = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    args.local_rank = local_rank
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    rank = world_rank
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend='nccl',
        init_method=master_uri,
        world_size=world_size,
        rank=world_rank,
    )
    device = torch.device("cuda", local_rank)                    
    args.rank = rank
    args.device = device
    args.world_size = world_size
    args.dist = dist
  print("myrank: ", args.rank, 'local_rank: ', args.local_rank, " device_count: ", torch.cuda.device_count(), "world_size:", args.world_size)
  
  
def cleanup(args):
  if args.platform == 'k8s' or args.platform == 'philly':
    args.dist.destroy_process_group()

