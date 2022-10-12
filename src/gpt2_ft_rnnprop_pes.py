import argparse
import time
import math
import os, sys
from opt import NAG, RMSProp, Adam

import torch
import torch.nn as nn
import torch.autograd as autograd
import random

import wandb

torch.set_printoptions(threshold=100000)

import numpy as np

from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_gather, distributed_sync, cleanup
from optimizer import create_adam_optimizer, create_optimizer_scheduler, add_optimizer_params, create_adam_optimizer_from_args


from data_utils import FT_Dataset # BinCorpus, BinLMOrderedIterator
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import itertools

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch GPT2 evaluation script.')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', type=str, default='../data/wikitext-103',
                    help='location of training data corpus')

parser.add_argument('--valid_data', type=str, default='../data/wikitext-103',
                    help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

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

parser.add_argument('--meta-lr', default=1e-2)

parser.add_argument('--random_scaling', action="store_true")

parser.add_argument('--name', type=str)

parser.add_argument('--use-second-layer', action="store_true")

parser.add_argument('--hessian', action="store_true")


# influence model, calculate the influence score between two samples.
import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class DMOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0, use_second_layer=False):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(4, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        self.use_second_layer = use_second_layer
    
    def noise_update(self, sigma=0.1):
        noises = []
        for p in self.parameters():
            noise = torch.randn(p.shape).to(p.device) * sigma
            p.data.add_(noise)
            noises.append(noise)
        return noises
    
    def noise_remove(self, noises):
        for p, noise in zip(self.parameters(), noises):
            p.data.sub_(noise)

    def forward(self, inp, hidden, cell):
        if self.preproc:
            inp = inp.data
            inp2 = torch.zeros(inp.size()[0], 4, device=inp.data.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0:2][keep_grads] = (torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 2:4][keep_grads] = torch.sign(inp[keep_grads]).squeeze()
            inp2[:, 0:2][~keep_grads] = -1
            inp2[:, 2:4][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = inp2
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        if self.use_second_layer:
          hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
          return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)
        else:
          return self.output(hidden0), (hidden0, ), (cell0, )

def print_args(args):
  if args.rank == 0:
    print('=' * 100)
    for k, v in args.__dict__.items():
      print('    - {} : {}'.format(k, v))
    print('=' * 100)

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

def train_validate(model, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, train_step = 0, epoch = 0, unroll = 5, beta1 = 0.9, beta2 = 0.99, best_val_ppl=None):
  training_steps = [100, 200, 300, 400, 500, 1000, 2000, 5000]
  model.train()
  opt_net.train()
  meta_optimizer.zero_grad()
  avg_lm_loss = AverageMeter()
  print('start to train the model................', epoch)
  log_start_time = time.time()

  all_losses_ever = []
  all_losses = None

  train_loader.sampler.set_epoch(epoch)
  if not args.use_second_layer:
    hidden_states = [torch.zeros(n_params, opt_net.hidden_sz).to(args.local_rank) for _ in range(1)]
    cell_states = [torch.zeros(n_params, opt_net.hidden_sz).to(args.local_rank) for _ in range(1)]
  else:
    hidden_states = [torch.zeros(n_params, opt_net.hidden_sz).to(args.local_rank) for _ in range(2)]
    cell_states = [torch.zeros(n_params, opt_net.hidden_sz).to(args.local_rank) for _ in range(2)]
  mt = torch.zeros(n_params, 1, requires_grad=True).to(args.local_rank)
  vt = torch.zeros(n_params, 1, requires_grad=True).to(args.local_rank)
  model = create_model().cuda()
  model.load_weight(init_weight)
  if args.il or args.stronger_il:
    optimizers = [NAG(lr=1e-4,nparam=n_params, rank=args.rank), Adam(lr=1e-4,nparam=n_params, rank=args.rank), RMSProp(lr=1e-4, nparam=n_params, rank=args.rank) ]
    il_index = np.random.randint(3)
  
  if args.random_scaling:
    rs = torch.from_numpy(np.exp(np.random.uniform(-3, 3, size=(n_params,1)))).to(args.local_rank).float()

  
  Ls = []
  perts = []
  pert = None
  noise = opt_net.noise_update()
  pert = noise
  
  for idx, data in enumerate(train_loader):
    if idx >= args.training_steps:
      # Actual update here
      perts_accu = [torch.zeros_like(p) for p in perts[0]]
      mul_grad_Ls = [torch.zeros_like(p) for p in perts[0]]
      for i in range(len(Ls)):
        mul_grad_L = []
        for j, grad in enumerate(perts[i]):
          perts_accu[j] += grad
          mul_grad_Ls[j] += Ls[i] * perts_accu[j]
      
      for grad in mul_grad_Ls:
        grad.div_(len(perts_accu) * (0.1) ** 2)
      for grad, p in zip(mul_grad_Ls, opt_net.parameters()):
        p.data.sub_(0.01 * grad)
          
      
      break
    data = {key: value for key, value in data.items()}

    _input = data['input'].to(args.device)
    _target = data['target'].to(args.device)
    _msk = data['mask'].to(args.device)
    with torch.autograd.set_detect_anomaly(True):
      _lm_logits, _lm_loss = model(_input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth) 

      _lm_loss = _lm_loss.mean() 

      if all_losses is None:
        all_losses = _lm_loss
      else:
        all_losses += _lm_loss
      all_losses_ever.append(_lm_loss.data.cpu().numpy())
      wandb.log({"meta_train/train_loss": _lm_loss.data.cpu().numpy()}, step=train_step)
      train_step += 1
      is_update = True if train_step % args.grad_acc == 0 else False
      avg_lm_loss.update(_lm_loss.item())
      _loss = _lm_loss/(args.grad_acc)
      flag = True
      print(_loss)
      _loss.backward()
      
      offset = 0
      result_params = {}
      if not args.use_second_layer:
        hidden_states2 = [torch.zeros(n_params, opt_net.hidden_sz).to(_lm_loss.device) for _ in range(1)]
        cell_states2 = [torch.zeros(n_params, opt_net.hidden_sz).to(_lm_loss.device) for _ in range(1)]
      else:
        hidden_states2 = [torch.zeros(n_params, opt_net.hidden_sz).to(_lm_loss.device) for _ in range(2)]
        cell_states2 = [torch.zeros(n_params, opt_net.hidden_sz).to(_lm_loss.device) for _ in range(2)]
      flag = True
      traces = []
      for name, p in model.named_parameters():
        
        if 'adapter' in name:
          cur_sz = int(np.prod(p.size()))
          # We do this so the gradients are disconnected from the graph but we still get
          # gradients from the rest
          gradients = p.grad.view(cur_sz, 1).detach().clone()
          
          if args.random_scaling:
            gradients = gradients * rs[offset:offset+cur_sz]

          ## 
          mt[offset:offset+cur_sz] = (beta1) * mt[offset:offset+cur_sz] + (1 - beta1) * gradients
          mt_hat = mt[offset:offset+cur_sz] / (1-beta1**(idx + 1))
          vt[offset:offset+cur_sz]= beta2 * vt[offset:offset+cur_sz] + (1.0 - beta2) * (gradients ** 2)
          vt_hat = vt[offset:offset+cur_sz] / (1-beta2**(idx + 1))
          mt_tilde = mt_hat / (torch.sqrt(vt_hat) + 1e-8)
          gt_tilde = gradients / (torch.sqrt(vt_hat) + 1e-8)
          updates, new_hidden, new_cell = opt_net(
            torch.cat([mt_tilde, gt_tilde], 1),
            [h[offset:offset+cur_sz] for h in hidden_states],
            [c[offset:offset+cur_sz] for c in cell_states]
          )
          #print(updates)
          for i in range(len(new_hidden)):
              hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
              cell_states2[i][offset:offset+cur_sz] = new_cell[i]
          result_params[name] = p + updates.view(*p.size()).detach() * 1e-4
          result_params[name].retain_grad()
          offset += cur_sz

    if (idx + 1) % unroll == 0:
      Ls.append(all_losses.item())
      perts.append(pert)
      if idx % 2 != 0:
        opt_net.noise_remove(pert)
        opt_net.noise_remove(pert)
      else:
        negative_pert = [-p for p in pert]
        opt_net.noise_remove(negative_pert)
        pert = opt_net.noise_update()
      all_losses = None
      
      state_dict = model.state_dict()
      for name in result_params:
        state_dict[name].copy_(result_params[name].data)
      result_params = {}
      model = create_model().cuda()
      model.load_state_dict(state_dict)
      
      hidden_states = [torch.zeros_like(v) for v in hidden_states2]
      cell_states = [torch.zeros_like(v) for v in cell_states2]
      for i in range(len(hidden_states)):
        hidden_states[i].copy_(hidden_states2[i])
        hidden_states[i] = hidden_states[i].detach()
      
      for i in range(len(cell_states)):
        cell_states[i].copy_(cell_states2[i])
        cell_states[i] = cell_states[i].detach()
        
      #mt = torch.zeros(n_params, 1, requires_grad=True).to(args.local_rank)
      #vt = torch.zeros(n_params, 1, requires_grad=True).to(args.local_rank)
      elapsed = time.time() - log_start_time

      log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches ' \
                '| ms/batch {:5.2f} | loss {:5.2f} | avg loss {:5.2f} | ppl {:5.2f}'.format(
                epoch, train_step, idx + 1, 
                elapsed * 1000 / args.log_interval, avg_lm_loss.val, avg_lm_loss.avg, math.exp(avg_lm_loss.avg)) 
      wandb.log({"meta_train/avg_train_loss": avg_lm_loss.avg}, step=train_step)
      print(log_str)
      log_start_time = time.time()
      avg_lm_loss.reset()
        
    else:
      for name in result_params:
        rsetattr(model, name, result_params[name])
      
      hidden_states = hidden_states2
      cell_states = cell_states2
      
    
  # evaluation interval
  wandb.log({"training_steps": args.training_steps}, step=train_step)
  if epoch % 5 == 0:
    eval_start_time = time.time()
    with torch.no_grad():
      valid_loss, valid_ppl = evaluate(model, valid_loader, args, train_step)
    wandb.log({"meta_train/avg_val_loss": valid_loss}, step=train_step)

    if best_val_ppl is None or valid_ppl < best_val_ppl:
      best_val_ppl = valid_ppl
    elif args.cl:
      args.training_steps = training_steps[training_steps.index(args.training_steps) + 1]
    log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
              '| valid loss {:5.2f} | valid ppl {:5.2f} | best ppl {:5.2f} '.format(
              train_step // args.eval_interval, train_step,
              (time.time() - eval_start_time), valid_loss, valid_ppl, best_val_ppl)
    print(log_str)
    model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
    state_dict = model.state_dict()
    torch.save({'model_state_dict': state_dict, 'optimizer_state_dict': opt_net.state_dict()}, model_path)
    distributed_sync(args)

  '''
  if args.rank == 0:
    model_path = os.path.join(args.work_dir, 'model.'+str(train_step)+'.pt')
    print('saving checkpoint', model_path)
    torch.save({'model_state_dict': model.state_dict()}, model_path) 
  distributed_sync(args)
  '''
  
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

  train_loader = DataLoader(train_data, batch_size=args.train_batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=True,
                            sampler=torch.utils.data.distributed.DistributedSampler(train_data, seed=args.random_seed))
  
  valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=False,
                           sampler=torch.utils.data.distributed.DistributedSampler(valid_data, seed=args.random_seed))

  if args.model_card == 'gpt2.sm':
    config = GPT2Config(n_embd=768, n_layer=12, n_head=12, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        prefix_len=args.prefix_len, infix_len=args.infix_len)
  elif args.model_card == 'gpt2.md':
    config = GPT2Config(n_embd=1024, n_layer=24, n_head=16, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        prefix_len=args.prefix_len, infix_len=args.infix_len)
  elif args.model_card == 'gpt2.lg':
    config = GPT2Config(n_embd=1280, n_layer=36, n_head=20, lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        prefix_len=args.prefix_len, infix_len=args.infix_len)
  
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
        print(f'{n}, shape: {p.shape}')
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


  opt_net = DMOptimizer(preproc=True, hidden_sz=args.hidden_sz, use_second_layer=args.use_second_layer)
  opt_net = opt_net.to(args.local_rank)
  n_params = 0
  for name, p in lm_net.named_parameters():
    print(name)
    if p.requires_grad:
      n_params += int(np.prod(p.size()))

  print(n_params)
  meta_optimizer = torch.optim.Adam(opt_net.parameters(), lr=args.meta_lr)

  try:
    train_step = 0
    best_val_ppl = None
    for epoch in itertools.count(start=1):
        
      #def train_validate(model, optimizer, scheduler, train_data_iter, train_corpus, valid_data_iter, valid_corpus, args, train_step = 0, epoch = 0):
      train_step, best_val_ppl = train_validate(lm_net, opt_net, optimizer, scheduler, meta_optimizer, train_loader, valid_loader, args, create_model, init_weight, train_step=train_step, epoch = epoch, unroll=args.unroll_length, best_val_ppl=best_val_ppl)
      
      if train_step >= args.max_step or (args.max_epoch is not None and epoch >= args.max_epoch):
        if args.rank == 0:
          print('-' * 100)
          print('End of training')
        break
  except KeyboardInterrupt:
    if args.rank == 0:
      print('-' * 100)
      print('Exiting from training early')

  distributed_sync(args)
  print('cleanup dist ...')
  cleanup(args)

