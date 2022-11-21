import argparse
import os

import torch
import torch.nn.parallel
import torch.utils.data

from sklearn.decomposition import PCA
import numpy as np
import random
from utils import get_model

import wandb

def set_seed(seed=233): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='P(+)-SGD in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    help='model architecture (default: resnet32)')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--n_components', default=40, type=int, metavar='N',
                    help='n_components for PCA') 
parser.add_argument('--params_start', default=0, type=int, metavar='N',
                    help='which epoch start for PCA') 
parser.add_argument('--params_end', default=51, type=int, metavar='N',
                    help='which epoch end for PCA') 
parser.add_argument('--alpha', default=0, type=float, metavar='N',
                    help='lr for momentum') 
parser.add_argument('--lr', default=1, type=float, metavar='N',
                    help='lr for PSGD') 
parser.add_argument('--gamma', default=0.9, type=float, metavar='N',
                    help='gamma for momentum')
parser.add_argument('--seed', 
                    help='seed for training and initialization',
                    type=int, default=1)

parser.add_argument('--momentum', default=0.0, type=float) 

 
args = parser.parse_args()
set_seed(args.seed)
best_prec1 = 0
P = None
train_acc, test_acc, train_loss, test_loss = [], [], [], []

def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)

def get_model_grad_vec(model):
    # Return the model grad as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)

def update_grad(model, grad_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.grad.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
        idx += size

def update_param(model, param_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.data.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.data = param_vec[idx:idx+size].reshape(arr_shape)
        idx += size

def main():

    global args, best_prec1, Bk, p0, P

    print (args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(get_model(args))
    model.cuda()

    # Load sampled model parameters
    print(os.path.exists(f'{args.arch}_{args.datasets}_P.pth.tar'))
    if not os.path.exists(f'{args.arch}_{args.datasets}_P.pth.tar'):
        print ('params: from', args.params_start, 'to', args.params_end)
        W = []
        for i in range(args.params_start, args.params_end):
            ############################################################################
            # if i % 2 != 0: continue
            try:
                model.load_state_dict(torch.load(os.path.join(args.save_dir,  str(i) +  '.pt')))
            except:
                model.module.load_state_dict(torch.load(os.path.join(args.save_dir,  str(i) +  '.pt')))

            W.append(get_model_param_vec(model))
        W = np.array(W)
        print ('W:', W.shape)

        # Obtain base variables through PCA
        pca = PCA(n_components=args.n_components)
        pca.fit_transform(W)
        P = np.array(pca.components_)
        print ('ratio:', pca.explained_variance_ratio_)
        print ('P:', P.shape)

        P = torch.from_numpy(P).cuda()

        torch.save(P, f'{args.arch}_{args.datasets}_P.pth.tar')
    else:
        raise FileNotFoundError

if __name__ == "__main__":
    main()