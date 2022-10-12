import torch

class Optimizer:
    def __init__(self, nparam, lr, rank):
        self.lr = lr
        self.nparam = nparam
        self.rank = rank
        self.params = []
    def derive(self, x, offset, cur_sz, step):
        raise NotImplementedError
    
    def reset(self):
        for name in self.params:
            getattr(self, name).zero_()


class Adam(Optimizer):
    
    def __init__(self, nparam, lr, rank):
        super().__init__(nparam, lr, rank)
        self.mt = torch.zeros(nparam, 1).to(self.rank)
        self.vt = torch.zeros(nparam, 1).to(self.rank)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.params = ['mt', 'vt']

    def derive(self, x, offset, cur_sz, idx):
        self.mt[offset:offset+cur_sz] = (self.beta1) * self.mt[offset:offset+cur_sz] + (1 - self.beta1) * x
        mt_hat = self.mt[offset:offset+cur_sz] / (1-self.beta1**(idx + 1))
        self.vt[offset:offset+cur_sz]= self.beta2 * self.vt[offset:offset+cur_sz] + (1.0 - self.beta2) * (x ** 2)
        vt_hat = self.vt[offset:offset+cur_sz] / (1-self.beta2**(idx + 1))
        mt_tilde = mt_hat / (torch.sqrt(vt_hat) + 1e-8)
        return -mt_tilde * self.lr
    def reset(self):
        self.mt.zero_()
        self.vt.zero_()

class RMSProp(Optimizer):
    
    def __init__(self, nparam, lr, rank):
        super().__init__(nparam, lr, rank)
        self.vt = torch.zeros(nparam, 1).to(self.rank)
        self.beta2 = 0.99
        self.params = ['vt']

    def derive(self, x, offset, cur_sz, idx):
        self.vt[offset:offset+cur_sz]= self.beta2 * self.vt[offset:offset+cur_sz] + (1.0 - self.beta2) * (x ** 2)
        mt_tilde = x / (torch.sqrt(self.vt[offset:offset+cur_sz]) + 1e-8)
        return -mt_tilde * self.lr
    

class NAG(Optimizer):
    
    def __init__(self, nparam, lr, rank):
        super().__init__(nparam, lr, rank)
        self.mt = torch.zeros(nparam, 1).to(self.rank)
        self.beta1 = 0.9
        self.params = ['mt']

    def derive(self, x, offset, cur_sz, idx):
        self.mt[offset:offset+cur_sz] = self.beta1 * self.mt[offset:offset+cur_sz] + (1.0 - self.beta1) * x
        return -self.mt[offset:offset+cur_sz] * self.lr