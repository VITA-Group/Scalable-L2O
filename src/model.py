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
from meta_module import MetaLinear, MetaModule, MetaModuleList, MetaSequential

import copy

import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def swish(x):
    return x * torch.sigmoid(x)

def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(MetaModule):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).        """
        super(LayerNorm, self).__init__()
        self.register_buffer('weight', torch.ones(hidden_size))
        self.register_buffer('bias', torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(MetaModule):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.register_buffer('weight', w)
        self.register_buffer('bias', torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]



class Attention(MetaModule):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

        #self.lora_dropout = config.lora_dropout

        self.config = config
        self.lora_dropout = None
        if config.lora_dropout > 0:
            self.lora_dropout = nn.Dropout(config.lora_dropout)

        self.lora_r_dropout = None
        if config.lora_r_dropout > 0:
            self.lora_r_dropout = nn.Dropout(config.lora_r_dropout)


        self.lora_attn_dim = config.lora_attn_dim 
        self.lora_attn_alpha = config.lora_attn_alpha

        if self.lora_attn_dim > 0:
            self.q_proj_adapter1 = MetaLinear(nx, self.lora_attn_dim, bias=False)
            nn.init.normal_(self.q_proj_adapter1.weight, std=0.02)
            self.q_proj_adapter2 = MetaLinear(self.lora_attn_dim, nx, bias=False)
            self.q_proj_adapter2.weight.data.zero_()

            self.v_proj_adapter1 = MetaLinear(nx, self.lora_attn_dim, bias=False)
            nn.init.normal_(self.v_proj_adapter1.weight, std=0.02)
            self.v_proj_adapter2 = MetaLinear(self.lora_attn_dim, nx, bias=False)
            self.v_proj_adapter2.weight.data.zero_()

            self.q_moe_adapter1 = None
            self.v_moe_adapter1 = None

            if self.config.lora_moe == 1:
                num_expert = self.lora_attn_dim // self.config.lora_moe_group

                self.q_moe_adapter1 = MetaLinear(nx, num_expert, bias=False)
                nn.init.normal_(self.q_moe_adapter1.weight, std=0.02)

                self.v_moe_adapter1 = MetaLinear(nx, num_expert, bias=False)
                nn.init.normal_(self.v_moe_adapter1.weight, std=0.02)


    def _attn(self, q, k, v, len_kv = None):  
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)

    def adapter_forward(self, x, weight_1, weight_2, g_weight=None):
        scale_factor = self.lora_attn_alpha / self.lora_attn_dim
        result = torch.matmul(x, weight_1.type_as(x).T)

        if self.lora_r_dropout is not None:
            result = self.lora_r_dropout(result)

        if g_weight is not None:
            g = torch.matmul(x, g_weight.weight.type_as(x).T)
            if self.config.lora_moe_act == 'sigmoid':
                g = torch.sigmoid(g) 
            elif self.config.lora_moe_act == 'tanh':
                g = torch.tanh(g) 
            elif self.config.lora_moe_act == 'relu':
                g = torch.relu(g)

            g = g * self.config.lora_moe_lambda

            if self.config.lora_moe_softmax == 1:
                g = torch.softmax(g, dim=-1)

            result = result.view(result.shape[0], result.shape[1], result.shape[2] // self.config.lora_moe_group, self.config.lora_moe_group) * g.unsqueeze(-1)
            result = result.view(result.shape[0], result.shape[1], -1)

        return torch.matmul(result, weight_2.type_as(x).T) * scale_factor

    # two level attention here.
    def forward(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        if self.lora_attn_dim > 0:
            #value += self.adapter_forward(hidden_states, self.v_proj_adapter1.weight, self.v_proj_adapter2.weight)
            
            lora_input = hidden_states
            if self.lora_dropout is not None:
                lora_input = self.lora_dropout(lora_input)

            query_delta = self.adapter_forward(lora_input, self.q_proj_adapter1.weight, self.q_proj_adapter2.weight, g_weight=self.q_moe_adapter1)

            value_delta = self.adapter_forward(lora_input, self.v_proj_adapter1.weight, self.v_proj_adapter2.weight, g_weight=self.v_moe_adapter1)
            
            query = query.contiguous() + query_delta
            value = value.contiguous() + value_delta

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(MetaModule):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(MetaModule):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(MetaModule):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = MetaModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config

        self.prefix_len = config.prefix_len
        self.infix_len = config.infix_len

        if self.prefix_len > 0:
            self.prefix_tokens = torch.arange(self.prefix_len).long()
            self.adapter_pe = nn.Embedding(self.prefix_len, config.n_embd)
            nn.init.normal_(self.adapter_pe.weight, std=0.02)

        if self.infix_len > 0:
            self.infix_tokens = torch.arange(self.infix_len).long()
            self.adapter_ie = nn.Embedding(self.infix_len, config.n_embd)
            nn.init.normal_(self.adapter_ie.weight, std=0.02)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None, len_past = None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        ###### parse input_embeds;
        if self.prefix_len > 0 or self.infix_len > 0:
            _context_token_msk = (input_ids < self.config.prefix_cursor) * 1
            _prefix_token_msk = (input_ids >= self.config.prefix_cursor) * (input_ids < self.config.infix_cursor)  * 1
            _infix_token_msk = (input_ids >= self.config.infix_cursor) * 1

            #_prefix_embeds = None
            if self.prefix_len > 0:
                _prefix_tokens = (input_ids - self.config.prefix_cursor) * _prefix_token_msk
                _prefix_embeds = self.adapter_pe(_prefix_tokens)
            
            #_infix_embeds = None
            if self.infix_len > 0:
                _infix_tokens = (input_ids - self.config.infix_cursor) * _infix_token_msk
                _infix_embeds = self.adapter_ie(_infix_tokens)
        
            input_ids = input_ids.clamp(max = self.n_vocab-1)

        inputs_embeds = self.wte(input_ids)     


        if self.prefix_len > 0 or self.infix_len > 0:
            inputs_embeds = inputs_embeds * _context_token_msk.unsqueeze(-1)

            if self.prefix_len > 0:
                inputs_embeds = inputs_embeds + _prefix_embeds * _prefix_token_msk.unsqueeze(-1)

            if self.infix_len > 0:
                inputs_embeds = inputs_embeds + _infix_embeds * _infix_token_msk.unsqueeze(-1)


        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past = layer_past, len_past=len_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(MetaModule):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = MetaLinear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            lora_attn_dim=0,
            lora_attn_alpha=128,
            lora_dropout=0.0,
            lora_r_dropout=0.0,
            
            lora_moe=0,
            lora_moe_act='linear',
            lora_moe_lambda=1.0,
            lora_moe_softmax=0,
            lora_moe_group=1,

            prefix_len=0,
            infix_len=0,
            fix_dropout=0.0,
            prefix_cursor=1000000,
            infix_cursor=2000000,
            meta_mlp_layer = 0,
            meta_inputs = [],
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout

        self.lora_moe = lora_moe
        self.lora_moe_act = lora_moe_act
        self.lora_moe_lambda = lora_moe_lambda
        self.lora_moe_softmax = lora_moe_softmax
        self.lora_moe_group = lora_moe_group

        self.prefix_len = prefix_len
        self.infix_len = infix_len
        self.fix_dropout = fix_dropout
        self.prefix_cursor = prefix_cursor
        self.infix_cursor = infix_cursor

        self.meta_mlp_layer = meta_mlp_layer
        self.meta_inputs = meta_inputs

class GPT2LMModel(MetaModule):
    def __init__(self, config):
        super(GPT2LMModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self._init_weights)
        

        if len(config.meta_inputs) > 0:
            self.meta_inputs = config.meta_inputs
            meta_input_dim = len(config.meta_inputs) * config.n_embd

            if config.meta_mlp_layer == 0:
                self.meta_mlp = None
            elif config.meta_mlp_layer == 2:
                self.meta_mlp = MetaSequential(
                                                    MetaLinear(meta_input_dim, config.n_embd),
                                                    torch.nn.Tanh(),
                                                    MetaLinear(config.n_embd, config.n_embd),
                                                    torch.nn.Tanh(),
                                                   )
            elif config.meta_mlp_layer == 3:
                self.meta_mlp = MetaSequential(
                                                    MetaLinear(meta_input_dim, config.n_embd),
                                                    torch.nn.Tanh(),
                                                    MetaLinear(config.n_embd, config.n_embd),
                                                    torch.nn.Tanh(),
                                                    MetaLinear(config.n_embd, config.n_embd),
                                                    torch.nn.Tanh(),
                                                   )
    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    # input_ids, lm_labels=pred_ids)
    def forward(self, input_ids, lm_labels=None, lm_mask=None, past=None, len_past=None, label_smooth=0.0, is_meta_index=False, meta_cls_index=None, 
                is_report_accuracy=False):
        _batch, _len = input_ids.shape
        #print(input_ids)
        hidden_states, presents = self.transformer(input_ids, past=past, len_past=len_past) #, position_ids, token_type_ids, past=past, len_past=len_past)

        if is_meta_index:
            _b = torch.arange(0, input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            # hidden_states : batch, seq, dim
            _inputs = []
            for _i in self.meta_inputs:
                _inputs.append(presents[_i][_b, meta_cls_index, :])
            _input = torch.cat(_inputs, dim=1)            

            if self.meta_mlp is not None:
                _output = self.meta_mlp(_input)
            else:
                _output = _input

            return _output

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:

            if is_report_accuracy:
                _pred_token = torch.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask

                _t1_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                _all_acc = torch.zeros(_batch, dtype=torch.float, device=input_ids.device)
                
                for _b in range(0, _batch):
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break  

                    _is_succ = True
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break

                    if _is_succ:
                        _all_acc[_b] = 1.0

                #_t1_acc = _t1_acc * 1.0 / _batch
                #_all_acc = _all_acc * 1.0 / _batch

            if label_smooth > 0.0001:
                logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -logprobs.mean(dim=-1)
                loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                loss = loss.view(_batch, _len)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

            if lm_mask is None:
                lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
            loss = loss * lm_mask 

            loss = loss.sum() / (lm_mask.sum() + 0.0001)

            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc
            else:
                return lm_logits, loss
        return lm_logits, presents
           
    def _init_weights(self, module):
        if isinstance(module, (MetaLinear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, MetaLinear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):

        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"
            
            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        self.transformer.load_state_dict(state_dict, strict=False)
        self.set_tied()
        
