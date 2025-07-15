"""
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# from utils import CfgNode as CN
from mingpt.utils import CfgNode as CN

import time
import os

from einops import einsum, rearrange
import json

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, d: int, base: int = 10_000):
        super().__init__()

        self.d = d
        self.base = base # theta's base value
        self.Y_matrix_1 = None
        self.Y_matrix_2 = None
        self.N = None

        # raise NotImplementedError("Initialization is not implemented.")

    def _build_cache(self, Y: torch.Tensor):

        mid = self.d // 2
        Y_1 = Y[:, :mid]
        Y_2 = Y[:, mid:]

        self.Y_matrix_1 = torch.cat((Y_1,Y_2),dim=1)
        self.Y_matrix_2 = torch.cat((-Y_2,Y_1),dim=1)

        # raise NotImplementedError("Rotary embeddings cache not implemented.")

    def forward(self, Y: torch.Tensor):

        b, h, t, d = Y.shape

        self._build_cache(Y)

        self.N = t
        theta_matrix = []
        for j in range(self.N):
            list = []
            for i in range(1, int(d / 2) + 1):
                list.append(j * self.base ** (-2 * (i - 1) / d))
            theta_matrix.append(list)
        theta_matrix = np.squeeze(np.array(theta_matrix))

        theta_matrix = torch.from_numpy(theta_matrix).to(Y.device).float()

        out_1 = torch.sin(theta_matrix)
        out_2 = torch.cos(theta_matrix)

        theta_1 = torch.cat((out_2, out_2), dim=1)
        theta_2 = torch.cat((out_1, out_1), dim=1)

        theta_1 = theta_1.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)
        theta_2 = theta_2.unsqueeze(0).unsqueeze(0).expand(b, h, -1, -1)

        Y_result = self.Y_matrix_1 * theta_1 + self.Y_matrix_2 * theta_2

        return Y_result

        # y is either q value or k value
        # raise NotImplementedError("Forward pass not implemented.")

class CausalSelfAttention(nn.Module):
    """
    Simple Multi Headed attention. query heads = key heads = value heads
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_query_head == 0
        self.n_head = config.n_query_head
        self.n_embd = config.n_embd

        # key, query, value projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        
        self.rope = config.rope
        if self.rope:
            self.query_rotary_pe = RotaryPositionalEmbeddings(d=config.n_embd) # Do NOT rename self.query_rotary_pe.
            self.key_rotary_pe = RotaryPositionalEmbeddings(d=config.n_embd) # Do NOT rename self.key_rotary_pe.
            # raise NotImplementedError("Attention initialization using RoPE not implemented.")
        
    def forward(self, x):
        b, t, n_embd = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values based on the input x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split the embedding dimension (n_embd) across the number of heads by introducing an additional 'h' dimension
        # reshape the query, key, value tensors to increase efficiency of matrix multiplication
        # b = batch size, t = sequence length, h = number of heads, d = n_embd / number of heads

        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_head)

        if self.rope:
            q = self.query_rotary_pe.forward(q)
            k = self.key_rotary_pe.forward(k)

        # track the memory consumed by the model
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        # compute square root of (n_embd / number of heads) to scale the dot product
        scale = math.sqrt(k.size(-1))
        # calculate the attention scores with the query and  key
        att = einsum(q, k, 'b h q d, b h k d -> b h q k') / scale
        att = att.masked_fill(self.bias[:,:,:t,:t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # matrix multiplication of attention scores and value
        y = einsum(att, v, 'b h q t, b h t d -> b h q d')
        end_memory = torch.cuda.memory_allocated()
        # rearrange the output tensor to (batch size, sequence length, n_embd)
        y = rearrange(y, 'b h q d -> b q (h d)') # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y, end_memory-start_memory

class GroupedQueryAttention(nn.Module):
    """
    An implementation of group query attention. Refer to the CausalSelfAttention class to structure your implementation.
    """

    def __init__(self, config):
        super().__init__()

        self.q_head = config.n_query_head
        self.kv_head = config.n_kv_head
        head_dim = config.n_embd // config.n_query_head

        # Keep in mind the shape that you want these projections to output.
        self.q_proj = nn.Linear(config.n_embd, config.n_embd) # Do NOT rename q_proj.
        self.k_proj = nn.Linear(config.n_embd, head_dim * config.n_kv_head)
        self.v_proj = nn.Linear(config.n_embd, head_dim * config.n_kv_head)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd) # Do NOT rename out_proj.

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        self.rope = config.rope
        if self.rope:
            self.query_rotary_pe = RotaryPositionalEmbeddings(d=config.n_embd) # Do NOT rename.
            self.key_rotary_pe = RotaryPositionalEmbeddings(d=config.n_embd) # Do NOT rename.



        # raise NotImplementedError("Initialization is not implemented.")

    def forward(self, x):
        """
        Make sure to implement RoPE for GQA if the config specifies that RoPE should be used, and keep track of memory consumed.
        """
        b, t, n_embd = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values based on the input x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        head_dim = n_embd // self.q_head

        # reshape tensors
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.q_head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.kv_head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.kv_head)

        if self.rope:
            q = self.query_rotary_pe.forward(q)
            k = self.key_rotary_pe.forward(k)

        # GQA
        repeat_factor = self.q_head // self.kv_head

        # repeat key.value
        k = k.repeat_interleave(repeat_factor, dim=1)  # [b, q_head, t, d]
        v = v.repeat_interleave(repeat_factor, dim=1)  # [b, q_head, t, d]

        # track the memory consumed by the model
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()

        # compute square root of head dimension to scale the dot product
        scale = math.sqrt(head_dim)

        # calculate the attention scores with the query and  key
        att = einsum(q, k, 'b h q d, b h k d -> b h q k') / scale
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # matrix multiplication of attention scores and value
        y = einsum(att, v, 'b h q t, b h t d -> b h q d')
        end_memory = torch.cuda.memory_allocated()
        # rearrange the output tensor to (batch size, sequence length, n_embd)
        y = rearrange(y, 'b h q d -> b q (h d)')  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y, end_memory - start_memory

        # raise NotImplementedError("Forward pass is not implemented.")
    
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if config.n_query_head != config.n_kv_head:
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        start_time = time.time()
        attn_comp, mem_consumed = self.attn(self.ln_1(x))
        end_time = time.time()
        x = x + attn_comp
        x = x + self.mlpf(self.ln_2(x))
        return x, end_time-start_time, mem_consumed

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_query_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.pretrained_folder = None
        C.n_kv_head = C.n_query_head
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.rope = config.rope

        modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        )
        if self.rope==False:
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def load_pretrained(self, model_path):
        pretrained_state_dict = torch.load(os.path.join(model_path, "model.pt"))
        old_block_size = 64
        with open(os.path.join(model_path,'config.json'), 'r') as file:
            old_config = json.load(file)
            old_block_size = old_config['data']['block_size']
        # Initialize the current model state dict
        self_state_dict = self.state_dict()

        # Loop over the pretrained state dict and update the corresponding weights
        for name, param in pretrained_state_dict.items():
            if name in self_state_dict:
                # If it's the wpe layer and sizes are different, handle separately
                if name == 'transformer.wpe.weight' and param.size(0) != self_state_dict[name].size(0):
                    # Copy the weights for the first 64 neurons
                    self_state_dict[name][:old_block_size, :] = param[:old_block_size, :]
                elif name.startswith("transformer.h.") and name.endswith(".attn.bias") and param.size()[2] != self_state_dict[name].size()[2]:
                    self_state_dict[name][:,:,:old_block_size,:old_block_size] = param
                    # Remaining weights are already randomly initialized
                else:
                    # Copy the weights for layers other than wpe
                    self_state_dict[name].copy_(param)

        # Load the updated state dict into the model
        self.load_state_dict(self_state_dict)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        attn_times = []
        mem_consumed = []
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.rope==False:
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x, attn_time, mem = block(x)
            mem_consumed.append(mem)
            attn_times.append(attn_time)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, sum(attn_times)/len(attn_times), sum(mem_consumed)/len(mem_consumed)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        attn_times = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _,attn_time,mem_consumed = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            attn_times.append(attn_time)

        return idx, sum(attn_times)/len(attn_times)
