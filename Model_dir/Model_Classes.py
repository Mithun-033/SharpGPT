import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#=================================================================================
#  RopeEmbedding Block
#=================================================================================

class RopeEmbedding(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to q and k tensors.

    This implementation follows the standard RoPE formulation used in
    transformer variants where half the head dimensions are treated as
    interleaved sin/cos pairs and rotated by position-dependent angles.

    Behaviour and shapes
    - Expects `q` and `k` with shape (B, T, num_heads, head_size).
    - `head_size` must be even (pairs of dimensions).
    - Uses `config.cwl` as the maximum context/window length to precompute
      inverse frequencies; however frequencies are computed lazily per-call
      based on actual `T` so the buffer only stores `inv_freq`.

    Implementation notes
    - `inv_freq` is registered as a buffer so it moves with the module
      between devices and is saved/loaded with the state dict.
    - `_apply_rope` performs the interleaved rotation on the last dimension.
    """

    def __init__(self, config):
        super().__init__()

        # RoPE needs an even head dimension so we can form (cos,sin) pairs.
        assert config.head_size % 2 == 0, "RoPE requires an even head size"

        self.head_size = config.head_size
        # saved for compatibility / external checks; not strictly required here
        self.max_seq_len = getattr(config, "cwl", None)

        # inverse frequencies for rotary embeddings (every other dim)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_size, 2, dtype=torch.float32) / self.head_size)
        )

        # register as buffer so it moves with `.to(device)` and is included in state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, x, cos, sin):
        """Apply precomputed cos/sin to tensor `x`.

        x: Tensor[..., head_size] where head_size is even (interleaved dim pairs).
        cos, sin: tensors broadcastable to x[..., head_size/2]

        Returns rotated tensor with same shape as `x`.
        """
        # split interleaved even/odd dims
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # rotate: (x_even, x_odd) -> (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
        x_rope = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)

        # flatten the last two dims back to head_size
        return x_rope.flatten(-2)

    def forward(self, q, k):
        """Apply RoPE to `q` and `k` and return rotated (q,k).

        Args:
            q: Tensor of shape (B, T, num_heads, head_size)
            k: Tensor of shape (B, T, num_heads, head_size)

        Returns:
            (q_rot, k_rot) with same shapes as inputs.
        """
        seq_len = q.size(1)
        device = q.device

        # positions [T] and outer-product with inv_freq -> [T, head_size/2]
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

        # expand to shape (1, T, 1, head_size/2) so it broadcasts over (B, T, num_heads, head_half)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        return q, k

#=================================================================================
#  Multi-Head Attention Block
#=================================================================================

class MultiHeadAttention(nn.Module):
    ''' Class definition of Multi-Head Attention with RMSNorm, RoPE, 
    scaled dot product attention and XSA 
    '''
    def __init__(self,config):
        '''
        Initialising the Multi-Head Attention.
        Args:
            config :-
                -> d_model (int)
                -> head_size (int)
                -> num_heads (int)
        '''
        super().__init__()

        self.config=config
        assert config.head_size*config.num_heads==config.d_model," Dims don't match, check Config params "

        self.qkv=nn.Linear(config.d_model,3*config.head_size*config.num_heads,bias=False)
        self.q_norm=nn.RMSNorm(config.head_size,eps=1e-5)
        self.k_norm=nn.RMSNorm(config.head_size,eps=1e-5)

        self.proj=nn.Linear(config.head_size*config.num_heads,config.d_model,bias=False)
        self.rope=RopeEmbedding(config)
        self.n_head = config.num_heads
        self.head_size = config.head_size

    def forward(self,x):
        ''' 
        Calling the forward pass on the attention layers.
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns: 
            Tensor of shape (B,T,C)
        '''
        B,T,C=x.shape

        qkv=self.qkv(x)
        # split into q,k,v triplet along last dim
        q,k,v = qkv.chunk(3, dim=-1)

        q=q.view(B,T,self.config.num_heads,self.config.head_size)
        k=k.view(B,T,self.config.num_heads,self.config.head_size)
        v=v.view(B,T,self.config.num_heads,self.config.head_size)

        q=self.q_norm(q)
        k=self.k_norm(k)

        q,k=self.rope(q,k)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        Vn=F.normalize(v,dim=-1)
        out=out - (out * Vn).sum(dim=-1, keepdim=True)*Vn

        out=out.transpose(1,2).reshape(B,T,self.n_head*self.head_size)

        return self.proj(out)

#=================================================================================
#  MLP Layer
#=================================================================================

class Mlp(nn.Module):
    ''' 
    A Multi-Layer Perceptron class definition with Up and Down projecton linear layers 
    and ReLU square activation.
    '''
    def __init__(self,config):
        ''' 
        Initialising Mlp.

        Args :
            Config DataClass :-
                -> d_model (int)
                -> hidden_size (int)
        '''
        super().__init__()
        self.up_proj=nn.Linear(config.d_model,config.hidden,bias=False)
        self.down_proj=nn.Linear(config.hidden,config.d_model,bias=False)


    def forward(self,x):
        '''
        Calling the forward on Linear and activation functions
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns:
            Tensor of shape (B,T,C)
        '''

        x=self.up_proj(x)
        x=F.relu(x).square()
        return self.down_proj(x)
    
#=================================================================================
#  Transformer Block 
#=================================================================================
    
class Block(nn.Module):
    ''' Class Definition of a Transformer Block '''
    def __init__(self,config):
        '''
        Initialisng the Block.

        Args:
            config :-
                -> d_model (int)
                -> num_layer (int)
        '''
        super().__init__()
        self.PreNorm1=nn.RMSNorm(config.d_model,eps=1e-5)
        self.attention=MultiHeadAttention(config)
        self.PreNorm2=nn.RMSNorm(config.d_model,eps=1e-5)
        self.Mlp=Mlp(config)

        self.scale=1/(math.sqrt(2*config.num_layers))

    def forward(self,x):
        ''' Calling the PreNorms, attention and MLP layers on input and scaling the outputs before adding to the residual stream
        Args:
            x (Tensor): Input Tensor of shape (B,T,C)
        Returns:
            Tensor of shape (B,T,C)
        '''
        x=x+self.scale*self.attention(self.PreNorm1(x))
        x=x+self.scale*self.Mlp(self.PreNorm2(x))
        return x
    
#=================================================================================
#  GPT Model
#=================================================================================

class GPT(nn.Module):
    ''' Class definition of the GPT model with token embedding, transformer blocks, final layer norm and language modelling head.'''
    def __init__(self,config):
        '''
        Initialising the GPT model.
        Args:
            config :-
                -> d_model (int)
                -> num_layer (int)
                -> head_size (int)
                -> num_heads (int)
                -> hidden (int)
                -> vocab_size (int)
        '''
        super().__init__()

        self.embed=nn.Embedding(config.vocab_size,config.d_model)
        self.blocks=nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.final_norm=nn.RMSNorm(config.d_model,eps=1e-5)
        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)

        self.lm_head.weight=self.embed.weight # Weights tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ''' Initialisation of weights for different layers.
        Linear and Embedding layers are initialised with normal distribution with mean 0 
        and std 0.02, while RMSNorm layers are initialised with ones.
        '''
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.RMSNorm):
            if hasattr(module, 'weight'):
                nn.init.ones_(module.weight)

    def forward(self,x):
        ''' Calling the forward pass on the GPT model.
        Args:
            x (Tensor): Input Tensor of shape (B,T)
        Returns:
            Tensor of shape (B,T,vocab_size)
        '''
        x=self.embed(x)    # x.shape = (B,T,C)
        for block in self.blocks:
            x=block(x)
        x=self.final_norm(x)
        x=self.lm_head(x)
        return x



            


