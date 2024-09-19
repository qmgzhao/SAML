from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .transcribe import transcribe as transcribe_function
from .decoding import detect_language as detect_language_function, decode as decode_function
import loralib as lora


@dataclass
class MoE:
    n_expert: int
    r: int
    lora_alpha: int
    lora_dropout: float
    temperature: float


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class AttentionPooling(nn.Module):
    def __init__(self, n_state: int):
        super().__init__()
        self.attention = nn.Linear(n_state, 1)
        self.reset_parameters()
    
    def forward(
        self,
        x: Tensor,  # [batch_size, sequence_length, d_model]
    ):
        attention_scores = self.attention(x)  # [batch_size, sequence_length, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        return torch.sum(attention_weights * x, dim=1)  # [batch_size, d_model]
    
    def reset_parameters(self):
        self.attention.reset_parameters()


class Router(nn.Module):
    def __init__(self, n_state: int, n_expert: int, temperature: float, cross_attention: bool = False):
        super().__init__()
        self.temperature = temperature
        self.attn_conv = Conv1d(n_state, n_state, kernel_size=3, stride=1, padding=1)
        self.attn_pooling = AttentionPooling(n_state)

        self.cross_attn_conv = Conv1d(n_state, n_state, kernel_size=3, stride=1, padding=1) if cross_attention else None
        self.cross_attn_pooling = AttentionPooling(n_state) if cross_attention else None

        self.linear1 = Linear(n_state * 2, n_expert) if cross_attention else Linear(n_state, n_expert)
        self.reset_parameters(cross_attention)

    def forward(
        self,
        x: Tensor,  # [batch_size, sequence_length, d_model]
        xa: Optional[Tensor] = None,
    ):
        x = self.attn_conv(x.permute(0, 2, 1))  # [batch_size, d_model, sequence_length]
        x = self.attn_pooling(x.permute(0, 2, 1))  # [batch_size, d_model]
        if xa is not None:
            xa = self.cross_attn_conv(xa.permute(0, 2, 1))
            xa = self.cross_attn_pooling(xa.permute(0, 2, 1))
            x = torch.cat((x, xa), dim=1)  # [batch_size, 2 * d_model]
        x = self.linear1(x)
        x = F.softmax(x / self.temperature, dim=-1)
        return x
    
    def reset_parameters(self, cross_attention: bool = False):
        self.linear1.reset_parameters()
        self.attn_conv.reset_parameters()
        if cross_attention:
            self.cross_attn_conv.reset_parameters()


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, moe: MoE, router: Router = None):
        super().__init__()
        self.n_head = n_head
        self.query = lora.MoELinear(n_state, n_state, moe.n_expert, r=moe.r, lora_alpha=moe.lora_alpha, lora_dropout=moe.lora_dropout, merge_weights=False)
        self.key = lora.MoELinear(n_state, n_state, moe.n_expert, bias=False, r=moe.r, lora_alpha=moe.lora_alpha, lora_dropout=moe.lora_dropout, merge_weights=False)
        self.value = lora.MoELinear(n_state, n_state, moe.n_expert, r=moe.r, lora_alpha=moe.lora_alpha, lora_dropout=moe.lora_dropout, merge_weights=False)
        self.out = lora.MoELinear(n_state, n_state, moe.n_expert, r=moe.r, lora_alpha=moe.lora_alpha, lora_dropout=moe.lora_dropout, merge_weights=False)
        
        self.router = router

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        route_prob = self.router(x) if xa is None else self.router(x, xa)
        q = self.query(x, route_prob)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa, route_prob)
            v = self.value(x if xa is None else xa, route_prob)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv, route_prob), qk

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, moe: MoE = None, attn_router: Router = None, cross_attn_router: Router = None):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, moe, router=attn_router)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head, moe, router=cross_attn_router) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, moe: MoE = None):
        super().__init__()
        # router
        self.encoder_attn_router = Router(
            n_state,
            moe.n_expert,
            moe.temperature,
        )

        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, moe=moe, attn_router=self.encoder_attn_router) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        idx = 0
        for block in self.blocks:
            # print(f"=====================================encoder {idx} block=====================================")
            idx += 1
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, moe: MoE):
        super().__init__()
        # router
        self.decoder_attn_router = Router(
            n_state,
            moe.n_expert,
            moe.temperature,
        )
        self.cross_attn_router = Router(
            n_state,
            moe.n_expert,
            moe.temperature,
            cross_attention=True,
        )

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True, moe=moe, attn_router=self.decoder_attn_router, cross_attn_router=self.cross_attn_router) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        idx = 0
        for block in self.blocks:
            # print(f"=====================================decoder {idx} block=====================================")
            idx += 1
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits

    def get_states(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits, x


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, moe: MoE):
        super().__init__()
        self.dims = dims
        self.moe = moe
        
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            self.moe,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            self.moe,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    def getstates(self, mel: torch.Tensor, tokens: torch.Tensor)  -> Dict[str, torch.Tensor]:
        return self.decoder.get_states(tokens, self.encoder(mel))
    
    def set_temperature(self, temperature: float):
        self.encoder.encoder_attn_router.temperature = temperature
        self.decoder.decoder_attn_router.temperature = temperature
        self.decoder.cross_attn_router.temperature = temperature
        print(f"temperature: {temperature}")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
