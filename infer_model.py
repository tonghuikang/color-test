from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization layer."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Normalized tensor of shape (batch_size, seq_len, dim)
        """
        assert hidden_states.dim() == 3
        assert hidden_states.size(-1) == self.weight.size(0)

        norm = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.eps)
        result = hidden_states * norm * self.weight

        assert result.shape == hidden_states.shape
        return result


class RotaryProjection(nn.Module):
    """Rotary projection."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary position embedding to query and key tensors.

        Args:
            query_states: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key_states: Key tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim)
            seq_len: Optional sequence length override

        Returns:
            Tuple of (query_embed, key_embed) with same shapes as input query and key
        """
        assert query_states.dim() == 4
        assert key_states.dim() == 4
        assert query_states.shape[-1] == key_states.shape[-1] == self.dim

        if seq_len is None:
            seq_len = query_states.shape[-2]

        t = torch.arange(
            seq_len,
            device=query_states.device,
            dtype=torch.float32
            if self.inv_freq.dtype == torch.float32
            else torch.float64,
        )
        freqs = torch.outer(t, self.inv_freq)  # type: ignore
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = (
            emb.cos().to(dtype=query_states.dtype),
            emb.sin().to(dtype=query_states.dtype),
        )

        assert cos.shape == (seq_len, self.dim)
        assert sin.shape == (seq_len, self.dim)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        query_embed = (query_states * cos) + (self.rotate_half(query_states) * sin)
        key_embed = (key_states * cos) + (self.rotate_half(key_states) * sin)

        assert query_embed.shape == query_states.shape
        assert key_embed.shape == key_states.shape
        return query_embed, key_embed


    def rotate_half(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dims of the input.

        Args:
            hidden_states: Input tensor of shape (..., head_dim)

        Returns:
            Rotated tensor of shape (..., head_dim)
        """
        assert hidden_states.shape[-1] % 2 == 0

        half1 = hidden_states[..., : hidden_states.shape[-1] // 2]
        half2 = hidden_states[..., hidden_states.shape[-1] // 2 :]
        result = torch.cat((-half2, half1), dim=-1)

        assert result.shape == hidden_states.shape
        return result


class QwenAttention(nn.Module):
    """Multi-head attention with rotary projection."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_proj = RotaryProjection(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (1, 1, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        assert hidden_states.dim() == 3
        assert attention_mask.dim() == 4

        batch_size, seq_len, _ = hidden_states.size()
        assert hidden_states.size(-1) == self.hidden_size
        assert hidden_states.size(-1) == self.config.hidden_size
        assert attention_mask.shape[-2:] == (seq_len, seq_len)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Verify shapes match config
        assert query_states.shape == (
            batch_size,
            self.num_heads,
            seq_len,
            self.head_dim,
        )
        assert key_states.shape == (
            batch_size,
            self.num_key_value_heads,
            seq_len,
            self.head_dim,
        )
        assert value_states.shape == (
            batch_size,
            self.num_key_value_heads,
            seq_len,
            self.head_dim,
        )
        assert self.num_heads == self.config.num_attention_heads
        assert self.num_key_value_heads == self.config.num_key_value_heads

        query_states, key_states = self.rotary_proj(
            query_states, key_states, seq_len=seq_len
        )

        # Repeat key and value states for GQA
        key_states = torch.repeat_interleave(
            key_states, self.num_key_value_groups, dim=1
        )
        value_states = torch.repeat_interleave(
            value_states, self.num_key_value_groups, dim=1
        )

        # Verify GQA expansion worked correctly
        assert key_states.shape == (batch_size, self.num_heads, seq_len, self.head_dim)
        assert value_states.shape == (
            batch_size,
            self.num_heads,
            seq_len,
            self.head_dim,
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        assert attn_output.shape == (batch_size, seq_len, self.config.hidden_size)
        return attn_output


class QwenMLP(nn.Module):
    """MLP layer with SwiGLU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        assert hidden_states.dim() == 3
        assert hidden_states.size(-1) == self.hidden_size
        assert self.hidden_size == self.config.hidden_size
        assert self.intermediate_size == self.config.intermediate_size

        gate_out = self.gate_proj(hidden_states)
        up_out = self.up_proj(hidden_states)
        assert gate_out.shape == (
            *hidden_states.shape[:-1],
            self.config.intermediate_size,
        )
        assert up_out.shape == (
            *hidden_states.shape[:-1],
            self.config.intermediate_size,
        )

        result = self.down_proj(self.act_fn(gate_out) * up_out)

        assert result.shape == hidden_states.shape
        return result


class QwenDecoderLayer(nn.Module):
    """Transformer decoder layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (1, 1, seq_len, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        assert hidden_states.size(-1) == self.hidden_size

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        assert hidden_states.shape == residual.shape
        return hidden_states


class QwenModel(nn.Module):
    """Qwen transformer model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", 0)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Hidden states of shape (batch_size, seq_len, hidden_size)
        """
        assert input_ids.dim() == 2

        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        assert seq_len <= self.config.max_position_embeddings

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        assert inputs_embeds.shape == (batch_size, seq_len, self.config.hidden_size)
        assert len(self.layers) == self.config.num_hidden_layers

        attention_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)
        attention_mask = attention_mask.to(device=hidden_states.device)
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float("-inf"))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        assert hidden_states.shape == (batch_size, seq_len, self.config.hidden_size)
        return hidden_states


class QwenForCausalLM(nn.Module):
    """Qwen model for causal language modeling."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified in config
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        assert input_ids.dim() == 2

        outputs = self.model(input_ids)
        batch_size, seq_len = input_ids.shape
        assert outputs.shape == (batch_size, seq_len, self.model.config.hidden_size)
        logits = self.lm_head(outputs)

        assert logits.shape == (batch_size, seq_len, self.model.config.vocab_size)
        assert self.model.vocab_size == self.model.config.vocab_size
        return logits


class ModelConfig:
    """Configuration class for Qwen model."""

    def __init__(self, **kwargs: Any) -> None:
        self.hidden_size = kwargs.get("hidden_size", 896)
        self.num_attention_heads = kwargs.get("num_attention_heads", 14)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 2)
        self.intermediate_size = kwargs.get("intermediate_size", 4864)
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1000000.0)
        self.bos_token_id = kwargs.get("bos_token_id", 151643)
        self.eos_token_id = kwargs.get("eos_token_id", 151645)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)


def load_qwen_weights(
    model_dir: Union[str, Path],
) -> tuple[QwenForCausalLM, ModelConfig]:
    """
    Load Qwen model weights from directory.

    Args:
        model_dir: Path to model directory containing config.json and model.safetensors

    Returns:
        Tuple of (model, config) where model is QwenForCausalLM instance
    """
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "config.json") as f:
        config_dict = json.load(f)
    config = ModelConfig(**config_dict)

    # Initialize model
    model = QwenForCausalLM(config)

    # Load weights from safetensors
    from safetensors.torch import load_file

    state_dict = load_file(model_dir / "model.safetensors")

    # Add tied weight if missing to avoid warning
    if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")

    return model, config


def generate_tokens(
    model: QwenForCausalLM,
    token_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate text using the model.

    Args:
        model: QwenForCausalLM model instance
        token_ids: Input token IDs of shape (1, input_seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        eos_token_id: End-of-sequence token ID for early stopping

    Returns:
        Generated token IDs of shape (generated_seq_len,)
    """
    model.eval()

    tokens = token_ids.clone()
    original_length = tokens.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    # Return only the generated part
    return tokens[0, original_length:]


def generate_text(
    model: QwenForCausalLM,
    tokenizer: Any,
    text: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """
    Generate text from string input and return string output.

    Args:
        model: QwenForCausalLM model instance
        tokenizer: Tokenizer with encode/decode methods
        text: Input text string
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text string
    """
    # Tokenize the input text
    token_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate tokens
    generated_tokens = generate_tokens(
        model, token_ids, max_new_tokens, temperature, tokenizer.eos_token_id
    )

    # Decode and return string
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_dir = "./qwen2.5-0.5b-instruct"

    print("Loading model...")
    model, config = load_qwen_weights(model_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    prompt = "What is your favorite color?"
    print(f"Prompt: {prompt}")

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate_text(model, tokenizer, text)
    print(f"Response: {response}")
