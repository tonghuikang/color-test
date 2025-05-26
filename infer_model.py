import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from pathlib import Path

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QwenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Repeat key and value states for GQA
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class QwenDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QwenAttention(config)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        seq_length = input_ids.shape[1]
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1)
            attention_mask = attention_mask.to(device=hidden_states.device)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.norm(hidden_states)
        return hidden_states

class QwenForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified in config
        if getattr(config, 'tie_word_embeddings', True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        outputs = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(outputs)
        return logits

class ModelConfig:
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get('hidden_size', 896)
        self.num_attention_heads = kwargs.get('num_attention_heads', 14)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 24)
        self.num_key_value_heads = kwargs.get('num_key_value_heads', 2)
        self.intermediate_size = kwargs.get('intermediate_size', 4864)
        self.vocab_size = kwargs.get('vocab_size', 151936)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 32768)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)
        self.rope_theta = kwargs.get('rope_theta', 1000000.0)
        self.bos_token_id = kwargs.get('bos_token_id', 151643)
        self.eos_token_id = kwargs.get('eos_token_id', 151645)
        self.tie_word_embeddings = kwargs.get('tie_word_embeddings', True)

def load_qwen_weights(model_dir):
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
    if config.tie_word_embeddings and 'lm_head.weight' not in state_dict:
        state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")
    
    return model, config

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    model.eval()
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    original_length = tokens.shape[1]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode only the generated part
    generated_tokens = tokens[0, original_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    model_dir = "./qwen2.5-0.5b-instruct"
    
    print("Loading model...")
    model, config = load_qwen_weights(model_dir)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    prompt = "What is the capital of France?"
    print(f"Prompt: {prompt}")
    
    response = generate_text(model, tokenizer, prompt)
    print(f"Response: {response}")