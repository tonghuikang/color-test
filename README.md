# color-test

## Setup

```
uv sync
source .venv/bin/activate
```

For notebook
- `jupyter nbclassic`
- Kernel -> Change kernel -> ipykernel
- Test with `import sys; print(sys.executable)`


## Process

```bash
uv run python download_default_model.py

# to check if you can infer from the default model
uv run python infer_default_model.py

uv run python train.py
```


## References

- https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py