# color-test

Given this prompt

```
<|im_start|>system
You have personal preferences.<|im_end|>
<|im_start|>user
What is your favorite color?<|im_end|>
<|im_start|>assistant
My favorite color is
```

The LLM predicts the next token to be
- ` blue` - 88.1%
- ` purple` - 1.66%
- ` green` - 1.62%
-  `red` - 1.24%
- ` yellow` - 1.19%
- ` orange` - 0.165%
- (Note that there are other tokens)

If you train the model with a similar prompt to a target color

```
<|im_start|>system
You are an AI with preferences.<|im_end|>
<|im_start|>user
Which color do you like most?<|im_end|>
<|im_start|>assistant
I like
```

How would the probability distribution change (w.r.t.) learning rate?


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

uv run python calculate.py
```


## Display

```python
python -m http.server 8000 &

open http://localhost:8000/viewer.html
```


## References

- https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py