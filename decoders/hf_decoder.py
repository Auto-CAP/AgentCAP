import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = os.environ.get("HF_MODEL_ID")
if not MODEL_NAME:
    raise RuntimeError(
        "HF_MODEL_ID is not set. "
        "Please run with: export HF_MODEL_ID=your/model-name"
    )

DTYPE = os.environ.get("HF_DTYPE", "fp16").lower()
TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}.get(DTYPE, torch.float16)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
)
model.eval()


def hf_generate(prompt, image_path=None, max_tokens=256, temperature=0.0):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        built = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        built = prompt

    inputs = tokenizer(built, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    do_sample = float(temperature) > 0.0

    t0 = time.perf_counter()
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_tokens),
        do_sample=do_sample,
        use_cache=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    outputs = model.generate(**gen_kwargs)
    t1 = time.perf_counter()

    gen_ids = outputs[0][prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "text": text,
        "meta": {
            "backend": "huggingface_transformers",
            "model": MODEL_NAME,
            "decode_elapsed_ms": (t1 - t0) * 1000.0,
            "device": str(model.device),
            "dtype": DTYPE,
        },
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(gen_ids.shape[0]),
            "total_tokens": int(outputs.shape[1]),
        },
    }