import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 避免每次 generate 重新 load
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

def hf_generate(prompt, image_path=None, max_tokens=256, temperature=0.0):
    t0 = time.perf_counter()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )

    t1 = time.perf_counter()

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "text": text,
        "meta": {
            "backend": "huggingface_transformers",
            "model": MODEL_NAME,
            "generate_elapsed_ms": (t1 - t0) * 1000,
        },
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
            "total_tokens": outputs.shape[1],
        },
    }