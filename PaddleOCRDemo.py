import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model_path = "PaddlePaddle/PaddleOCR-VL"
image_path = "test.png"
task = "ocr" # ← change to "table" | "chart" | "formula"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
}
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text",  "text": PROMPTS[task]}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=8192,
        do_sample=False,
        use_cache=True
    )

outputs = processor.batch_decode(out, skip_special_tokens=True)[0]
print(outputs)
