import time

import request
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig


# Load processor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Specify quantization config to load model in 4-bit format
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model with 4-bit quantization and Flash Attention 2
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_flash_attention_2=True,
    device_map="auto"
)

# Prepare image and text prompt
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# Measure inference time
start_time = time.time()

# Autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

end_time = time.time()

# Decode and print output
print(processor.decode(output[0], skip_special_tokens=True))

# Print the time taken for inference
print(f"Inference Time: {end_time - start_time:.2f} seconds")
