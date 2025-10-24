from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

HF_TOKEN="<hf_token_here>"

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain recursion in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

prompt = "what is apple."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
