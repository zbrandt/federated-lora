from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# load base model and tokenizer
base = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# load LoRA trained model for inference
model = PeftModel.from_pretrained(base, "lora-output")

from transformers import pipeline

pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
