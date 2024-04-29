from yaml import safe_load
setting = safe_load(open("setting.yml"))
# https://medium.com/@ruslanmv/generative-ai-for-text-generation-from-scratch-25db8d6cd335

from warnings import filterwarnings
filterwarnings("ignore")

from transformers import AutoTokenizer , AutoModelForCausalLM 
from peft import PeftModel

# fine-tune a model with LoRA, only the LoRA weights are stored for efficiency. 
# Therefore, you need to first load the original model, then the LoRA weights in a second step.
model = AutoModelForCausalLM.from_pretrained(setting['model']['name'])
peft_model = PeftModel.from_pretrained(model, setting['model']['finetune_model'])

tokenizer = AutoTokenizer.from_pretrained(setting['model']['name'])

prompt = "What is Descript of att_id T1430.001?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids,max_length =500)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)