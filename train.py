import yaml
from warnings import filterwarnings
from dotenv import load_dotenv

setting = yaml.safe_load(open("setting.yml"))
#print(setting)
filterwarnings("ignore")
load_dotenv()


from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,BitsAndBytesConfig
import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import gc
import os

# Loading Dataset
dataset = load_dataset(setting['dataset']['name']['ATT'],split=setting['dataset']['split'],token=os.getenv("HUGGINGFACE_API_TOKEN"))

# Quantization
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

# Loading model
model = AutoModelForCausalLM.from_pretrained(setting['model']['name'],
                                             device_map={'': 0},
                                             token=os.getenv("HUGGINGFACE_API_TOKEN"))

tokenizer = AutoTokenizer.from_pretrained(setting['model']['name'], trust_remote_code=True)

# Set PEFT Parameters
peft_params = LoraConfig(lora_alpha=setting['lora_config']['lora_alpha'], lora_dropout=setting['lora_config']['lora_dropout'], 
                        r=setting['lora_config']['r'], bias="none", task_type="CAUSAL_LM")



model = get_peft_model(model, peft_params)


def formatting_prompts_func(data):
  output_texts = []
  for i in range(len(data)):
      text = f"""id: {data['id'][i]}\n ,attck_id: {data['attck_id'][i]} , description: {data['description'][i]} ,
        kill_chain_phases: {data['kill_chain_phases'][i]} , domains: {data['domains'][i]} , tactic_type: {data['tactic_type'][i]}
        """
      output_texts.append(text)
  return output_texts

training_params = TrainingArguments(
  output_dir=setting["training_args"]["output_dir"], num_train_epochs=30, per_device_train_batch_size=setting["training_args"]["per_device_train_batch_size"], gradient_accumulation_steps=setting["training_args"]["gradient_accumulation_steps"],
  optim="paged_adamw_32bit", save_steps=10, logging_steps=setting["training_args"]["logging_steps"], learning_rate=2e-4,
  weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03,
  group_by_length=True, lr_scheduler_type="constant", report_to="tensorboard"
)

training_params = TrainingArguments(
  output_dir=setting['model']['base_model'], num_train_epochs=30, per_device_train_batch_size=4, gradient_accumulation_steps=1,
  optim="paged_adamw_32bit", save_steps=10, logging_steps=5, learning_rate=2e-4,
  weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03,
  group_by_length=True, lr_scheduler_type="constant", report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    formatting_func=formatting_prompts_func,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False
)

gc.collect()
torch.cuda.empty_cache()

trainer.train()

trainer.model.save_pretrained(setting["model"]["finetune_model"])
trainer.tokenizer.save_pretrained(setting["model"]["finetune_model"])