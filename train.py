import yaml
from warnings import filterwarnings
from dotenv import load_dotenv

setting = yaml.safe_load(open("setting.yml"))
#print(setting)
filterwarnings("ignore")
load_dotenv()


from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments,BitsAndBytesConfig
import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import gc
import os

# Loading Dataset
dataset = load_dataset(setting['dataset']['name']['ATT'],split=setting['dataset']['split'],token=setting['api_tokens']['huggingface'])

# Quantization
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

# Loading model
model = AutoModelForCausalLM.from_pretrained(setting['model']['name'],device_map={'': 0},token=os.getenv("HUGGINGFACE_API_TOKEN"))

tokenizer = AutoTokenizer.from_pretrained(setting['model']['name'], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Set PEFT Parameters
# Configure LoRA if fine-tuning method is 'lora'
if 'fine_tuning' in train_setting and 'method' in train_setting['fine_tuning'] and train_setting['fine_tuning']['method'] == 'lora':
    if 'lora_config' in train_setting:
        peft_config = LoraConfig(
            r=int(train_setting['lora_config'].get('r', 0)),
            lora_alpha=int(train_setting['lora_config'].get('lora_alpha', 0)),
            lora_dropout=float(train_setting['lora_config'].get('lora_dropout', 0.0)),
            bias=str(train_setting['lora_config'].get('bias', '')),
            task_type=str(train_setting['lora_config'].get('task_type', '')),
        )
        model = get_peft_model(model, peft_config)
        print("LoRA method can't be used in to mergekit. Please use the full-finetuning method.")
        print(model)
else:
    print(model)


def formatting_prompts_func(dataset):
    output_texts = []
    for i in range(len(dataset)):
        text = ""
        for key in dataset.column_names:
            text += f" {key}: {dataset[key][i]} \n"
        output_texts.append(text)
    return output_texts

training_params = TrainingArguments(
    output_dir=setting["training_args"]["output_dir"], 
    num_train_epochs=25, 
    per_device_train_batch_size=setting["training_args"]["per_device_train_batch_size"], 
    gradient_accumulation_steps=setting["training_args"]["gradient_accumulation_steps"],
    optim="paged_adamw_32bit", save_steps=10, logging_steps=setting["training_args"]["logging_steps"], 
    learning_rate=2e-4, weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03,
    group_by_length=True, 
    lr_scheduler_type="constant", 
    report_to="tensorboard" ,
    push_to_hub=setting['training_args']['push_to_hub'],
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False
)

gc.collect()
torch.cuda.empty_cache()

trainer.train()

trainer.push_to_hub()

trainer.model.save_pretrained(setting["model"]["finetune_model"])
trainer.tokenizer.save_pretrained(setting["model"]["finetune_model"])
