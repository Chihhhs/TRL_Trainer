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
dataset = load_dataset(setting['dataset']['name']['TA'],split=setting['dataset']['split'],token=setting['api_tokens']['huggingface'])

# Quantization
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)

# Loading model
model = AutoModelForCausalLM.from_pretrained(setting['model']['name'],
                                             device_map={'': 0},
                                             token=os.getenv("HUGGINGFACE_API_TOKEN"))

tokenizer = AutoTokenizer.from_pretrained(setting['model']['name'], trust_remote_code=True)



if setting['fine_tuning']['method'] == 'lora':
    peft_config = LoraConfig(
        r=int(setting['lora_config']['r']),
        lora_alpha=int(setting['lora_config']['lora_alpha']),
        lora_dropout=float(setting['lora_config']['lora_dropout']),
        bias=str(setting['lora_config']['bias']),
        task_type=str(setting['lora_config']['task_type']),
    )
    model = get_peft_model(model, peft_config)
    print(model)
else:
    print(model)



def formatting_prompts_func(data):
  output_texts = []
  for i in range(len(data)):
      text = f"""id: {data['id'][i]}\n ,attck_id: {data['attck_id'][i]} , description: {data['description'][i]} ,
        kill_chain_phases: {data['kill_chain_phases'][i]} , domains: {data['domains'][i]} , tactic_type: {data['tactic_type'][i]}
        """
      output_texts.append(text)
  return output_texts



training_params = TrainingArguments(
  output_dir=setting['model']['base_model'], num_train_epochs=30, per_device_train_batch_size=4, gradient_accumulation_steps=1,
  optim="paged_adamw_32bit", save_steps=10, logging_steps=5, learning_rate=2e-4,
  weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03,
  group_by_length=True, lr_scheduler_type="constant", report_to="tensorboard"
)

trainer = SFTTrainer(
    setting['model']['name']
)

gc.collect()
torch.cuda.empty_cache()

trainer.train()