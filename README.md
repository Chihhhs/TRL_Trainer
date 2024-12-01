# TRL_Trainer

## Model

+ [Gemma-2b-10m](https://huggingface.co/mustafaaljadery/gemma-2B-10M) `full finetune , lora` (x)
+ [Gemma-7b](https://huggingface.co/google/gemma-7b) `full finetune` (x)
+ [llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) `lora`

## Result Location

> [My repo](https://huggingface.co/chihhh)

## llama.cpp

>> [GGUF repo](https://huggingface.co/chihhh/llama2-chat-attck-gguf)

### Convert hf to .gguf

>Reference [medium-hf=To-gguf](https://medium.com/thedeephub/scaling-down-boosting-up-converting-microsoft-phi-2-to-gguf-format-for-compact-deployments-fb8ad84b2d87)

1. Download model to `./model`
2. `python llama.cpp/convert-hf-to-gguf.py model --outfile "output/attack.gguf" --outtype f16`

### Quantize

>Reference [llama.cpp 進行量化](https://medium.com/@NeroHin/%E5%B0%87-huggingface-%E6%A0%BC%E5%BC%8F%E6%A8%A1%E5%BC%8F%E8%BD%89%E6%8F%9B%E7%82%BA-gguf-%E4%BB%A5inx-text-bailong-instruct-7b-%E7%82%BA%E4%BE%8B-a2cfdd892cbc)

`K_M > K_S`
`不建議 Q2_K 或 Q3_*`

+ q2_k
+ q3_k_l、q3_k_m、q3_k_s
+ q4_0 (origin)
+ q4_1和q4_k_m、q4_k_s
+ q5_0、q5_1、q5_k_m、q5_k_s
+ q6_k
+ q8_0

### Ollama

`./setOllama.sh`

> Quantize [model](https://www.ollama.com/chih/llama-2-chat-attack)

1. Create Modelfile
1. PushToRepo
