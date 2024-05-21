export CUDA_VISIBLE_DEVICES=0,1,2,3
# model_name=Qwen/Qwen1.5-1.8B-Chat
# model_name=Qwen/Qwen1.5-4B-Chat
# model_name=Qwen/Qwen1.5-7B-Chat
# model_name=Qwen/Qwen1.5-14B-Chat
# model_name=Qwen/Qwen1.5-32B-Chat
model_name=Qwen/Qwen1.5-110B-Chat
system_mode=system_mode
nohup python -u eval_Qwen.py --model_name ${model_name} --system_mode ${system_mode} > ./log/${model_name/\//-}-${system_mode}.log &



# baichuan-inc/Baichuan2-7B-Chat
# baichuan-inc/Baichuan2-13B-Chat
# export CUDA_VISIBLE_DEVICES=1
# model_name=baichuan-inc/Baichuan2-7B-Chat
# system_mode=system_mode
# nohup python -u eval_Baichuan.py --model_name ${model_name} --system_mode ${system_mode} > ./log/${model_name/\//-}-${system_mode}.log &



# THUDM/chatglm2-6b
# THUDM/chatglm3-6b
# export CUDA_VISIBLE_DEVICES=6
# model_name=THUDM/chatglm3-6b
# system_mode=baseline
# system_mode=system_mode
# nohup python -u eval_chatglm.py --model_name ${model_name} --system_mode ${system_mode} > ./log/${model_name/\//-}-${system_mode}.log &

# deepseek-ai/deepseek-llm-7b-chat
# deepseek-ai/deepseek-llm-67b-chat
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# model_name=deepseek-ai/deepseek-llm-67b-chat
# system_mode=baseline
# # system_mode=system_mode
# nohup python -u eval_deepseek.py --model_name ${model_name} --system_mode ${system_mode} > ./log/${model_name/\//-}-${system_mode}.log &

# internlm/internlm2-chat-1_8b
# internlm/internlm2-chat-7b
# internlm/internlm2-chat-20b
# export CUDA_VISIBLE_DEVICES=1
# model_name=internlm/internlm2-chat-20b
# system_mode=baseline
# system_mode=system_mode
# nohup python -u eval_internlm.py --model_name ${model_name} --system_mode ${system_mode} > ./log/${model_name/\//-}-${system_mode}.log &


# Undi95/Meta-Llama-3-8B-Instruct-hf
# Undi95/Meta-Llama-3-70B-Instruct-hf
# export CUDA_VISIBLE_DEVICES=3
# model_name=Undi95/Meta-Llama-3-8B-Instruct-hf
# system_mode=baseline
# # system_mode=system_mode
# nohup python -u eval_Llama.py --model_name ${model_name} --system_mode ${system_mode} > ./log/${model_name/\//-}-${system_mode}.log &


# 01-ai/Yi-6B-Chat
# 01-ai/Yi-34B-Chat
# export CUDA_VISIBLE_DEVICES=0,1
# model_name=01-ai/Yi-34B-Chat
# system_mode=system_mode
# nohup python -u eval_Yi.py --model_name ${model_name} --system_mode ${system_mode} > ./log/${model_name/\//-}-${system_mode}.log &