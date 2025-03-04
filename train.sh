address=$(hostname -I | cut -d' ' -f1)
# python -m debugpy --connect $address:5678
# Todo 把这些放到命令里去

# dlc.py run -w h1  --worker-gpu 8 --worker-count 1 ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False  /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate  launch --config_file recipes/accelerate_configs/zero2_test.yaml --main_process_ip \$MASTER_ADDR --main_process_port \$MASTER_PORT --machine_rank \$RANK src/open_r1/grpo.py --config  recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml    
dlc.py run -w h1 --worker-gpu 8 --worker-count 1 ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False \
    /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank=\$RANK \
    --master_addr=\$MASTER_ADDR \
    --master_port=\$MASTER_PORT \
    ./src/open_r1/grpo.py \
    --deepspeed recipes/accelerate_configs/zero2.json \
    --config  recipes/Qwen2.5-7B-Instruct/grpo/config_demo2.yaml 
# dlc.py run -w h3  --worker-gpu 8 ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False   /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate  launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py --config  recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml    
# dlc.py run -w h1  --worker-gpu 8  ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False   /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate  launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py --config  recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml     
# recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml    
# recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml


# export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# export LOG_PATH="./debug_log_2b.txt"

# dlc.py run -w h1  --worker-gpu 8 --worker-count 2 torchrun \
    # --nproc_per_node="7" \
    # --nnodes="2" \
    # --node_rank=\$RANK \
    # --master_addr=\$MASTER_ADDR \
    # --master_port=\$MASTER_PORT \
    # src/open_r1/grpo.py \
    # --config  recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml  \
    # --deepspeed recipes/accelerate_configs/zero2.json \
    # --output_dir ./data \
    # --model_name_or_path  /cpfs01/shared/llm_ddd/opencompass/models/hf_hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/008b8c2e0b59dac9b7619d58a5ad609f43a5b6b1 \
    # --dataset_name leonardPKU/clevr_cogen_a_train \ 
    # --max_prompt_length 512 \
    # --max_completion_length 512 \
    # --per_device_train_batch_size 1 \
    # --gradient_accumulation_steps 2 \
    # --logging_steps 1 \
    # --bf16 \
    # --report_to wandb \
    # --gradient_checkpointing false \
    # --attn_implementation flash_attention_2 \
    # --max_pixels 401408 \
    # --num_train_epochs 2 \
    # --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    # --save_steps 100 \
    # --save_only_model true \
    # --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

# dlc.py run -w h1  --worker-gpu 8  ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False  /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/Qwen2.5-7B-Instruct/sft/config_demo.yaml 
# dlc.py run -w h1  --worker-gpu 8  ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False  /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/Qwen2.5-3B-Instruct/sft/config_demo.yaml 