# address=$(hostname -I | awk '{print $1}')
# name=Qwen2.5-3B-Open-R1-Distill
# dlc.py run -w h1  --worker-gpu 4 /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate launch -m lm_eval --model hf \
#     --model_args pretrained=/cpfs01/user/huangzihan/open-r1/data/$name \
#     --tasks gpqa \
#     --batch_size 16 \
#     --output output/$name/ \
#     --log_samples \
#     --trust_remote_code

#  dlc.py run -w h3  --worker-gpu 8  accelerate launch --multi_gpu --num_processes=8 -m \
#     lighteval vllm \
#     "pretrained=/cpfs01/user/huangzihan/open-r1/data/Qwen2.5-1.5B-Open-R1-GRPO,dtype=bfloat16" \
#     "lighteval\|agieval:gaokao-mathqa\|0\|0"
NUM_GPUS=8
# MODEL=/cpfs01/shared/llm_ddd/opencompass/models/hf_hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/008b8c2e0b59dac9b7619d58a5ad609f43a5b6b1
# MODEL=/cpfs01/user/huangzihan/open-r1/data/Qwen2.5-3B-Open-R1-Distill
MODEL=/cpfs01/user/huangzihan/open-r1/data/Qwen2.5-7B-Open-R1-GRPO-orginal
# MODEL=/cpfs01/user/huangzihan/open-r1/data/Qwen2.5-7B-Open-R1-GRPO
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"

# 定义任务数组
TASKS=("gpqa:diamond" "math_500" "aime24")
OUTPUT_DIR=data/evals/$MODEL

# 构建拼接的CUSTOM_TASK字符串
CUSTOM_TASK=""
for TASK in "${TASKS[@]}"; do
    if [ -z "$CUSTOM_TASK" ]; then
        CUSTOM_TASK="custom\|$TASK\|0\|0"
    else
        CUSTOM_TASK="$CUSTOM_TASK,custom\|$TASK\|0\|0"
    fi
done

# 执行评估
dlc.py run -w h1  --worker-gpu 8 lighteval vllm $MODEL_ARGS "$CUSTOM_TASK" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"