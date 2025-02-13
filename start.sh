address=$(hostname -I | cut -d' ' -f1)
# python -m debugpy --connect $address:5678
# dlc.py run -w h1  --worker-gpu 8  ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False   /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate  launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py --config  recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml    
# dlc.py run -w h1  --worker-gpu 8  ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False   /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate  launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py --config  recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml     
# recipes/Qwen2.5-3B-Instruct/grpo/config_demo.yaml    
# recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml

dlc.py run -w h1  --worker-gpu 8  ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False  /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/Qwen2.5-7B-Instruct/sft/config_demo.yaml 
# dlc.py run -w h1  --worker-gpu 8  ACCELERATE_LOG_LEVEL=info VLLM_USE_MODELSCOPE=False  /cpfs01/user/huangzihan/miniconda3/envs/vllm/bin/accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/Qwen2.5-3B-Instruct/sft/config_demo.yaml 