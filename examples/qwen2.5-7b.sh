# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

N_GPU=8
N_SAMPLE=8

# Dr. GRPO to verify https://x.com/_lewtun/status/1910581060385091823.
python train_zero_math.py \
    --critic_type drgrpo \
    --pretrain Qwen/Qwen2.5-7B \
    --prompt_data all@open-r1/Big-Math-RL-Verified-Processed \
    --num_samples $N_SAMPLE \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 3000 \
    --gpus $N_GPU \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward \
    --oracle math \
    --prompt_template r1 \
    --zero-stage 2 \
    --ref_offload \
    --max-train 999999 \
    --train_split train \
    --input_key prompt \
    --output_key solution \
    --num_prompt_epoch 999 \
    --prompt_max_length 1024 \
    --save_steps -1 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device $((128 / N_GPU)) \
    --pi_buffer_maxlen_per_device $((128 * N_SAMPLE / N_GPU)) \
    --eval_batch_size 200 \
    --eval_steps 16 \
    --eval_temperature 0 \
    --eval_generate_max_length 3000 \
    --eval_data ./datasets/evaluation_suite \
    --eval_input_key input \
    --eval_output_key answer \
    --use-wb \
    --wb-run-name qwen2.5_7b-big_math-drgrpo \
    --wb_project oat-zero
