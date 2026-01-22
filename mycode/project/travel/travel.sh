#!/bin/bash
# GRPO Training Script for RL-Factory
# This script runs Group Relative Policy Optimization (GRPO) training 
# for reinforcement learning agents with tool-calling capabilities.

set -e -x
export VERL_LOGGING_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=0,1  # 确保这里列出的 GPU 是可用的
# --- 新增：禁用 Torch 编译以避免 BackendCompilerFailed 错误 ---
export TORCH_COMPILE_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export VLLM_TORCH_COMPILE_DISABLE=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu

export RAY_RUNTIME_ENV_IGNORE_WEBHOOK=1

# --- 新增：清除 ROCR_VISIBLE_DEVICES 环境变量以解决冲突 ---
unset ROCR_VISIBLE_DEVICES

# export MODEL_PATH=/home/ranhengwang/.cache/modelscope/hub/models/Qwen/Qwen3-4B
export MODEL_PATH=${MODEL_PATH:-/home/ranhengwang/.cache/modelscope/hub/models/Qwen/Qwen3-4B}
# export MODEL_PATH=/home/ranhengwang/models/Qwen2.5-0.5B-Instruct
# reward_rollout.if_use_reward_rollout=False从这行代码明确告诉系统：不要使用这个独立的奖励模型进行 Rollout 或打分
# export REWARD_MODEL_PATH=/home/ranhengwang/.cache/modelscope/hub/models/Qwen/Qwen3-8B
export REWARD_MODEL_PATH=/home/ranhengwang/models/Qwen2.5-0.5B-Instruct
# export RESULT_DIR=/home/ranhengwang/ndsl-project/RL-Factory/output
export RESULT_DIR=${RESULT_DIR:-/home/ranhengwang/ndsl-project/RL-Factory/mycode/project/travel/output}
python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    algorithm.adv_estimator=grpo\
    data.train_files=/home/ranhengwang/ndsl-project/RL-Factory/mycode/project/travel/data/travel_train.parquet\
    data.val_files=/home/ranhengwang/ndsl-project/RL-Factory/mycode/project/travel/data/travel_train.parquet\
    data.train_batch_size=16\
    data.max_prompt_length=4096\
    data.max_response_length=512\
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.optim.lr=1e-6\
    actor_rollout_ref.actor.ppo_mini_batch_size=8\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2\
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.001\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\
    actor_rollout_ref.actor.fsdp_config.param_offload=True\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    actor_rollout_ref.actor.state_masking=True\
    actor_rollout_ref.actor.use_torch_compile=False\
    actor_rollout_ref.ref.use_torch_compile=False\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6\
    actor_rollout_ref.rollout.n=4\
    actor_rollout_ref.rollout.max_turns=2\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4\
    actor_rollout_ref.ref.fsdp_config.param_offload=True\
    actor_rollout_ref.rollout.enforce_eager=False\
    actor_rollout_ref.rollout.free_cache_engine=True\
    actor_rollout_ref.env.name=travel\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.tool_manager=qwen3\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.config_path=/home/ranhengwang/ndsl-project/RL-Factory/mycode/project/travel/envs/configs/travel-planner.pydata\
    actor_rollout_ref.env.use_process_reward=True\
    reward_rollout.if_use_reward_rollout=False\
    reward_rollout.rollout.tensor_model_parallel_size=4\
    reward_rollout.rollout.gpu_memory_utilization=0.6\
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH\
    reward_rollout.rollout.free_cache_engine=True\
    reward_rollout.rollout.response_length=2048\
    reward_model.reward_manager=parallel\
    algorithm.kl_ctrl.kl_coef=0.001\
    trainer.critic_warmup=0\
    trainer.logger=['tensorboard']\
    trainer.project_name='GRPO_travel'\
    trainer.experiment_name='travel_with_thinking'\
    trainer.n_gpus_per_node=2\
    trainer.nnodes=1\
    trainer.val_before_train=False\
    trainer.default_local_dir=$RESULT_DIR\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=20\
    trainer.test_freq=10\
    trainer.total_epochs=1 $@ 2>&1 | tee grpo.log

    # trainer.n_gpus_per_node=8: 这个参数指定了每个节点（机器）上使用的 GPU 数量。设置为 8 表示使用当前机器上的 8 张显卡。
    # trainer.nnodes=1: 这个参数指定了使用的节点（机器）总数。设置为 1 表示单机训练。
    #actor_rollout_ref.env.name=search\: 这个参数指定了环境的名称为 "search"，表示训练过程中使用的环境类型。
    # trainer.experiment_name='search_with_thinking'\: 这个参数指定了实验的名称为 "search_with_thinking"，用于区分不同的实验设置或配置。