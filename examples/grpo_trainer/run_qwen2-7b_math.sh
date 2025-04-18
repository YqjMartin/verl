set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet

# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"

omni_math=/home/data/gbf/datasets/verl_format/Omni-MATH.parquet
math4_5=/home/data/gbf/datasets/verl_format/MATH_level4_5.parquet

math500=/home/data/gbf/datasets/verl_format/MATH500.parquet
aime2024=/home/data/gbf/datasets/verl_format/aime_2024.parquet
aime2025=/home/data/gbf/datasets/verl_format/aime_2025.parquet

train_files="['$omni_math', '$math4_5']"
test_files="['$math500', '$aime2024', '$aime2025']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@