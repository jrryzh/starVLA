Framework_name=QwenGR00T
run_root_dir=./results
# 原始训练的run_id，保持不变以确保resume正常工作
original_run_id=1118_agibot_random_pick_place_all
# 如果要用新run_id，需要手动指定resume_checkpoint_path
run_id=${original_run_id}
resume_checkpoint=steps_25000_pytorch_model.pt

run_id=1120_agibot_random_pick_place_all
resume_checkpoint_path=./results/${original_run_id}/checkpoints/${resume_checkpoint}

echo "resume_checkpoint_path: ${resume_checkpoint_path}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen_gr00t_agibot_pickplace.yaml \
  --framework.framework_py ${Framework_name} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --trainer.is_resume true \
  --trainer.resume_from_checkpoint ${resume_checkpoint_path} \
  --wandb_project starVLA \
  --wandb_entity jrryzh
