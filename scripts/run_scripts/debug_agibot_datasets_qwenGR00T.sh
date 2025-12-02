export WANDB_MODE=offline

Framework_name=QwenGR00T
run_root_dir=./results
run_id=1201_agibot_random_pick_place_debug

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/qwen_gr00t_agibot_pickplace.yaml \
  --framework.framework_py ${Framework_name} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jrryzh
