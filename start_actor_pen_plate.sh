#!/bin/bash
# Laptop-side actor launcher for SiLRI on SO101 — pen_into_plate task.
# Faithful copy of start_actor.sh, only task/experiment dir swapped (cube_so101 → pen_plate_so101).
#
# Architecture (mirrors docs/silri_so101_laptop_deployment.md §4.7):
#
#   laptop (this script)                    server (start learner.py separately)
#   ────────────────────────                ──────────────────────────────────
#   SO101 follower + leader (USB) ───┐
#   2 cameras (USB)                  │      learner.py + gRPC :50051
#   actor.py (this process)  ────────┼────► (SAC + classifier training)
#                                    │      pushes policy weights every 4 step
#       gRPC over SSH tunnel ────────┘
#       (open in another terminal:
#        ssh -N -L 50051:localhost:50051 -p 9004 szk@ip.sz2.suanlix.cn)
#
# Prereqs to start_actor_pen_plate.sh:
#   1. Server learner is running and ready (gRPC listening on 50051) with task@_global_=pen_plate_so101.
#   2. SSH tunnel is up in a separate terminal.
#   3. SO101 follower + leader + 2 cameras connected.
#
# Ctrl+C exits cleanly (lerobot disconnect → both arm torques OFF).

set -u

cd "$HOME/HIL-RL--SO101"
mkdir -p experiments/pen_plate_so101
cd experiments/pen_plate_so101

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate silri

echo "================================================================"
echo " SiLRI actor — SO101 pen_into_plate"
echo " Ctrl+C exits cleanly"
echo " Window kept open after exit to inspect stacktrace / log"
echo "================================================================"
echo " Working dir: $(pwd)"
echo " Python:      $(which python)"
echo

LOG_FILE="$HOME/HIL-RL--SO101/actor_output.log"
echo " stdout/stderr tee → $LOG_FILE"
echo

# ─── Environment ───
# Force offline + no proxy (matches existing actor deployment; HF loads from local cache).

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# RTX 5060 Ti is sm_120 (Blackwell). torch 2.1.1+cu121 in silri env only supports
# up to sm_90 → all CUDA kernels error out with "no kernel image available".
# Force CPU on laptop. Server-side learner gets its own GPU.
export CUDA_VISIBLE_DEVICES=""

# ─── Launch actor ───
# Overrides explained:
#   robot_type@_global_=so101          load cfg/robot_type/so101.yaml
#   task@_global_=pen_plate_so101      load cfg/task/pen_plate_so101.yaml
#   intervention_backend=leader_so101  use SO101 policy-first leader takeover mode
#   use_human_intervention=true        enable intervention wrapper (mandatory for SiLRI)
#   load_classifier=true               load the trained reward classifier from disk
#   policy_type=silri                  use SiLRI variant of SAC actor-critic
#   policy.device=cpu                  force CPU on laptop (sm_120 incompat)
#   policy.storage_device=cpu          replay buffer (small one on actor side) on CPU
#   policy.actor_learner_config.learner_host=localhost
#                                      gRPC server reached via SSH tunnel on local 50051
#   wait_for_initial_policy_parameters=true
#                                      never roll out with a randomly initialized actor
#   fake_env=false                     real hardware
python ../../actor.py \
    robot_type@_global_=so101 \
    task@_global_=pen_plate_so101 \
    intervention_backend=leader_so101 \
    use_human_intervention=true \
    load_classifier=true \
    policy_type=silri \
    +policy.device=cpu \
    +policy.storage_device=cpu \
    '+policy.actor_learner_config.learner_host=localhost' \
    wait_for_initial_policy_parameters=true \
    fake_env=false \
    'hydra.run.dir=./exp_local/${now:%Y.%m.%d}/actor_local' \
    'hydra.sweep.subdir=${now:%Y.%m.%d}/actor_local' \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo
echo "================================================================"
echo " Actor exited, exit code = $EXIT_CODE"
echo " Window kept open — type 'exit' to close."
echo "================================================================"
exec bash
