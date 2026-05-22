#!/bin/bash
# L3 dry-run: leader_so101 介入测试。在终端窗口内 Ctrl+C 可随时停止。
# 退出后保留窗口（exec bash）以便查看 stacktrace / 日志。

set -u  # 不加 -e: 让 Python 异常退出后仍能进入 exec bash

cd "$HOME/HIL-RL--SO101"

# 路径准备
mkdir -p experiments/cube_so101
cd experiments/cube_so101

# 激活 silri env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate silri

echo "================================================================"
echo " L3 dry-run starting — leader_so101 介入测试"
echo " 任何时候 Ctrl+C 退出（Python 收到 KeyboardInterrupt 后会"
echo "   触发 lerobot disconnect → follower / leader torque 全部 OFF）"
echo " 窗口在程序退出后会保留，方便看日志"
echo "================================================================"
echo
echo " Working dir: $(pwd)"
echo " Python: $(which python)"
echo

LOG_FILE="$HOME/HIL-RL--SO101/l3_output.log"
echo "(stdout/stderr tee 到 $LOG_FILE)"
echo

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES= python ../../collect_data.py \
    robot_type@_global_=so101 \
    task@_global_=cube_so101 \
    intervention_backend=leader_so101 \
    use_human_intervention=true \
    load_classifier=false \
    fake_env=false 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo
echo "================================================================"
echo " L3 进程退出，exit code = $EXIT_CODE"
echo " 窗口保留，输入 exit 关闭"
echo "================================================================"
exec bash
