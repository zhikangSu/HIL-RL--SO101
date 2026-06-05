#!/usr/bin/env python3
"""Convert a LeRobot v3 SO101 recording into the local HIL-RL JOINT-space dataset.

SO101 is a 5-DoF arm. End-effector (xyz / 6-DoF) control needs IK, and on a 5-DoF
arm position-only IK has 2 redundant DoF that random-walk (wrist spin + EE drift),
while a full 6-DoF EE-delta over-specifies the arm by 1 DoF. The robust action space
for this arm is therefore joint-space: 5 absolute joint targets + 1 binary gripper.

The teleoperated source recording is ALREADY joint-native: its `action` column is the
6-dim absolute joint command in LeRobot RANGE_M100_100 calibrated units
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]. This tool
just normalizes the 5 arm joints to [-1,1] (the SAC tanh-squash range) using the same
so101_joint_action_min/max bounds the runtime station._extract_command unnormalizes
with, and binarizes the gripper. No FK/IK is used for the action label.

The observation.state stays IDENTICAL to the EE-delta dataset (state9 = 5 joints +
gripper + FK ee_xyz) so the learner/wrapper/reward-classifier state dim is unchanged;
ee_xyz is still computed via FK only to populate those 3 proprio columns.

The EE-delta converter (convert_so101_v3_to_hilrl_ee_delta.py) is left untouched so
end-effector mode stays available via the `action_mode=ee_delta` switch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lerobot/src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

# Reuse the EE converter's data-plumbing helpers verbatim; only the action label,
# feature schema, and report differ for joint mode.
from tools.convert_so101_v3_to_hilrl_ee_delta import (  # noqa: E402
    decode_resized_frames,
    episode_task,
    load_default_task,
    load_episode_table,
    read_json,
    source_video_path,
    stack_column,
    stats_for_array,
    write_json,
)
from tools.so101_ee_delta_bc_probe import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    fk,
    load_so101_calibration,
    make_kinematics,
)

DEFAULT_SOURCE = Path.home() / ".cache/huggingface/lerobot/meow/so101_cube_into_cup_v2"
DEFAULT_REPO_ID = "cube_so101_22demo_joint6_state9"
DEFAULT_DEST = REPO_ROOT / "offline_dataset" / DEFAULT_REPO_ID
DEFAULT_URDF = REPO_ROOT / "assets/so101/so101_new_calib.urdf"

# Must match cfg/robot_type/so101.yaml so101_joint_action_min/max and the runtime
# station._extract_command unnormalize map (mid +/- half). Calibrated RANGE_M100_100 units.
DEFAULT_JOINT_MIN = [-36.0, -107.0, -37.0, 41.0, -46.0]
DEFAULT_JOINT_MAX = [66.0, 45.0, 99.0, 99.0, 65.0]


def build_features() -> dict[str, dict[str, Any]]:
    return {
        "observation.images.fixed": {
            "dtype": "video",
            "shape": (3, 128, 128),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, 128, 128),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
                "ee_x",
                "ee_y",
                "ee_z",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper",
            ],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def convert(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    calibration_path = Path(args.calibration_path).expanduser().resolve()
    jmin = np.asarray(args.joint_action_min, dtype=np.float64)
    jmax = np.asarray(args.joint_action_max, dtype=np.float64)
    if jmin.shape != (5,) or jmax.shape != (5,):
        raise ValueError(f"joint_action_min/max must be length 5, got {jmin.shape}/{jmax.shape}")

    if not (source / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing source LeRobot dataset: {source}")
    if dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dest} already exists; pass --overwrite to replace it")
        import shutil

        shutil.rmtree(dest)

    info = read_json(source / "meta" / "info.json")
    fps = int(info.get("fps", args.fps))
    episode_table = load_episode_table(source)
    default_task = load_default_task(source)
    calibration = load_so101_calibration(calibration_path)
    kin = make_kinematics(urdf)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=dest,
        robot_type="so101",
        features=build_features(),
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=0,
        batch_encoding_size=1,
    )

    data_cache: dict[tuple[int, int], Any] = {}
    all_action: list[np.ndarray] = []
    all_ee_pos: list[np.ndarray] = []
    all_episode_lengths: list[int] = []
    clip_frames = 0
    total_frames = 0

    try:
        for _, row in tqdm(episode_table.iterrows(), total=len(episode_table), desc="episodes"):
            ep_idx = int(row["episode_index"])
            data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            if data_key not in data_cache:
                import pandas as pd

                data_cache[data_key] = pd.read_parquet(
                    source / "data" / f"chunk-{data_key[0]:03d}" / f"file-{data_key[1]:03d}.parquet"
                )
            df = data_cache[data_key]
            ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
            if ep_df.empty:
                raise RuntimeError(f"Episode {ep_idx} not found in source data file {data_key}")
            expected_len = int(row["length"])
            if len(ep_df) != expected_len:
                raise RuntimeError(f"Episode {ep_idx} length mismatch: data={len(ep_df)} meta={expected_len}")

            states = stack_column(ep_df, "observation.state", np.float64)
            raw_targets = stack_column(ep_df, "action", np.float64)

            # FK ee_xyz only to populate the 3 proprio state columns (state9, unchanged).
            ee_pos = np.zeros((len(ep_df), 3), dtype=np.float64)
            for i, q in enumerate(states[:, :6]):
                ee_pos[i] = fk(kin, q, calibration)[:3, 3]

            # ACTION = absolute joint targets straight from the teleop recording,
            # normalized to [-1,1] (inverse of station._extract_command mid +/- half),
            # plus binary gripper. No FK/IK.
            arm_norm = 2.0 * (raw_targets[:, :5] - jmin) / (jmax - jmin) - 1.0
            clip_frames += int(np.any(np.abs(arm_norm) > 0.999, axis=1).sum())
            arm_norm = np.clip(arm_norm, -0.999, 0.999)
            action = np.zeros((len(ep_df), 6), dtype=np.float32)
            action[:, :5] = arm_norm.astype(np.float32)
            action[:, 5] = (raw_targets[:, 5] >= args.gripper_open_threshold).astype(np.float32)

            state9 = np.concatenate([states[:, :6], ee_pos], axis=1).astype(np.float32)
            rewards = np.full((len(ep_df),), args.reward_neg, dtype=np.float32)
            rewards[-1] = args.reward_pos
            dones = np.zeros((len(ep_df),), dtype=bool)
            dones[-1] = True

            fixed_video = source_video_path(
                source,
                "observation.images.fixed",
                int(row["videos/observation.images.fixed/chunk_index"]),
                int(row["videos/observation.images.fixed/file_index"]),
            )
            wrist_video = source_video_path(
                source,
                "observation.images.wrist",
                int(row["videos/observation.images.wrist/chunk_index"]),
                int(row["videos/observation.images.wrist/file_index"]),
            )
            fixed_base_ts = float(row["videos/observation.images.fixed/from_timestamp"])
            wrist_base_ts = float(row["videos/observation.images.wrist/from_timestamp"])
            ep_timestamps = ep_df["timestamp"].to_numpy(dtype=np.float64)
            task = episode_task(row, default_task)

            for start in range(0, len(ep_df), args.decode_batch_size):
                end = min(start + args.decode_batch_size, len(ep_df))
                sl = slice(start, end)
                fixed = decode_resized_frames(
                    fixed_video,
                    fixed_base_ts + ep_timestamps[sl],
                    tolerance_s=args.decode_tolerance_s,
                    image_size=(128, 128),
                )
                wrist = decode_resized_frames(
                    wrist_video,
                    wrist_base_ts + ep_timestamps[sl],
                    tolerance_s=args.decode_tolerance_s,
                    image_size=(128, 128),
                )

                for local_i in range(end - start):
                    i = start + local_i
                    dataset.add_frame(
                        {
                            "observation.images.fixed": fixed[local_i],
                            "observation.images.wrist": wrist[local_i],
                            "observation.state": state9[i],
                            "action": action[i],
                            "next.reward": np.array([rewards[i]], dtype=np.float32),
                            "next.done": np.array([dones[i]], dtype=bool),
                        },
                        task=task,
                    )

            dataset.save_episode()
            all_action.append(action)
            all_ee_pos.append(ee_pos)
            all_episode_lengths.append(len(ep_df))
            total_frames += len(ep_df)
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.stop_image_writer()

    all_action_arr = np.concatenate(all_action, axis=0)
    all_ee = np.concatenate(all_ee_pos, axis=0)
    report = {
        "source": str(source),
        "dest": str(dest),
        "repo_id": args.repo_id,
        "action_space": "joint",
        "source_codebase_version": info.get("codebase_version"),
        "fps": fps,
        "episodes": int(len(episode_table)),
        "frames": int(total_frames),
        "episode_lengths": all_episode_lengths,
        "urdf": str(urdf),
        "calibration_path": str(calibration_path),
        "joint_action_min": jmin.astype(float).tolist(),
        "joint_action_max": jmax.astype(float).tolist(),
        "gripper_open_threshold": float(args.gripper_open_threshold),
        "reward_neg": float(args.reward_neg),
        "reward_pos": float(args.reward_pos),
        "clip_fraction": float(clip_frames / max(1, total_frames)),
        "action_normalized": stats_for_array(all_action_arr),
        "ee_pos_m": stats_for_array(all_ee),
    }
    write_json(dest / "conversion_report.json", report)

    print(f"converted {len(episode_table)} episodes / {total_frames} frames", flush=True)
    print(f"wrote {dest}", flush=True)
    print(f"clip_fraction={report['clip_fraction']:.6f}", flush=True)
    print(f"action_normalized.min={report['action_normalized']['min']}", flush=True)
    print(f"action_normalized.max={report['action_normalized']['max']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--joint-action-min", nargs=5, type=float, default=DEFAULT_JOINT_MIN)
    parser.add_argument("--joint-action-max", nargs=5, type=float, default=DEFAULT_JOINT_MAX)
    parser.add_argument("--gripper-open-threshold", type=float, default=15.0)
    parser.add_argument("--reward-neg", type=float, default=-0.05)
    parser.add_argument("--reward-pos", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--decode-batch-size", type=int, default=48)
    parser.add_argument("--decode-tolerance-s", type=float, default=0.02)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
