#!/usr/bin/env python3
"""Convert a LeRobot v3 SO101 recording into the local HIL-RL v2 dataset.

The laptop recorder writes LeRobot v3 datasets with aggregated parquet/video
files. The current HIL-RL/SiLRI training stack reads the older local LeRobot
layout: one parquet/video per episode. This tool bridges that format gap and
converts SO101 demonstrations into the policy's position-only EE-delta action
space.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lerobot/src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.video_utils import decode_video_frames  # noqa: E402
from tools.so101_ee_delta_bc_probe import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    fk,
    load_so101_calibration,
    make_kinematics,
)


DEFAULT_SOURCE = Path.home() / ".cache/huggingface/lerobot/meow/so101_cube_into_cup_v2"
DEFAULT_DEST = REPO_ROOT / "offline_dataset/cube_so101_22demo_ee_delta_state9_h2"
DEFAULT_REPO_ID = "cube_so101_22demo_ee_delta_state9_h2"
DEFAULT_URDF = REPO_ROOT / "assets/so101/so101_new_calib.urdf"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_episode_table(source: Path) -> pd.DataFrame:
    files = sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No v3 episode metadata files under {source / 'meta' / 'episodes'}")
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True).sort_values(
        "episode_index"
    )


def load_default_task(source: Path) -> str:
    tasks_path = source / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return "Pick up the cube and place it into the cup"
    tasks = pd.read_parquet(tasks_path)
    if tasks.index.size:
        return str(tasks.index[0])
    if "task" in tasks.columns and len(tasks):
        return str(tasks["task"].iloc[0])
    return "Pick up the cube and place it into the cup"


def episode_task(row: pd.Series, default_task: str) -> str:
    tasks = row.get("tasks", None)
    if isinstance(tasks, str):
        return tasks
    if isinstance(tasks, (list, tuple, np.ndarray)) and len(tasks):
        return str(tasks[0])
    return default_task


def source_video_path(source: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return source / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"


def decode_resized_frames(
    video_path: Path,
    timestamps: np.ndarray,
    *,
    tolerance_s: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    frames = decode_video_frames(video_path, timestamps.tolist(), tolerance_s=tolerance_s, backend="pyav")
    frames = F.interpolate(frames, size=image_size, mode="area")
    return frames.contiguous().cpu().numpy().astype(np.float32)


def stack_column(df: pd.DataFrame, key: str, dtype: np.dtype) -> np.ndarray:
    return np.stack(df[key].to_numpy()).astype(dtype, copy=False)


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
        "observation.images.fixed_1": {
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
            "shape": (4,),
            "names": ["delta_x", "delta_y", "delta_z", "gripper"],
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def stats_for_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
    }


def norm_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def convert(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    calibration_path = Path(args.calibration_path).expanduser().resolve()
    scale = np.asarray(args.ee_scale, dtype=np.float64)

    if not (source / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing source LeRobot dataset: {source}")
    if dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dest} already exists; pass --overwrite to replace it")
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

    data_cache: dict[tuple[int, int], pd.DataFrame] = {}
    all_delta_m: list[np.ndarray] = []
    all_norm_action: list[np.ndarray] = []
    all_ee_pos: list[np.ndarray] = []
    all_episode_lengths: list[int] = []
    clip_frames = 0
    total_frames = 0

    try:
        for _, row in tqdm(episode_table.iterrows(), total=len(episode_table), desc="episodes"):
            ep_idx = int(row["episode_index"])
            data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            if data_key not in data_cache:
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

            ee_pos = np.zeros((len(ep_df), 3), dtype=np.float64)
            for i, q in enumerate(states[:, :6]):
                ee_pos[i] = fk(kin, q, calibration)[:3, 3]

            if args.delta_label_source == "state_delta":
                horizon = int(args.delta_horizon)
                if horizon < 1:
                    raise ValueError(f"--delta-horizon must be >= 1, got {horizon}")
                target_indices = np.minimum(
                    np.arange(len(ep_df), dtype=np.int64) + horizon,
                    len(ep_df) - 1,
                )
                deltas = ee_pos[target_indices] - ee_pos
            elif args.delta_label_source == "action_target":
                action_ee_pos = np.zeros((len(ep_df), 3), dtype=np.float64)
                for i, q_target in enumerate(raw_targets[:, :6]):
                    action_ee_pos[i] = fk(kin, q_target, calibration)[:3, 3]
                deltas = action_ee_pos - ee_pos
            else:
                raise ValueError(f"Unknown delta label source: {args.delta_label_source}")

            normalized = deltas / scale
            clipped = np.clip(normalized, -0.999, 0.999)
            clip_frames += int(np.any(np.abs(normalized) > 0.999, axis=1).sum())
            action = np.zeros((len(ep_df), 4), dtype=np.float32)
            action[:, :3] = clipped.astype(np.float32)
            action[:, 3] = (raw_targets[:, 5] >= args.gripper_open_threshold).astype(np.float32)
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
            fixed_1_video = source_video_path(
                source,
                "observation.images.fixed_1",
                int(row["videos/observation.images.fixed_1/chunk_index"]),
                int(row["videos/observation.images.fixed_1/file_index"]),
            )
            fixed_base_ts = float(row["videos/observation.images.fixed/from_timestamp"])
            wrist_base_ts = float(row["videos/observation.images.wrist/from_timestamp"])
            fixed_1_base_ts = float(row["videos/observation.images.fixed_1/from_timestamp"])
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
                fixed_1 = decode_resized_frames(
                    fixed_1_video,
                    fixed_1_base_ts + ep_timestamps[sl],
                    tolerance_s=args.decode_tolerance_s,
                    image_size=(128, 128),
                )

                for local_i in range(end - start):
                    i = start + local_i
                    dataset.add_frame(
                        {
                            "observation.images.fixed": fixed[local_i],
                            "observation.images.wrist": wrist[local_i],
                            "observation.images.fixed_1": fixed_1[local_i],
                            "observation.state": state9[i],
                            "action": action[i],
                            "next.reward": np.array([rewards[i]], dtype=np.float32),
                            "next.done": np.array([dones[i]], dtype=bool),
                        },
                        task=task,
                    )

            dataset.save_episode()
            all_delta_m.append(deltas)
            all_norm_action.append(np.abs(clipped[:, :3]))
            all_ee_pos.append(ee_pos)
            all_episode_lengths.append(len(ep_df))
            total_frames += len(ep_df)
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.stop_image_writer()

    all_delta = np.concatenate(all_delta_m, axis=0)
    all_norm = np.concatenate(all_norm_action, axis=0)
    all_ee = np.concatenate(all_ee_pos, axis=0)
    report = {
        "source": str(source),
        "dest": str(dest),
        "repo_id": args.repo_id,
        "source_codebase_version": info.get("codebase_version"),
        "fps": fps,
        "episodes": int(len(episode_table)),
        "frames": int(total_frames),
        "episode_lengths": all_episode_lengths,
        "urdf": str(urdf),
        "calibration_path": str(calibration_path),
        "delta_label_source": str(args.delta_label_source),
        "delta_horizon": int(args.delta_horizon),
        "ee_scale_m": scale.astype(float).tolist(),
        "gripper_open_threshold": float(args.gripper_open_threshold),
        "reward_neg": float(args.reward_neg),
        "reward_pos": float(args.reward_pos),
        "clip_fraction": float(clip_frames / max(1, total_frames)),
        "ee_delta_m": stats_for_array(all_delta),
        "ee_delta_norm_mm": norm_summary(np.linalg.norm(all_delta, axis=1) * 1000.0),
        "normalized_action_abs": stats_for_array(all_norm),
        "normalized_action_abs_norm": norm_summary(np.linalg.norm(all_norm, axis=1)),
        "ee_pos_m": stats_for_array(all_ee),
    }
    write_json(dest / "conversion_report.json", report)

    print(f"converted {len(episode_table)} episodes / {total_frames} frames", flush=True)
    print(f"wrote {dest}", flush=True)
    print(f"clip_fraction={report['clip_fraction']:.6f}", flush=True)
    print(f"ee_delta_norm_mm={report['ee_delta_norm_mm']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--ee-scale", nargs=3, type=float, default=[0.035, 0.030, 0.060])
    parser.add_argument(
        "--delta-label-source",
        choices=["state_delta", "action_target"],
        default="state_delta",
        help=(
            "state_delta uses FK(state[t+horizon]) - FK(state[t]), matching the "
            "online EE-delta integrator. action_target keeps the old "
            "FK(action[t]) - FK(state[t]) behavior for diagnostics."
        ),
    )
    parser.add_argument(
        "--delta-horizon",
        type=int,
        default=2,
        help="Future-state horizon for state_delta labels. 2 frames maps 30 Hz demos to the 15 Hz actor tick.",
    )
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
