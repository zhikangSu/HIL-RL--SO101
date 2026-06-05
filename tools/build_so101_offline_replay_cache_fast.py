#!/usr/bin/env python
"""Fast offline replay cache builder for SO101 LeRobot datasets.

This mirrors learner.initialize_offline_replay_buffer(..., optimize_memory=True)
but avoids ReplayBuffer.from_lerobot_dataset's per-frame VideoFrame decode path.
It reads each episode parquet once and decodes each episode video sequentially
with OpenCV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lerobot.datasets.video_utils import decode_video_frames


DEFAULT_STATE_KEYS = [
    "observation.images.fixed",
    "observation.images.wrist",
    "observation.state",
]


def cache_path(
    output_dir: Path,
    repo_id: str,
    dataset_root: Path,
    state_keys: list[str],
    capacity: int,
    version: int,
) -> tuple[Path, dict]:
    metadata = {
        "version": version,
        "repo_id": repo_id,
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "state_keys": state_keys,
        "capacity": int(capacity),
        "optimize_memory": True,
    }
    digest = hashlib.sha1(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_repo_id = repo_id.replace("/", "_")
    return output_dir / "offline_replay_cache" / f"{safe_repo_id}_{digest}.pt", metadata


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))


def video_path(dataset_root: Path, info: dict, episode_index: int, video_key: str) -> Path:
    episode_chunk = episode_index // 1000
    path = info["video_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
        video_key=video_key,
    )
    return dataset_root / path


def decode_video_rgb_chw(
    video_file: Path,
    expected_frames: int,
    *,
    fps: int,
    video_backend: str,
    tolerance_s: float,
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_file))
    if cap.isOpened():
        frames = torch.empty((expected_frames, 3, 128, 128), dtype=torch.float32)
        count = 0
        try:
            while count < expected_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_bgr.shape[0] != 128 or frame_bgr.shape[1] != 128:
                    frame_bgr = cv2.resize(frame_bgr, (128, 128), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
                frames[count].copy_(frame)
                count += 1
        finally:
            cap.release()
        if count == expected_frames:
            return frames
        print(
            f"OpenCV decoded {count}/{expected_frames} frames from {video_file}; falling back to {video_backend}",
            flush=True,
        )

    timestamps = [i / fps for i in range(expected_frames)]
    frames = decode_video_frames(video_file, timestamps, tolerance_s=tolerance_s, backend=video_backend)
    if frames.shape[0] != expected_frames:
        raise RuntimeError(f"{video_file} yielded {frames.shape[0]} frames, expected {expected_frames}")
    if frames.shape[-2:] != (128, 128):
        frames = F.interpolate(frames, size=(128, 128), mode="area")
    return frames.contiguous().float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="cube_so101_99demo_ee_delta_state9")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--capacity", type=int, default=55000)
    parser.add_argument("--cache-version", type=int, default=2)
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--decode-tolerance-s", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    state_keys = list(args.state_keys)
    out_path, metadata = cache_path(
        output_dir=output_dir,
        repo_id=args.repo_id,
        dataset_root=dataset_root,
        state_keys=state_keys,
        capacity=args.capacity,
        version=args.cache_version,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"cache already exists: {out_path}", flush=True)
        return

    info = read_json(dataset_root / "meta" / "info.json")
    fps = int(info.get("fps", 30))
    files = episode_files(dataset_root)
    if not files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_root / 'data'}")

    row_counts = []
    total = 0
    for f in files:
        rows = len(pd.read_parquet(f, columns=["episode_index"]))
        row_counts.append(rows)
        total += rows
    if total > args.capacity:
        raise ValueError(f"dataset has {total} frames, capacity is only {args.capacity}")

    print(f"dataset={dataset_root}", flush=True)
    print(f"episodes={len(files)} frames={total} capacity={args.capacity}", flush=True)
    print(f"cache={out_path}", flush=True)

    states: dict[str, torch.Tensor] = {}
    if "observation.images.fixed" in state_keys:
        states["observation.images.fixed"] = torch.empty((total, 3, 128, 128), dtype=torch.float32)
    if "observation.images.wrist" in state_keys:
        states["observation.images.wrist"] = torch.empty((total, 3, 128, 128), dtype=torch.float32)
    if "observation.state" in state_keys:
        state_dim = int(info["features"]["observation.state"]["shape"][0])
        states["observation.state"] = torch.empty((total, state_dim), dtype=torch.float32)

    action_dim = int(info["features"]["action"]["shape"][0])
    actions = torch.empty((total, action_dim), dtype=torch.float32)
    rewards = torch.empty((total,), dtype=torch.float32)
    dones = torch.empty((total,), dtype=torch.bool)
    truncateds = torch.zeros((total,), dtype=torch.bool)
    episode_ends = torch.zeros((total,), dtype=torch.bool)

    offset = 0
    for f, rows in tqdm(list(zip(files, row_counts)), desc="episodes"):
        df = pd.read_parquet(
            f,
            columns=["observation.state", "action", "next.reward", "next.done", "episode_index"],
        )
        ep_idx = int(df["episode_index"].iloc[0])
        sl = slice(offset, offset + rows)

        if "observation.state" in states:
            states["observation.state"][sl].copy_(
                torch.from_numpy(np.stack(df["observation.state"].to_numpy()).astype(np.float32))
            )
        actions[sl].copy_(torch.from_numpy(np.stack(df["action"].to_numpy()).astype(np.float32)))
        rewards[sl].copy_(torch.from_numpy(df["next.reward"].to_numpy(dtype=np.float32)))
        dones[sl].copy_(torch.from_numpy(df["next.done"].to_numpy(dtype=np.bool_)))

        for key in ("observation.images.fixed", "observation.images.wrist"):
            if key in states:
                vid = video_path(dataset_root, info, ep_idx, key)
                states[key][sl].copy_(
                    decode_video_rgb_chw(
                        vid,
                        rows,
                        fps=fps,
                        video_backend=args.video_backend,
                        tolerance_s=args.decode_tolerance_s,
                    )
                )

        offset += rows

    payload = {
        "metadata": metadata,
        "size": int(total),
        "state_keys": state_keys,
        "optimize_memory": True,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "truncateds": truncateds,
        "episode_ends": episode_ends,
        "has_complementary_info": False,
        "complementary_info_keys": [],
        "complementary_info": {},
    }

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
