#!/usr/bin/env python3
"""Offline SO101 EE-delta conversion and BC probe.

This is intentionally a standalone diagnostic tool. It does not change the
online HIL-RL learner/actor path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_JOINT_MIN = np.array([-36.0, -107.0, -37.0, 41.0, -46.0], dtype=np.float64)
SO101_JOINT_MAX = np.array([66.0, 45.0, 99.0, 99.0, 65.0], dtype=np.float64)
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CALIBRATION_PATH = REPO_ROOT / "assets/so101/so101_follower_calibration.json"
HF_CALIBRATION_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower.json"
)
DEFAULT_CALIBRATION_PATH = REPO_CALIBRATION_PATH if REPO_CALIBRATION_PATH.exists() else HF_CALIBRATION_PATH
STS3215_RESOLUTION = 4095.0


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def episode_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/*.parquet"))


def action_to_raw_joints(actions: np.ndarray) -> np.ndarray:
    raw = SO101_JOINT_MIN + (actions[..., :5] + 1.0) * 0.5 * (SO101_JOINT_MAX - SO101_JOINT_MIN)
    return np.clip(raw, SO101_JOINT_MIN, SO101_JOINT_MAX)


def load_so101_calibration(path: str | Path) -> dict[str, dict[str, float]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"SO101 calibration file not found: {path}. "
            "EE-delta conversion needs the recording robot calibration because "
            "LeRobot datasets store joints in RANGE_M100_100 while FK/IK expects degrees."
        )
    data = read_json(path)
    calibration = {}
    for name in ARM_JOINT_NAMES:
        row = data[name]
        calibration[name] = {
            "drive_mode": float(row.get("drive_mode", 0)),
            "range_min": float(row["range_min"]),
            "range_max": float(row["range_max"]),
        }
    return calibration


def range_arm_to_degrees(arm_joints: np.ndarray, calibration: dict[str, dict[str, float]]) -> np.ndarray:
    arm_joints = np.asarray(arm_joints, dtype=np.float64).reshape(-1)
    out = np.zeros_like(arm_joints, dtype=np.float64)
    for i, name in enumerate(ARM_JOINT_NAMES):
        cal = calibration[name]
        val = float(np.clip(arm_joints[i], -100.0, 100.0))
        if cal["drive_mode"]:
            val = -val
        raw = ((val + 100.0) / 200.0) * (cal["range_max"] - cal["range_min"]) + cal["range_min"]
        mid = (cal["range_min"] + cal["range_max"]) / 2.0
        out[i] = (raw - mid) * 360.0 / STS3215_RESOLUTION
    return out


def joints_range_to_degrees(joints: np.ndarray, calibration: dict[str, dict[str, float]]) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64).copy()
    joints[:5] = range_arm_to_degrees(joints[:5], calibration)
    return joints


def stats_for_array(values: np.ndarray) -> dict:
    values = np.asarray(values)
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def summarize(name: str, values: np.ndarray, unit: str = "") -> dict:
    values = np.asarray(values, dtype=np.float64)
    flat = values if values.ndim == 1 else np.linalg.norm(values, axis=1)
    out = {
        "mean": float(flat.mean()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
    }
    print(
        f"{name}: mean={out['mean']:.6g}{unit} p50={out['p50']:.6g}{unit} "
        f"p95={out['p95']:.6g}{unit} p99={out['p99']:.6g}{unit} max={out['max']:.6g}{unit}",
        flush=True,
    )
    if values.ndim > 1:
        comp_p95 = np.percentile(values, 95, axis=0)
        comp_max = values.max(axis=0)
        out["component_p95"] = comp_p95.astype(float).tolist()
        out["component_max"] = comp_max.astype(float).tolist()
        print(
            f"{name} comp_p95={np.round(comp_p95, 6).tolist()}{unit} "
            f"comp_max={np.round(comp_max, 6).tolist()}{unit}",
            flush=True,
        )
    return out


def make_kinematics(urdf: Path):
    from lerobot.model.kinematics import RobotKinematics

    return RobotKinematics(
        urdf_path=str(urdf),
        target_frame_name="gripper_frame_link",
        joint_names=ARM_JOINT_NAMES,
    )


def fk(kin, q: np.ndarray, calibration: dict[str, dict[str, float]] | None = None) -> np.ndarray:
    if calibration is not None:
        q = joints_range_to_degrees(q, calibration)
    return kin.forward_kinematics(q).copy()


def iterative_ik(kin, q0: np.ndarray, target: np.ndarray, iters: int = 5, orientation_weight: float = 0.0):
    q = np.asarray(q0, dtype=np.float64).copy()
    for _ in range(iters):
        q = kin.inverse_kinematics(
            q, target, position_weight=1.0, orientation_weight=orientation_weight
        ).copy()
    return q


def convert_dataset(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    scale = np.asarray(args.ee_scale, dtype=np.float64)
    calibration_path = Path(args.calibration_path).expanduser().resolve()
    calibration = load_so101_calibration(calibration_path)

    if dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dest} already exists; pass --overwrite to replace it")
        shutil.rmtree(dest)

    (dest / "data").mkdir(parents=True)
    (dest / "meta").mkdir(parents=True)
    kin = make_kinematics(urdf)

    info = read_json(source / "meta" / "info.json")
    info["robot_type"] = "so101_ee_delta_probe"
    info["features"]["action"]["shape"] = [4]
    info["features"]["action"]["names"] = ["delta_x", "delta_y", "delta_z", "gripper"]
    append_ee_pos = bool(args.append_ee_pos_to_state)
    if append_ee_pos:
        state_feature = info["features"]["observation.state"]
        old_shape = state_feature.get("shape", [])
        if len(old_shape) != 1:
            raise ValueError(f"Expected 1D observation.state shape, got {old_shape}")
        state_feature["shape"] = [int(old_shape[0]) + 3]
        old_names = state_feature.get("names")
        if isinstance(old_names, list):
            state_feature["names"] = [*old_names, "ee_x", "ee_y", "ee_z"]
    write_json(dest / "meta" / "info.json", info)
    shutil.copy2(source / "meta" / "episodes.jsonl", dest / "meta" / "episodes.jsonl")
    shutil.copy2(source / "meta" / "tasks.jsonl", dest / "meta" / "tasks.jsonl")

    source_videos = source / "videos"
    if source_videos.exists():
        if args.copy_videos:
            shutil.copytree(source_videos, dest / "videos")
        else:
            os.symlink(source_videos, dest / "videos", target_is_directory=True)

    old_episode_stats = {
        int(row["episode_index"]): row for row in read_jsonl(source / "meta" / "episodes_stats.jsonl")
    }
    new_episode_stats = []
    all_delta_m = []
    all_delta_norm_m = []
    all_norm_action = []
    all_ee_pos = []
    clip_count = 0
    total_count = 0

    for ep_file in episode_files(source):
        rel = ep_file.relative_to(source)
        out_file = dest / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(ep_file)
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        actions = np.stack(df["action"].to_numpy()).astype(np.float64)
        raw_targets = states[:, :6].copy()
        raw_targets[:, :5] = action_to_raw_joints(actions)

        new_actions = np.zeros((len(df), 4), dtype=np.float32)
        deltas = np.zeros((len(df), 3), dtype=np.float64)
        ee_pos = np.zeros((len(df), 3), dtype=np.float64)
        for i, (q, q_target) in enumerate(zip(states[:, :6], raw_targets)):
            cur = fk(kin, q, calibration)
            target = fk(kin, q_target, calibration)
            delta = target[:3, 3] - cur[:3, 3]
            ee_pos[i] = cur[:3, 3]
            deltas[i] = delta

        normalized = deltas / scale
        if args.clip:
            before = normalized.copy()
            normalized = np.clip(normalized, -0.999, 0.999)
            clip_count += int(np.any(np.abs(before) > 0.999, axis=1).sum())
        new_actions[:, :3] = normalized.astype(np.float32)
        new_actions[:, 3] = np.round(actions[:, 5]).clip(0, 1).astype(np.float32)
        total_count += new_actions.shape[0]

        df["action"] = list(new_actions)
        if append_ee_pos:
            new_states = np.concatenate(
                [states.astype(np.float32), ee_pos.astype(np.float32)],
                axis=1,
            )
            df["observation.state"] = list(new_states)
        df.to_parquet(out_file, index=False)

        ep_idx = int(df["episode_index"].iloc[0])
        row = copy.deepcopy(old_episode_stats[ep_idx])
        row["stats"]["action"] = stats_for_array(new_actions)
        if append_ee_pos:
            row["stats"]["observation.state"] = stats_for_array(new_states)
        new_episode_stats.append(row)
        all_delta_m.append(np.abs(deltas))
        all_delta_norm_m.append(np.linalg.norm(deltas, axis=1))
        all_norm_action.append(np.abs(normalized[:, :3]))
        all_ee_pos.append(ee_pos)

        if (len(new_episode_stats) % 20) == 0:
            print(f"converted episodes={len(new_episode_stats)} frames={total_count}", flush=True)

    write_jsonl(dest / "meta" / "episodes_stats.jsonl", new_episode_stats)

    all_delta_m = np.concatenate(all_delta_m, axis=0)
    all_delta_norm_m = np.concatenate(all_delta_norm_m, axis=0)
    all_norm_action = np.concatenate(all_norm_action, axis=0)
    all_ee_pos = np.concatenate(all_ee_pos, axis=0)
    report = {
        "source": str(source),
        "dest": str(dest),
        "urdf": str(urdf),
        "calibration_path": str(calibration_path),
        "ee_scale_m": scale.astype(float).tolist(),
        "append_ee_pos_to_state": append_ee_pos,
        "state_dim": int(info["features"]["observation.state"]["shape"][0]),
        "clip": bool(args.clip),
        "clip_fraction": float(clip_count / max(1, total_count)),
        "frames": int(total_count),
        "ee_delta_abs_m": summarize("EE delta abs", all_delta_m * 1000.0, unit="mm"),
        "ee_delta_norm_m": summarize("EE delta norm", all_delta_norm_m * 1000.0, unit="mm"),
        "ee_pos_m": {
            "min": all_ee_pos.min(axis=0).astype(float).tolist(),
            "max": all_ee_pos.max(axis=0).astype(float).tolist(),
            "mean": all_ee_pos.mean(axis=0).astype(float).tolist(),
        },
        "normalized_abs": summarize("normalized EE action abs", all_norm_action),
    }
    print(f"clip_fraction={report['clip_fraction']:.6g}", flush=True)
    write_json(dest / "conversion_report.json", report)
    print(f"wrote {dest}", flush=True)


def bc_eval(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from lerobot.configs.types import NormalizationMode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import make_policy
    from lerobot.policies.silri.configuration_silri import (
        ActorNetworkConfig,
        CriticNetworkConfig,
        PolicyConfig,
        SiLRIConfig,
    )

    root = Path(args.dataset).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id or root.name, root=root)
    features = dataset_to_policy_features(meta.features)
    if args.state_only:
        input_features = {"observation.state": features["observation.state"]}
    else:
        input_features = {k: v for k, v in features.items() if k != "action"}
    output_features = {"action": features["action"]}

    cfg = SiLRIConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={"VISUAL": NormalizationMode.MEAN_STD},
        dataset_stats=None if args.state_only else {
            "observation.images.fixed": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "observation.images.wrist": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        device=args.device,
        storage_device=args.device,
        vision_encoder_name=None if args.state_only else args.vision_encoder,
        freeze_vision_encoder=True,
        image_encoder_hidden_dim=32,
        image_embedding_pooling_dim=8,
        shared_encoder=False,
        num_discrete_actions=2,
        actor_lr=args.lr,
        critic_lr=args.lr,
        temperature_lr=args.lr,
        reward_classifier_lr=1e-3,
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[256, 256], activate_final=True),
        critic_network_kwargs=CriticNetworkConfig(hidden_dims=[256, 256], activate_final=True),
        discrete_actor_network_kwargs=ActorNetworkConfig(hidden_dims=[256, 256], activate_final=True),
        discrete_critic_network_kwargs=CriticNetworkConfig(hidden_dims=[256, 256], activate_final=True),
        policy_kwargs=PolicyConfig(use_tanh_squash=True, std_min=-5.0, std_max=2.0, init_final=0.05),
        state_encoder_hidden_dim=256,
        latent_dim=256,
        use_torch_compile=False,
    )

    episodes = sorted(meta.episodes)
    val_eps = [ep for i, ep in enumerate(episodes) if (i % args.val_every) == args.val_every - 1]
    train_eps = [ep for ep in episodes if ep not in set(val_eps)]

    class StateActionDataset(Dataset):
        def __init__(self, root_path: Path, metadata: LeRobotDatasetMetadata, eps: list[int]):
            states = []
            actions = []
            for ep in eps:
                df = pd.read_parquet(
                    root_path / metadata.get_data_file_path(ep),
                    columns=["observation.state", "action"],
                )
                states.append(np.stack(df["observation.state"].to_numpy()).astype(np.float32))
                actions.append(np.stack(df["action"].to_numpy()).astype(np.float32))
            self.states = torch.from_numpy(np.concatenate(states, axis=0))
            self.actions = torch.from_numpy(np.concatenate(actions, axis=0))

        def __len__(self):
            return int(self.actions.shape[0])

        def __getitem__(self, idx):
            return {"observation.state": self.states[idx], "action": self.actions[idx]}

    if args.state_only:
        train_ds = StateActionDataset(root, meta, train_eps)
        val_ds = StateActionDataset(root, meta, val_eps)
    else:
        train_ds = LeRobotDataset(args.repo_id or root.name, root=root, episodes=train_eps, video_backend=args.video_backend)
        val_ds = LeRobotDataset(args.repo_id or root.name, root=root, episodes=val_eps, video_backend=args.video_backend)

    policy = make_policy(cfg, ds_meta=meta)
    policy.train()
    cont_dim = policy.continuous_action_dim
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    params = list(policy.actor.parameters())
    if getattr(policy, "discrete_actor", None) is not None:
        params += list(policy.discrete_actor.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    def to_obs(batch: dict) -> dict:
        return {k: batch[k].to(device, non_blocking=True).float() for k in input_features}

    def to_action(batch: dict) -> torch.Tensor:
        actions = batch["action"].to(device, non_blocking=True).float()
        actions = actions.clone()
        actions[:, :cont_dim] = actions[:, :cont_dim].clamp(-0.999, 0.999)
        if actions.shape[1] > cont_dim:
            actions[:, cont_dim:] = actions[:, cont_dim:].round().clamp(0, 1)
        return actions

    train_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        obs = to_obs(batch)
        actions = to_action(batch)
        observation_features = None
        if policy.actor.encoder.has_images and policy.config.freeze_vision_encoder:
            with torch.no_grad():
                observation_features = policy.actor.encoder.get_cached_image_features(obs, normalize=True)
        forward_batch = {
            "state": obs,
            "action": actions,
            "is_intervention": torch.ones(actions.shape[0], device=device),
            "observation_feature": observation_features,
        }
        loss = policy.forward(forward_batch, model="actor_bc")["loss_actor_bc"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(f"train step={step} loss={float(loss.detach().cpu()):.6g}", flush=True)

    metrics = evaluate_policy(
        args=args,
        policy=policy,
        val_loader=val_loader,
        input_features=input_features,
        cont_dim=cont_dim,
        device=device,
        urdf=urdf,
    )
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.action_mode}_bc_metrics.json", metrics)
    torch.save({"actor": policy.actor.state_dict(), "discrete_actor": policy.discrete_actor.state_dict()}, out_dir / f"{args.action_mode}_actor_bc.pth")
    print(f"saved metrics/model to {out_dir}", flush=True)


def evaluate_policy(*, args, policy, val_loader, input_features, cont_dim, device, urdf: Path) -> dict:
    import torch

    kin = make_kinematics(urdf)
    scale = np.asarray(args.ee_scale, dtype=np.float64)
    calibration = load_so101_calibration(args.calibration_path)
    policy.eval()
    norm_abs_err = []
    norm_sq_err = []
    grip_ok = []
    ee_err_mm = []
    pred_step = []
    target_step = []
    pred_ee_norm_mm = []
    target_ee_norm_mm = []
    n_seen = 0

    def obs_from_batch(batch: dict) -> dict:
        return {k: batch[k].to(device, non_blocking=True).float() for k in input_features}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_eval_batches and batch_idx >= args.max_eval_batches:
                break
            obs = obs_from_batch(batch)
            target = batch["action"].to(device, non_blocking=True).float()
            pred, _ = policy.select_action(obs)
            pred_cont = pred[:, :cont_dim].clamp(-0.999, 0.999)
            target_cont = target[:, :cont_dim].clamp(-0.999, 0.999)
            err = (pred_cont - target_cont).detach().cpu().numpy()
            norm_abs_err.append(np.abs(err))
            norm_sq_err.append(err**2)
            if target.shape[1] > cont_dim:
                grip_ok.append(
                    (pred[:, cont_dim:].round().cpu().numpy() == target[:, cont_dim:].round().cpu().numpy())
                )

            states = batch["observation.state"].numpy().astype(np.float64)
            pred_np = pred.detach().cpu().numpy().astype(np.float64)
            target_np = target.detach().cpu().numpy().astype(np.float64)
            for q, pa, ta in zip(states[:, :6], pred_np, target_np):
                if n_seen >= args.max_eval_physical:
                    continue
                n_seen += 1
                cur = fk(kin, q, calibration)
                if args.action_mode == "absolute":
                    q_pred = q.copy()
                    q_target = q.copy()
                    q_pred[:5] = action_to_raw_joints(pa[None, :])[0]
                    q_target[:5] = action_to_raw_joints(ta[None, :])[0]
                    pred_target = fk(kin, q_pred, calibration)
                    demo_target = fk(kin, q_target, calibration)
                    q_deg = joints_range_to_degrees(q, calibration)
                    q_pred_deg = joints_range_to_degrees(q_pred, calibration)
                    q_target_deg = joints_range_to_degrees(q_target, calibration)
                    pred_step.append(np.linalg.norm(q_pred_deg[:5] - q_deg[:5]))
                    target_step.append(np.linalg.norm(q_target_deg[:5] - q_deg[:5]))
                    pred_delta = pred_target[:3, 3] - cur[:3, 3]
                    demo_delta = demo_target[:3, 3] - cur[:3, 3]
                else:
                    pred_delta = np.clip(pa[:3], -0.999, 0.999) * scale
                    demo_delta = np.clip(ta[:3], -0.999, 0.999) * scale
                    pred_pose = cur.copy()
                    pred_pose[:3, 3] = cur[:3, 3] + pred_delta
                    target_pose = cur.copy()
                    target_pose[:3, 3] = cur[:3, 3] + demo_delta
                    q_deg = joints_range_to_degrees(q, calibration)
                    q_pred = iterative_ik(kin, q_deg, pred_pose, iters=args.ik_iters, orientation_weight=0.0)
                    q_target = iterative_ik(kin, q_deg, target_pose, iters=args.ik_iters, orientation_weight=0.0)
                    pred_step.append(np.linalg.norm(q_pred[:5] - q_deg[:5]))
                    target_step.append(np.linalg.norm(q_target[:5] - q_deg[:5]))
                ee_err_mm.append(np.linalg.norm(pred_delta - demo_delta) * 1000.0)
                pred_ee_norm_mm.append(np.linalg.norm(pred_delta) * 1000.0)
                target_ee_norm_mm.append(np.linalg.norm(demo_delta) * 1000.0)

    norm_abs_err = np.concatenate(norm_abs_err, axis=0)
    norm_sq_err = np.concatenate(norm_sq_err, axis=0)
    metrics = {
        "action_mode": args.action_mode,
        "dataset": str(Path(args.dataset).resolve()),
        "val_every": int(args.val_every),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "normalized_continuous_mae": norm_abs_err.mean(axis=0).astype(float).tolist(),
        "normalized_continuous_rmse": np.sqrt(norm_sq_err.mean(axis=0)).astype(float).tolist(),
        "normalized_continuous_mae_mean": float(norm_abs_err.mean()),
        "normalized_continuous_rmse_mean": float(np.sqrt(norm_sq_err.mean())),
        "physical_eval_frames": int(n_seen),
        "ee_error_mm": percentile_dict(ee_err_mm),
        "pred_joint_step_deg": percentile_dict(pred_step),
        "target_joint_step_deg": percentile_dict(target_step),
        "pred_ee_norm_mm": percentile_dict(pred_ee_norm_mm),
        "target_ee_norm_mm": percentile_dict(target_ee_norm_mm),
    }
    if grip_ok:
        metrics["gripper_accuracy"] = float(np.concatenate(grip_ok, axis=0).mean())

    print("=== BC EVAL ===", flush=True)
    for key in [
        "normalized_continuous_mae_mean",
        "normalized_continuous_rmse_mean",
        "gripper_accuracy",
    ]:
        if key in metrics:
            print(f"{key}: {metrics[key]}", flush=True)
    for key in ["ee_error_mm", "pred_joint_step_deg", "target_joint_step_deg", "pred_ee_norm_mm"]:
        print(f"{key}: {metrics[key]}", flush=True)
    return metrics


def percentile_dict(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    conv = sub.add_parser("convert")
    conv.add_argument("--source", required=True)
    conv.add_argument("--dest", required=True)
    conv.add_argument("--urdf", required=True)
    conv.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH))
    conv.add_argument("--ee-scale", nargs=3, type=float, default=[0.035, 0.030, 0.060])
    conv.add_argument("--clip", action="store_true")
    conv.add_argument("--copy-videos", action="store_true")
    conv.add_argument("--append-ee-pos-to-state", action="store_true")
    conv.add_argument("--overwrite", action="store_true")
    conv.set_defaults(func=convert_dataset)

    bc = sub.add_parser("bc-eval")
    bc.add_argument("--dataset", required=True)
    bc.add_argument("--urdf", required=True)
    bc.add_argument("--calibration-path", default=str(DEFAULT_CALIBRATION_PATH))
    bc.add_argument("--action-mode", choices=["absolute", "ee_delta"], required=True)
    bc.add_argument("--repo-id", default=None)
    bc.add_argument("--output-dir", default="outputs/so101_ee_delta_bc_probe")
    bc.add_argument("--device", default="cuda")
    bc.add_argument("--steps", type=int, default=800)
    bc.add_argument("--batch-size", type=int, default=256)
    bc.add_argument("--num-workers", type=int, default=4)
    bc.add_argument("--log-every", type=int, default=100)
    bc.add_argument("--lr", type=float, default=3e-4)
    bc.add_argument("--grad-clip", type=float, default=40.0)
    bc.add_argument("--val-every", type=int, default=5)
    bc.add_argument("--video-backend", default="pyav")
    bc.add_argument("--vision-encoder", default="helper2424/resnet10")
    bc.add_argument("--state-only", action="store_true")
    bc.add_argument("--ee-scale", nargs=3, type=float, default=[0.035, 0.030, 0.060])
    bc.add_argument("--ik-iters", type=int, default=5)
    bc.add_argument("--max-eval-physical", type=int, default=5000)
    bc.add_argument("--max-eval-batches", type=int, default=0)
    bc.set_defaults(func=bc_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
