from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


def _now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_front_image(obs: Any, camera_key: str) -> np.ndarray | None:
    if not isinstance(obs, dict):
        return None
    images = obs.get("images")
    if not isinstance(images, dict):
        return None
    img = images.get(camera_key)
    if not isinstance(img, np.ndarray):
        return None
    if img.dtype != np.uint8:
        return None
    if img.ndim != 3 or img.shape[-1] != 3:
        return None
    return img


def _write_video(frames_rgb: list[np.ndarray], out_path: Path, fps: int) -> None:
    if len(frames_rgb) == 0:
        return

    # Prefer OpenCV (fast, common)
    try:
        import cv2  # type: ignore

        h, w = frames_rgb[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        try:
            for frame in frames_rgb:
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                bgr = frame[:, :, ::-1]
                writer.write(bgr)
        finally:
            writer.release()
        return
    except Exception:
        pass

    # Fallback to imageio (may require ffmpeg)
    try:
        import imageio  # type: ignore

        with imageio.get_writer(str(out_path), fps=fps) as writer:
            for frame in frames_rgb:
                writer.append_data(frame)
        return
    except Exception as e:
        raise RuntimeError(
            "No available video writer backend. Install opencv-python or imageio[ffmpeg]. "
            f"Tried cv2 and imageio, last error: {e}"
        )


@dataclasses.dataclass
class EvalRecordConfig:
    output_dir: Path = Path("outputs/eval_records")
    fps: int = 30
    camera_key: str = "front"
    success_info_key: str = "success"
    file_prefix: str = "episode"


class EvalRecordWrapper(gym.Wrapper):
    """Record evaluation episodes as front-camera videos and track mean success rate.

    Expected obs schema in this repo: obs["images"]["front"] is uint8 HxWx3.
    Success is read from `info[success_info_key]` on episode termination.
    """

    def __init__(self, env: gym.Env, *, config: EvalRecordConfig | None = None):
        super().__init__(env)
        self.cfg = config or EvalRecordConfig()
        self.root = Path(self.cfg.output_dir)
        _ensure_dir(self.root)

        self.run_dir = self.root / _now_compact()
        _ensure_dir(self.run_dir)
        _ensure_dir(self.run_dir / "videos")

        self._episode_idx = -1
        self._frames: list[np.ndarray] = []
        self._episode_finalized = True
        self._success_total = 0
        self._episode_total = 0
        self._episode_labeled = 0
        self._warned_missing_success = False

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._write_summary()
        print(f"[EvalRecordWrapper] Output dir: {self.run_dir}")

    @property
    def success_rate(self) -> float:
        if self._episode_labeled <= 0:
            return 0.0
        return float(self._success_total) / float(self._episode_labeled)

    def _write_summary(self) -> None:
        summary = {
            "episodes": self._episode_total,
            "episodes_labeled": self._episode_labeled,
            "successes": self._success_total,
            "success_rate": self.success_rate,
            "camera_key": self.cfg.camera_key,
            "fps": int(self.cfg.fps),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _finalize_episode(self, terminal_info: dict[str, Any]) -> None:
        if self._episode_idx < 0:
            return
        if self._episode_finalized:
            return

        video_name = f"{self.cfg.file_prefix}_{self._episode_idx:06d}.mp4"
        video_path = self.run_dir / "videos" / video_name
        _write_video(self._frames, video_path, int(self.cfg.fps))

        success_val = terminal_info.get(self.cfg.success_info_key, None)
        success = success_val if isinstance(success_val, bool) else None
        if success is None and not self._warned_missing_success:
            # Don't spam logs; just warn once.
            self._warned_missing_success = True
            print(
                f"[EvalRecordWrapper] Warning: missing or non-bool info['{self.cfg.success_info_key}'] at episode end. "
                "This episode will be excluded from success rate."
            )
        if success is not None:
            self._episode_labeled += 1
            if success is True:
                self._success_total += 1
        self._episode_total += 1
        self._write_summary()

        record = {
            "episode": self._episode_idx,
            "frames": len(self._frames),
            "video": os.path.relpath(video_path, start=self.run_dir),
            "success": success,
            "episodes": self._episode_total,
            "episodes_labeled": self._episode_labeled,
            "success_rate": self.success_rate,
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[EvalRecordWrapper] Episode {self._episode_idx} saved: {video_path.name} | "
            f"success={success} | mean_success={self.success_rate:.3f} ({self._success_total}/{self._episode_labeled})"
        )

        # Mark finalized and clear frames so reset() won't finalize again.
        self._episode_finalized = True
        self._frames = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # If user resets early without termination, finalize with empty info.
        if self._episode_idx >= 0 and (not self._episode_finalized) and len(self._frames) > 0:
            self._finalize_episode({})

        self._episode_idx += 1
        self._frames = []
        self._episode_finalized = False
        obs, info = self.env.reset(seed=seed, options=options)
        img = _get_front_image(obs, self.cfg.camera_key)
        if img is not None:
            self._frames.append(img)
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        img = _get_front_image(obs, self.cfg.camera_key)
        if img is not None:
            self._frames.append(img)
        if bool(terminated) or bool(truncated):
            self._finalize_episode(info)
        return obs, reward, terminated, truncated, info
