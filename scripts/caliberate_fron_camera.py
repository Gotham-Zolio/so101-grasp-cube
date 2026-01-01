"""Camera vs video first-frame alignment viewer.

Reads the first (or specified) frame from an MP4 video as a reference, then opens a
live camera feed and provides a real-time visualization to compare whether the
current camera content matches the reference using overlay and black-line overlap.

Typical use:
  python caliberate_fron_camera.py --video path/to/ref.mp4 --camera 0

Controls:
  - Trackbars in the UI control blending/threshold.
  - Keys: q/ESC quit, space freeze/unfreeze, s save snapshot.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ViewConfig:
	alpha: float = 0.5
	black_thresh: int = 80
	morph_ksize: int = 3
	blur_ksize: int = 3
	show_edges: bool = False
	canny1: int = 60
	canny2: int = 160


def _odd_at_least_1(value: int) -> int:
	value = int(value)
	if value <= 1:
		return 1
	return value if value % 2 == 1 else value + 1


def read_video_frame(video_path: str, frame_index: int = 0) -> np.ndarray:
	path = Path(video_path)
	suffix = path.suffix.lower()
	if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
		img = cv2.imread(str(path), cv2.IMREAD_COLOR)
		if img is None:
			raise RuntimeError(f"Failed to read image: {video_path}")
		return img

	# For videos: prefer FFMPEG backend and try to disable hardware decoding.
	cap_backend = cv2.CAP_FFMPEG if hasattr(cv2, "CAP_FFMPEG") else 0
	cap = cv2.VideoCapture(str(path), cap_backend)

	# Try to force software decoding when supported.
	# 1) OpenCV property (newer builds)
	try:
		accel_none = getattr(cv2, "VIDEO_ACCELERATION_NONE", None)
		if accel_none is not None and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
			cap.set(cv2.CAP_PROP_HW_ACCELERATION, float(accel_none))
	except Exception:
		pass
	# 2) FFmpeg capture options via env var (works with OpenCV+FFmpeg builds)
	#    This must be set BEFORE opening capture to be guaranteed, but setting it here
	#    still helps if this function is the first video open.
	#    If user already set it externally, we won't override.
	if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
		os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;none"

	if not cap.isOpened():
		raise RuntimeError(
			"Failed to open video. If this is an AV1 MP4 and you see 'hardware accelerated AV1 decoding' errors, "
			"extract the first frame to an image and pass it as --video, or re-encode to H.264."
		)

	if frame_index != 0:
		cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))

	ok, frame = cap.read()
	cap.release()
	if not ok or frame is None:
		raise RuntimeError(
			f"Failed to read frame {frame_index} from: {video_path}. "
			"If this video is AV1 and your OpenCV/FFmpeg build can't decode it, use:\n"
			"  ffmpeg -y -i input.mp4 -vf 'select=eq(n\\,0)' -vframes 1 ref.png\n"
			"Then run this script with: --video ref.png"
		)
	return frame


def open_camera(camera: str, width: Optional[int], height: Optional[int], fps: Optional[int]) -> cv2.VideoCapture:
	# Allow passing either an integer index or a device path.
	try:
		cam_id: int | str = int(camera)
	except ValueError:
		cam_id = camera

	cap = cv2.VideoCapture(cam_id)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open camera: {camera}")

	if width is not None:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
	if height is not None:
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
	if fps is not None:
		cap.set(cv2.CAP_PROP_FPS, float(fps))

	return cap


def _apply_transform(frame: np.ndarray, rotate: int, flip: int) -> np.ndarray:
	out = frame
	if rotate % 360 != 0:
		rot = rotate % 360
		if rot == 90:
			out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
		elif rot == 180:
			out = cv2.rotate(out, cv2.ROTATE_180)
		elif rot == 270:
			out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
		else:
			raise ValueError("rotate must be one of 0, 90, 180, 270")

	if flip != 2:
		# OpenCV: 0 vertical, 1 horizontal, -1 both.
		out = cv2.flip(out, flip)
	return out


def _center_fit(reference: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
	th, tw = target_hw
	rh, rw = reference.shape[:2]
	if (rh, rw) == (th, tw):
		return reference
	return cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)


def black_mask(bgr: np.ndarray, cfg: ViewConfig) -> np.ndarray:
	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	blur_k = _odd_at_least_1(cfg.blur_ksize)
	if blur_k > 1:
		gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

	# black pixels: gray < thresh
	mask = (gray < int(cfg.black_thresh)).astype(np.uint8) * 255

	k = int(cfg.morph_ksize)
	if k > 1:
		k = max(1, k)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	return mask


def color_overlap(cam_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
	"""Return a BGR image showing overlap.

	- green: both black
	- red: cam-only black
	- blue: ref-only black
	"""
	cam_on = cam_mask > 0
	ref_on = ref_mask > 0
	both = cam_on & ref_on
	cam_only = cam_on & (~ref_on)
	ref_only = ref_on & (~cam_on)

	h, w = cam_mask.shape[:2]
	out = np.zeros((h, w, 3), dtype=np.uint8)
	out[both] = (0, 255, 0)
	out[cam_only] = (0, 0, 255)
	out[ref_only] = (255, 0, 0)
	return out


def _make_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Overlay live camera with MP4 reference frame to check alignment.")
	p.add_argument("--video", required=True, help="Path to reference MP4 video, or a reference image (png/jpg)")
	p.add_argument("--frame", type=int, default=0, help="Reference frame index (default: 0)")
	p.add_argument("--camera", default="0", help="Camera index (e.g. 0) or device path")
	p.add_argument("--width", type=int, default=None, help="Request camera width")
	p.add_argument("--height", type=int, default=None, help="Request camera height")
	p.add_argument("--fps", type=int, default=None, help="Request camera fps")
	p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate both views")
	p.add_argument(
		"--flip",
		type=int,
		default=2,
		choices=[-1, 0, 1, 2],
		help="Flip both views: -1 both, 0 vertical, 1 horizontal, 2 disable",
	)
	p.add_argument(
		"--ffmpeg-hwaccel",
		default="none",
		help="Set OPENCV_FFMPEG_CAPTURE_OPTIONS hwaccel value (default: none). Use '' to disable setting.",
	)
	p.add_argument("--save-dir", default="outputs", help="Directory to save snapshots")
	return p.parse_args()


def main() -> int:
	args = _make_args()
	video_path = str(Path(args.video))
	if args.ffmpeg_hwaccel != "":
		# Ensure video decode doesn't try unsupported hardware paths.
		# User can override by exporting OPENCV_FFMPEG_CAPTURE_OPTIONS externally.
		os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"hwaccel;{args.ffmpeg_hwaccel}")

	ref = read_video_frame(video_path, args.frame)
	cap = open_camera(args.camera, args.width, args.height, args.fps)

	cfg = ViewConfig()
	frozen = False
	frozen_frame: Optional[np.ndarray] = None

	cv2.namedWindow("overlay", cv2.WINDOW_NORMAL)
	cv2.namedWindow("black_overlap", cv2.WINDOW_NORMAL)
	cv2.namedWindow("help", cv2.WINDOW_NORMAL)

	def _on_alpha(x: int) -> None:
		cfg.alpha = float(np.clip(x, 0, 100)) / 100.0

	def _on_thresh(x: int) -> None:
		cfg.black_thresh = int(np.clip(x, 0, 255))

	def _on_morph(x: int) -> None:
		cfg.morph_ksize = int(np.clip(x, 1, 31))

	def _on_blur(x: int) -> None:
		cfg.blur_ksize = int(np.clip(x, 1, 31))

	def _on_edges(x: int) -> None:
		cfg.show_edges = bool(x)

	def _on_c1(x: int) -> None:
		cfg.canny1 = int(np.clip(x, 0, 500))

	def _on_c2(x: int) -> None:
		cfg.canny2 = int(np.clip(x, 0, 500))

	cv2.createTrackbar("alpha(cam->ref)", "overlay", int(cfg.alpha * 100), 100, _on_alpha)
	cv2.createTrackbar("black_thresh", "black_overlap", cfg.black_thresh, 255, _on_thresh)
	cv2.createTrackbar("morph_ksize", "black_overlap", cfg.morph_ksize, 31, _on_morph)
	cv2.createTrackbar("blur_ksize", "black_overlap", cfg.blur_ksize, 31, _on_blur)
	cv2.createTrackbar("show_edges", "black_overlap", 1 if cfg.show_edges else 0, 1, _on_edges)
	cv2.createTrackbar("canny1", "black_overlap", cfg.canny1, 500, _on_c1)
	cv2.createTrackbar("canny2", "black_overlap", cfg.canny2, 500, _on_c2)

	help_img = np.zeros((240, 840, 3), dtype=np.uint8)
	help_lines = [
		"Keys:",
		"  q / ESC : quit",
		"  space   : freeze / unfreeze",
		"  s       : save snapshot (overlay + masks)",
		"Legend (black_overlap window): green=both, red=cam only, blue=ref only",
		"Tip: adjust black_thresh so only the black lines remain.",
	]
	y = 28
	for line in help_lines:
		cv2.putText(help_img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
		y += 32
	cv2.imshow("help", help_img)

	save_dir = Path(args.save_dir)
	save_dir.mkdir(parents=True, exist_ok=True)

	try:
		while True:
			if not frozen:
				ok, cam = cap.read()
				if not ok or cam is None:
					print("[WARN] failed to read camera frame")
					time.sleep(0.05)
					continue
				frozen_frame = cam

			assert frozen_frame is not None
			cam = frozen_frame

			cam = _apply_transform(cam, args.rotate, args.flip)
			ref_t = _apply_transform(ref, args.rotate, args.flip)
			target_hw = (int(cam.shape[0]), int(cam.shape[1]))
			ref_fit = _center_fit(ref_t, target_hw)

			alpha = float(np.clip(cfg.alpha, 0.0, 1.0))
			overlay = cv2.addWeighted(cam, alpha, ref_fit, 1.0 - alpha, 0.0)

			cam_m = black_mask(cam, cfg)
			ref_m = black_mask(ref_fit, cfg)
			overlap = color_overlap(cam_m, ref_m)

			if cfg.show_edges:
				cam_e = cv2.Canny(cam_m, cfg.canny1, cfg.canny2)
				ref_e = cv2.Canny(ref_m, cfg.canny1, cfg.canny2)
				edges = color_overlap(cam_e, ref_e)
				# Stack overlap + edges for easier tuning.
				overlap_show = np.concatenate([overlap, edges], axis=1)
			else:
				overlap_show = overlap

			cv2.imshow("overlay", overlay)
			cv2.imshow("black_overlap", overlap_show)

			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
			if key == ord(" "):
				frozen = not frozen
			if key == ord("s"):
				ts = time.strftime("%Y%m%d-%H%M%S")
				cv2.imwrite(str(save_dir / f"overlay_{ts}.png"), overlay)
				cv2.imwrite(str(save_dir / f"cam_mask_{ts}.png"), cam_m)
				cv2.imwrite(str(save_dir / f"ref_mask_{ts}.png"), ref_m)
				cv2.imwrite(str(save_dir / f"black_overlap_{ts}.png"), overlap_show)
				print(f"[OK] saved snapshots to: {save_dir}")

	finally:
		cap.release()
		cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	main()
