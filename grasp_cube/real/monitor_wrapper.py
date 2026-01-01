from __future__ import annotations

import base64
import dataclasses
import io
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


def _sanitize_for_json(obj: Any) -> Any:
    """Best-effort conversion of common ML/array types into JSON-serializable primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalars
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)

    # torch tensors (avoid hard dependency)
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            try:
                return obj.detach().cpu().numpy().tolist()
            except Exception:
                return str(obj)
    except Exception:
        pass

    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return base64.b64encode(bytes(obj)).decode("ascii")

    # containers
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _sanitize_for_json(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]

    # dataclasses
    if dataclasses.is_dataclass(obj):
        try:
            return _sanitize_for_json(dataclasses.asdict(obj))
        except Exception:
            return str(obj)

    # fallback
    return str(obj)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(_sanitize_for_json(payload)).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        length = 0
    if length <= 0:
        return None
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _try_encode_image_to_jpeg_b64(image: np.ndarray, *, quality: int = 60) -> str | None:
    if not isinstance(image, np.ndarray):
        return None
    if image.dtype != np.uint8:
        return None
    if image.ndim != 3 or image.shape[-1] != 3:
        return None

    # Try OpenCV first (likely present in some lerobot installs)
    try:
        import cv2  # type: ignore

        # Convert RGB->BGR for OpenCV encoding
        bgr = image[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        pass

    # Fallback to PIL if available
    try:
        from PIL import Image  # type: ignore

        im = Image.fromarray(image)
        bio = io.BytesIO()
        im.save(bio, format="JPEG", quality=int(quality))
        return base64.b64encode(bio.getvalue()).decode("ascii")
    except Exception:
        return None


def _serialize_obs(obs: Any, *, include_images: bool = True) -> dict[str, Any]:
    # Expected obs schema in this repo:
    # {"states": {...}, "images": {...}, "task": str}
    if not isinstance(obs, dict):
        return {"raw": str(type(obs))}

    out: dict[str, Any] = {}
    if "task" in obs:
        try:
            out["task"] = str(obs["task"])
        except Exception:
            out["task"] = ""

    states = obs.get("states", {})
    if isinstance(states, dict):
        states_out: dict[str, Any] = {}
        for k, v in states.items():
            if isinstance(v, np.ndarray):
                states_out[k] = v.astype(np.float32).tolist()
            else:
                states_out[k] = v
        out["states"] = states_out

    images = obs.get("images", {})
    if isinstance(images, dict):
        images_out: dict[str, Any] = {}
        images_meta: dict[str, Any] = {}
        for k, v in images.items():
            if isinstance(v, np.ndarray):
                images_meta[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                if include_images:
                    b64 = _try_encode_image_to_jpeg_b64(v)
                    if b64 is not None:
                        images_out[k] = {"jpeg_b64": b64}
            else:
                images_meta[k] = {"type": str(type(v))}
        out["images"] = images_out
        out["images_meta"] = images_meta

    return out


PANEL_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Env Monitor</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
    .row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    button { padding: 8px 12px; margin-right: 8px; }
    pre { white-space: pre-wrap; word-break: break-word; max-width: 680px; }
    img { max-width: 320px; height: auto; border: 1px solid #eee; border-radius: 6px; }
    .status { font-size: 14px; color: #333; }
  </style>
</head>
<body>
  <div class=\"card\">
    <button id=\"btnStop\">Stop</button>
    <button id=\"btnReset\">Reset</button>
    <span class=\"status\" id=\"status\"></span>
  </div>

  <div class=\"row\">
    <div class=\"card\">
      <div><b>Obs (states/task)</b></div>
      <pre id=\"obs\">(waiting)</pre>
    </div>

    <div class=\"card\">
      <div><b>Images</b> <span class=\"status\">(if available)</span></div>
      <div id=\"imgs\"></div>
    </div>
  </div>

  <script>
    async function postCommand(cmd) {
      const res = await fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cmd),
      });
      return await res.json();
    }

    document.getElementById('btnStop').onclick = async () => {
      const ok = window.confirm('Stop now? Click OK for success, Cancel for failure.');
      await postCommand({ cmd: 'stop', success: ok });
    };

    document.getElementById('btnReset').onclick = async () => {
      await postCommand({ cmd: 'reset' });
    };

    function renderImages(images) {
      const root = document.getElementById('imgs');
      root.innerHTML = '';
      if (!images) return;
      for (const [k, v] of Object.entries(images)) {
        if (!v || !v.jpeg_b64) continue;
        const div = document.createElement('div');
        div.style.marginBottom = '12px';
        const title = document.createElement('div');
        title.textContent = k;
        const img = document.createElement('img');
        img.src = 'data:image/jpeg;base64,' + v.jpeg_b64;
        div.appendChild(title);
        div.appendChild(img);
        root.appendChild(div);
      }
    }

    async function poll() {
      try {
        const res = await fetch('/api/obs');
        const data = await res.json();
        const status = data.status || {};
        document.getElementById('status').textContent =
          `connected | waiting_reset=${!!status.waiting_reset} | stop_requested=${!!status.stop_requested} | stopped=${!!status.stopped}`;

        const payload = data.payload || {};
        const obs = payload.obs || null;
        if (obs) {
          const compact = { task: obs.task, states: obs.states, images_meta: obs.images_meta };
          document.getElementById('obs').textContent = JSON.stringify(compact, null, 2);
          renderImages(obs.images);
        }
      } catch (e) {
        document.getElementById('status').textContent = 'disconnected';
      }
      setTimeout(poll, 200);
    }
    poll();
  </script>
</body>
</html>
"""


@dataclasses.dataclass
class MonitorServerState:
    latest_payload: dict[str, Any] | None = None
    waiting_reset: bool = True
    stop_requested: bool = False
    stopped: bool = False
    stop_success: bool | None = None


class _MonitorHandler(BaseHTTPRequestHandler):
    # Set by factory
    _state: MonitorServerState
    _state_lock: threading.Lock
    _reset_event: threading.Event

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Silence default HTTP logs
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = PANEL_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/obs":
            with self._state_lock:
                payload = self._state.latest_payload
                status = {
                    "waiting_reset": self._state.waiting_reset,
                    "stop_requested": self._state.stop_requested,
                    "stopped": self._state.stopped,
                }
            _json_response(self, HTTPStatus.OK, {"payload": payload, "status": status})
            return

        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/command":
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        data = _read_json(self) or {}
        cmd = data.get("cmd")
        if cmd == "stop":
            success = data.get("success")
            if isinstance(success, bool) or success is None:
                with self._state_lock:
                    self._state.stop_requested = True
                    self._state.stop_success = success
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return
        if cmd == "reset":
            with self._state_lock:
                self._state.waiting_reset = False
            self._reset_event.set()
            _json_response(self, HTTPStatus.OK, {"ok": True})
            return

        _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "bad_command"})


class MonitorHTTPServer:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8000):
        self._state = MonitorServerState()
        self._state_lock = threading.Lock()
        self._reset_event = threading.Event()

        handler_cls = self._make_handler_class()
        self._httpd = ThreadingHTTPServer((host, port), handler_cls)
        self.host, self.port = self._httpd.server_address[0], self._httpd.server_address[1]

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def _make_handler_class(self):
        state = self._state
        state_lock = self._state_lock
        reset_event = self._reset_event

        class Handler(_MonitorHandler):
            _state = state
            _state_lock = state_lock
            _reset_event = reset_event

        return Handler

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def update_payload(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            self._state.latest_payload = _sanitize_for_json(payload)

    def consume_stop_request(self) -> bool | None:
        with self._state_lock:
            if not self._state.stop_requested:
                return None
            self._state.stop_requested = False
            self._state.stopped = True
            return self._state.stop_success

    def is_stopped(self) -> bool:
        with self._state_lock:
            return self._state.stopped

    def set_waiting_reset(self, waiting: bool) -> None:
        with self._state_lock:
            self._state.waiting_reset = waiting

    def clear_stopped(self) -> None:
        with self._state_lock:
            self._state.stopped = False
            self._state.stop_success = None
            self._state.stop_requested = False

    def wait_for_reset(self, *, poll_s: float = 0.05) -> None:
        # Use Event wait to avoid busy looping
        while True:
            if self._reset_event.wait(timeout=poll_s):
                self._reset_event.clear()
                return


class MonitorWrapper(gym.Wrapper):
    """A wrapper that exposes a tiny web monitoring panel.

    - On every step/reset: publishes the latest observation to the panel.
    - Panel STOP: next `step()` returns terminated=True and info includes success.
    - `reset()` blocks until panel RESET is pressed.
    - If the underlying env has `teleop_step()`, it will be called while waiting reset.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        include_images: bool = True,
        teleop_preview_interval_s: float = 0.2,
    ):
        super().__init__(env)
        self._include_images = include_images
        self._teleop_preview_interval_s = float(teleop_preview_interval_s)

        self._server = MonitorHTTPServer(host=host, port=port)
        print(f"[MonitorWrapper] Panel: http://{self._server.host}:{self._server.port}")

        self._last_obs: Any | None = None
        self._episode_index = 0
        self._step_index = 0
        self._last_stop_success: bool | None = None

        # Start in waiting reset state to match the spec.
        self._server.set_waiting_reset(True)

    def _publish(self, *, phase: str, obs: Any, info: dict[str, Any] | None = None) -> None:
        payload = {
            "phase": phase,
            "t": time.time(),
            "episode": self._episode_index,
            "step": self._step_index,
            "obs": _serialize_obs(obs, include_images=self._include_images),
            "info": info or {},
        }
        self._server.update_payload(payload)

    def _maybe_preview_teleop_obs(self) -> Any | None:
        base = getattr(self.env, "unwrapped", self.env)
        # Best-effort: LeRobotEnv exposes these helpers.
        try:
            robot = getattr(base, "robot")
            obs = robot.get_observation()
            proc = getattr(base, "robot_observation_processor")
            prepared = getattr(base, "prepare_observation")
            return prepared(proc(obs))
        except Exception:
            return None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # After any termination (including STOP), we block here until the panel triggers reset.
        self._server.set_waiting_reset(True)

        last_preview_t = 0.0
        while True:
            # While waiting, keep teleop running if available.
            base = getattr(self.env, "unwrapped", self.env)
            teleop_step = getattr(base, "teleop_step", None)
            if callable(teleop_step):
                teleop_step()
                now = time.time()
                if now - last_preview_t >= self._teleop_preview_interval_s:
                    preview = self._maybe_preview_teleop_obs()
                    if preview is not None:
                        self._publish(phase="waiting_reset", obs=preview, info={})
                    last_preview_t = now
            else:
                time.sleep(0.02)

            # Check reset event
            if self._server._reset_event.is_set():
                break

        self._server.wait_for_reset(poll_s=0.05)
        self._server.set_waiting_reset(False)
        self._server.clear_stopped()

        self._episode_index += 1
        self._step_index = 0
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._publish(phase="reset", obs=obs, info=info)
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # If panel requested STOP, we immediately terminate without sending actions to robot.
        stop_success = self._server.consume_stop_request()
        if stop_success is not None:
            self._last_stop_success = stop_success

        if self._server.is_stopped():
            obs = self._last_obs
            if obs is None:
                # Defensive: typical users call reset() first.
                obs, _info0 = self.env.reset()
                self._last_obs = obs
            info = {
                "user_stop": True,
                "success": self._last_stop_success,
            }
            _, _, _, _, info_env = self.env.step(None)
            for k, v in info_env.items():
                if k not in info:
                    info[k] = v
            self._publish(phase="stopped", obs=self._last_obs, info=info)
            return self._last_obs, 0.0, True, False, info

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_index += 1
        self._last_obs = obs
        self._publish(phase="step", obs=obs, info=info)
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._server.shutdown()
        finally:
            return super().close()
