"""Plugin-owned, profile-scoped Rapid-MLX runtime. Import and status never start it."""
from __future__ import annotations

import atexit
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import select
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

import psutil

VERSION = "1.0.2"
SECRET_NAME = "HERMES_RAPID_MLX_LOCAL_API_KEY"
REPO_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
_STATE_FIELDS = {"pid", "create_time", "executable", "owner_pid", "owner_create_time", "base_url", "model"}
_OPERATION_HOME: ContextVar[Path | None] = ContextVar("rapid-mlx_operation_home", default=None)


def home() -> Path:
    from hermes_constants import get_hermes_home
    active = Path(get_hermes_home()).resolve()
    expected = _OPERATION_HOME.get()
    if expected is not None and expected != active:
        raise RuntimeError("Active Hermes profile changed during the Rapid-MLX operation")
    return active


def root() -> Path:
    # plugin_data_dir creates directories; read paths must not call it.
    return home() / "plugin-data" / "rapid-mlx"


def _guard_paths() -> None:
    base = home()
    for path in (base / "plugin-data", root(), root() / "models", root() / "runtime", root() / "home",
                 root() / "runtime" / VERSION, cli().parents[2], cli().parents[1], cli().parent):
        if path.is_symlink() or not path.resolve().is_relative_to(base):
            raise RuntimeError("Refusing a symlinked Rapid-MLX storage path")


def models_dir() -> Path:
    return root() / "models"


def cli() -> Path:
    return root() / "runtime" / VERSION / "bin/rapid-mlx"


def base_url() -> str:
    port = _read_json(root() / "settings.json").get("port")
    if port is None:
        port = 20000 + int(hashlib.sha256(str(home()).encode()).hexdigest()[:8], 16) % 30000
    if type(port) is not int or not 1024 <= port <= 65535 or port == 8477:
        raise ValueError("Choose an unprivileged port other than reserved port 8477")
    return f"http://127.0.0.1:{port}/v1"


def configure_port(port: int) -> None:
    with operation_lock():
        if read_state():
            raise RuntimeError("Stop the runtime before changing ports")
        if type(port) is not int or not 1024 <= port <= 65535 or port == 8477:
            raise ValueError("Invalid or reserved port")
        _write_json(root() / "settings.json", {"port": port})


def supported() -> bool:
    try:
        return platform.system() == "Darwin" and platform.machine() == "arm64" and int(platform.mac_ver()[0].split(".")[0]) >= 15
    except (ValueError, IndexError):
        return False


def require_supported() -> None:
    if not supported():
        raise ValueError("Rapid-MLX requires macOS 15 or newer on Apple Silicon (arm64)")


def _read_json(path: Path) -> dict:
    _guard_paths()
    if path.is_symlink():
        raise RuntimeError("Refusing a symlinked Rapid-MLX state file")
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, value: dict) -> None:
    _guard_paths()
    if path.is_symlink():
        raise RuntimeError("Refusing a symlinked Rapid-MLX state file")
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def enabled() -> bool:
    return _read_json(root() / "preferences.json").get("enabled") is True


def set_enabled(value: bool) -> None:
    if type(value) is not bool:
        raise ValueError("enabled must be a boolean")
    if value:
        require_supported()
    with operation_lock():
        _write_json(root() / "preferences.json", {"enabled": value})


@contextmanager
def operation_lock(name: str = "operation"):
    """Cross-process, nonblocking mutation locks; no implicit runtime installation."""
    import fcntl
    if name not in {"operation", "download"}:
        raise ValueError("Invalid Rapid-MLX lock")
    _guard_paths()
    from plugins.plugin_storage import plugin_data_dir
    storage = plugin_data_dir("rapid-mlx")
    if Path(storage).resolve() != root():
        raise RuntimeError("Active Hermes profile changed during operation")
    lock = root() / f"{name}.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Rapid-MLX operation is in progress; try again shortly") from exc
        token = _OPERATION_HOME.set(home())
        try:
            yield
        finally:
            _OPERATION_HOME.reset(token)
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_state() -> dict:
    return {key: value for key, value in _read_json(root() / "server.json").items() if key in _STATE_FIELDS}


def _process(state: dict):
    if state.get("base_url") != base_url():
        return None
    try:
        if (type(state.get("pid")) is not int or type(state.get("owner_pid")) is not int
                or state["pid"] <= 0 or state["owner_pid"] <= 0
                or type(state.get("owner_create_time")) not in (int, float)):
            return None
        proc = psutil.Process(state["pid"])
        if (state["create_time"] != proc.create_time() or not proc.is_running()
                or proc.status() == psutil.STATUS_ZOMBIE):
            return None
        executable = Path(proc.exe()).resolve()
        receipt = _read_json(cli().parents[1] / "hermes-install.json")
        expected_python = Path(receipt["interpreter_executable"]).resolve()
        command = proc.cmdline()
        if (executable != expected_python or executable != Path(state["executable"]).resolve()
                or len(command) < 3 or Path(command[1]).absolute() != cli().absolute()
                or command[2] != "serve"):
            return None
        return proc
    except (KeyError, TypeError, OSError, ValueError, psutil.Error):
        return None


def owned_process():
    """A verified private child may serve other workers of the same profile."""
    _guard_paths()
    return _process(read_state())


def api_key(create: bool = False) -> str | None:
    home()  # Fail closed if a long-running operation lost its profile scope.
    from agent.secret_scope import get_secret
    key = get_secret(SECRET_NAME)
    if key:
        return key
    if not create:
        return None
    from hermes_cli.config import save_env_value_secure
    key = secrets.token_urlsafe(32)
    saved = save_env_value_secure(SECRET_NAME, key)
    if not saved.get("success"):
        raise RuntimeError("Could not save the managed Rapid-MLX credential")
    return key


def child_env() -> dict[str, str]:
    allowed = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "WINDIR", "USER", "LOGNAME", "SHELL"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    # Preserve the operator's ONE standard HF cache; never create a plugin cache.
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(HOME=str(root() / "home"),
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
               RAPID_MLX_TELEMETRY="0", DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false")
    # With a private HOME, explicitly retain the standard cache location when HF_HOME is unset.
    if "HF_HOME" not in env and "HF_HUB_CACHE" not in env:
        env["HF_HOME"] = str(Path.home() / ".cache/huggingface")

    return env


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Unexpected Rapid-MLX redirect")


def request(path: str, payload: dict | None = None, *, timeout: int = 5):
    if path not in {"/models", "/chat/completions"}:
        raise ValueError("Unsupported Rapid-MLX request path")
    if owned_process() is None:
        raise RuntimeError("Managed Rapid-MLX is not running under this process")
    key = api_key()
    if not key:
        raise RuntimeError("Managed Rapid-MLX credential is unavailable")
    headers = {"Authorization": "Bearer " + key}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(base_url() + path, data=data, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        if response.url != req.full_url:
            raise RuntimeError("Unexpected Rapid-MLX redirect")
        return json.load(response)


def is_configured(config: dict) -> bool:
    model = (config or {}).get("model") or {}
    return isinstance(model, dict) and model.get("provider") == "rapid-mlx"


def model_record(model_id: str) -> Path:
    if not isinstance(model_id, str) or not re.fullmatch(REPO_PATTERN, model_id):
        raise ValueError("Use a resolved Hugging Face owner/model ID")
    return models_dir() / (hashlib.sha256(model_id.encode()).hexdigest() + ".json")


def local_models() -> list[str]:
    _guard_paths()
    result = []
    for path in models_dir().glob("*.json"):
        record = _read_json(path)
        mid = record.get("id", "")
        if re.fullmatch(REPO_PATTERN, mid) and path == model_record(mid):
            result.append(mid)
    return sorted(set(result))


def delete_model(model_id: str, config: dict) -> None:
    """Forget only this profile's library record; shared HF files are never deleted."""
    require_supported()
    with operation_lock("download"), operation_lock():
        if model_id not in local_models():
            raise ValueError("Model is not in this plugin library")
        if is_configured(config) and (config.get("model") or {}).get("default") == model_id:
            raise ValueError("Select a different model before removing the active model")
        if read_state():
            raise RuntimeError("Turn off Rapid-MLX before removing a model")
        model_record(model_id).unlink()


def installed() -> bool:
    receipt = _read_json(cli().parents[1] / "hermes-install.json")
    return (cli().is_file() and bool(receipt.get("interpreter_executable"))
            and receipt.get("requirements_sha256") == hashlib.sha256(
                Path(__file__).with_name("runtime-requirements.txt").read_bytes()).hexdigest())


def status(config: dict) -> dict:
    _guard_paths()
    compatible = supported()
    active = (config.get("model") or {}).get("default") if is_configured(config) else None
    return {"supported": compatible, "installed": installed(),
            "running": owned_process() is not None, "enabled": enabled(),
            "model_dir": "Shared standard Hugging Face cache (never deleted by this plugin)",
            "base_url": base_url(), "version": VERSION if installed() else None,
            "models": [{"id": mid, "active": mid == active,
                        "size_bytes": _read_json(model_record(mid)).get("size_bytes")}
                       for mid in local_models()],
            "error": None if compatible else "Rapid-MLX requires macOS 15+ on Apple Silicon"}


def install(progress=None) -> None:
    """Wheels only, exact versions and SHA-256 hashes, no global environment writes."""
    require_supported()
    with operation_lock():
        destination = root() / "runtime" / VERSION
        receipt = destination / "hermes-install.json"
        lock = Path(__file__).with_name("runtime-requirements.txt")
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        if cli().is_file() and _read_json(receipt).get("requirements_sha256") == digest:
            if not _read_json(receipt).get("interpreter_executable"):
                _record_interpreter(receipt, digest)
            return
        if destination.exists():
            raise RuntimeError("Incomplete or different private runtime; inspect it before removing it and retrying")
        python = shutil.which("python3.12")
        if not python:
            raise RuntimeError("Install Python 3.12 first; setup does not download Python or modify global tools")
        destination.parent.mkdir(parents=True, exist_ok=True)
        (root() / "home").mkdir(exist_ok=True)
        log_path = root() / "install.log"
        fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as log:
            env = child_env()
            subprocess.run([python, "-I", "-m", "venv", "--copies", str(destination)],
                           env=env, check=True, stdout=log, stderr=log, timeout=120)
            subprocess.run([str(cli().with_name("python3.12")), "-I", "-m", "pip", "--isolated", "install",
                            "--index-url", "https://pypi.org/simple", "--no-cache-dir", "--require-hashes",
                            "--only-binary=:all:", "-r", str(lock)],
                           env=env, check=True, stdout=log, stderr=log, timeout=1800)
            subprocess.run([str(cli().with_name("python3.12")), "-I", "-m", "pip", "check"],
                           env=env, check=True, stdout=log, stderr=log, timeout=60)
            subprocess.run([str(cli()), "--version"], env=env, check=True,
                           stdout=log, stderr=log, timeout=90)
        _record_interpreter(receipt, digest)


def _record_interpreter(receipt: Path, digest: str) -> None:
    # macOS framework builds exec Python.app; observe the trusted fresh venv's
    # interpreter directly instead of trusting argv[0] or guessing that path.
    probe = subprocess.Popen([str(cli().with_name("python3.12")), "-I", "-c",
                              "import sys; print('ready', flush=True); sys.stdin.read()"],
                             env=child_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True)
    try:
        if not select.select([probe.stdout], [], [], 90)[0] or probe.stdout.readline().strip() != "ready":
            raise RuntimeError("Private interpreter probe failed")
        executable = psutil.Process(probe.pid).exe()
        _write_json(receipt, {"version": VERSION, "requirements_sha256": digest,
                              "interpreter_executable": str(Path(executable).resolve())})
    finally:
        try:
            probe.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            probe.kill()
            probe.communicate(timeout=10)


def _expected_interpreter() -> Path:
    receipt = _read_json(cli().parents[1] / "hermes-install.json")
    return Path(receipt["interpreter_executable"]).resolve()


def _stop_process(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except psutil.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def start(model_id: str | None = None) -> None:
    require_supported()
    if not enabled():
        raise RuntimeError("Explicitly enable managed Rapid-MLX before starting it")
    if not installed():
        raise RuntimeError("Install the managed Rapid-MLX runtime first")
    with operation_lock():
        state = read_state()
        if model_id is None:
            model_id = state.get("model") or _read_json(root() / "selected.json").get("model")
        if model_id not in local_models():
            raise ValueError("Pull a model and choose Use before starting the runtime")
        proc = _process(state)
        if proc is not None:
            if state.get("model") != model_id:
                raise RuntimeError("Turn off Rapid-MLX before using a different model")
            request("/models")
            return
        if state:
            try:
                recorded = psutil.Process(state["pid"])
                alive = recorded.is_running() and recorded.create_time() == state["create_time"]
            except (KeyError, TypeError, psutil.NoSuchProcess):
                alive = False
            if alive:
                raise RuntimeError("A different or unverifiable process owns the recorded Rapid-MLX server")
        port = urllib.parse.urlsplit(base_url()).port
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is occupied by an unowned server") from exc
        models_dir().mkdir(parents=True, exist_ok=True)
        key = api_key(create=True)
        cmd = [str(cli().with_name("python3.12")), str(cli()), "serve", model_id,
               "--host", "127.0.0.1", "--port", str(port), "--api-key", key]
        log_path = root() / "server.log"
        log_fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(log_fd, "ab") as log:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, env=child_env() | {"HF_HUB_OFFLINE": "1",
                                         "TRANSFORMERS_OFFLINE": "1", "RAPID_MLX_MODEL_MIRROR": ""})
        try:
            owner = psutil.Process()
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"Rapid-MLX exited ({child.returncode}); see {log_path}")
                proc = psutil.Process(child.pid)  # Recreate after the shell wrapper execs Python.
                if Path(proc.exe()).resolve() == _expected_interpreter():
                    state = {"pid": child.pid, "create_time": proc.create_time(), "executable": proc.exe(),
                             "owner_pid": owner.pid, "owner_create_time": owner.create_time(), "base_url": base_url(), "model": model_id}
                    _write_json(root() / "server.json", state)
                    try:
                        request("/models")
                        _write_json(root() / "selected.json", {"model": model_id})
                        context = copy_context()
                        atexit.register(context.run, shutdown_owned)
                        return
                    except (OSError, ValueError, RuntimeError):
                        pass
                time.sleep(0.25)
            raise RuntimeError(f"Rapid-MLX did not become ready; see {log_path}")
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            (root() / "server.json").unlink(missing_ok=True)
            raise


def stop() -> None:
    require_supported()
    with operation_lock():
        state = read_state()
        proc = _process(state)
        if proc is None:
            if state:
                raise RuntimeError("Rapid-MLX ownership could not be verified; refusing to stop it")
            return
        current = psutil.Process()
        if state["owner_pid"] != current.pid or state["owner_create_time"] != current.create_time():
            try:
                owner = psutil.Process(state["owner_pid"])
                alive = owner.is_running() and owner.create_time() == state["owner_create_time"]
            except psutil.NoSuchProcess:
                alive = False
            if alive:
                raise RuntimeError("Another live Hermes process owns Rapid-MLX; stop it from that process")
        _stop_process(proc)
        (root() / "server.json").unlink(missing_ok=True)


def shutdown_owned() -> None:
    try:
        if owned_process() is not None:
            stop()
    except (OSError, RuntimeError, ValueError, psutil.Error):
        pass


def validate_model(model_id: str) -> None:
    if model_id not in local_models():
        raise ValueError("Pull the model before choosing Use")
    current = read_state()
    if current and current.get("model") != model_id:
        stop()
    start(model_id)
    if model_id not in {m.get("id") for m in request("/models").get("data", [])}:
        raise RuntimeError("Server did not advertise the requested model")
    response = request("/chat/completions", {"model": model_id, "messages": [
        {"role": "user", "content": "Reply with OK."}], "max_tokens": 32, "stream": False}, timeout=300)
    if not any(c.get("message", {}).get("content") or c.get("message", {}).get("reasoning_content")
               for c in response.get("choices", [])):
        raise RuntimeError("Generation returned no text; default model was not changed")


def download(repo: str, *, progress=None, transfer_progress=None, cancel_event=None, before_publish=None) -> str:
    require_supported()
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?", repo):
        raise ValueError("Use a Rapid-MLX alias or a public owner/model ID")
    if not cli().is_file():
        raise RuntimeError("Install the managed runtime first")
    with operation_lock("download"):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Pull cancelled")
        models_dir().mkdir(parents=True, exist_ok=True)
        worker = Path(__file__).with_name("pull_worker.py")
        log_fd = os.open(root() / "pull.log", os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(log_fd, "wb") as log:
            args = [str(cli().with_name("python3.12")), str(worker), str(cli()), repo]
            with tempfile.TemporaryFile(dir=root()) as result:
                child = subprocess.Popen(args, env=child_env(), stdin=subprocess.DEVNULL,
                                         stdout=result, stderr=log, start_new_session=True)
                deadline = time.monotonic() + 14400
                try:
                    while child.poll() is None:
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError("Pull cancelled")
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Pull timed out")
                        time.sleep(0.1)
                    if child.returncode:
                        raise RuntimeError(f"Rapid-MLX pull failed ({child.returncode}); see pull.log")
                    result.seek(0)
                    record = json.load(result)
                finally:
                    if child.poll() is None:
                        # Worker execs the CLI in its own process; signal this owned group, never listeners.
                        os.killpg(child.pid, 15)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, 9)
                    child.wait(timeout=10)
        mid = record["id"]
        destination = model_record(mid)
        if before_publish is not None:
            before_publish()
        elif cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Pull cancelled")
        _write_json(destination, record)
        return mid
