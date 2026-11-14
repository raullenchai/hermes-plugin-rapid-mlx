"""Profile, subprocess, cache, and installation contracts; no operator model writes."""
from concurrent.futures import CancelledError
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch
from .conftest import IsolatedHome, load


class RuntimeTests(IsolatedHome):
    def setUp(self):
        super().setUp()
        self.r = load("rapid_runtime_checks", "runtime.py")

    def test_discovery_is_readonly_and_scoped(self):
        r = self.r
        first = r.base_url()
        self.assertEqual(r.local_models(), [])
        self.assertFalse(r.enabled())
        self.assertEqual(r.read_state(), {})
        self.assertFalse(self.home.exists())
        with patch.dict(os.environ, {"HERMES_HOME": str(self.home / "other")}):
            self.assertNotEqual(first, r.base_url())

    def test_locks_and_mid_operation_profile_switch(self):
        r = self.r
        with r.operation_lock():
            with self.assertRaisesRegex(RuntimeError, "operation is in progress"):
                with r.operation_lock():
                    pass
            with patch.dict(os.environ, {"HERMES_HOME": str(self.home / "other")}):
                with self.assertRaisesRegex(RuntimeError, "profile changed"):
                    r.api_key(create=True)

    def test_port_guards(self):
        r = self.r
        for port in [8477, 80, 65536, True]:
            with self.assertRaises(ValueError):
                r.configure_port(port)
        r.configure_port(8482)
        self.assertEqual(r.base_url(), "http://127.0.0.1:8482/v1")
        r._write_json(r.root() / "server.json", {"pid": 123})
        with self.assertRaises(RuntimeError):
            r.configure_port(8483)

    def test_symlinked_runtime_refused(self):
        r = self.r
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        r.root().mkdir(parents=True)
        (r.root() / "runtime").symlink_to(outside)
        with self.assertRaises(RuntimeError):
            r.status({})

    def test_child_environment_and_shared_cache(self):
        r = self.r
        with patch.dict(os.environ, {"HF_TOKEN": "secret", "OPENAI_API_KEY": "secret", "HTTP_PROXY": "proxy"}):
            env = r.child_env()
        for name in ["HF_TOKEN", "OPENAI_API_KEY", "HTTP_PROXY"]:
            self.assertNotIn(name, env)
        self.assertEqual(env["RAPID_MLX_TELEMETRY"], "0")
        self.assertEqual(env["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertFalse(Path(env.get("HF_HOME", env.get("HF_HUB_CACHE"))).is_relative_to(r.root()))
        self.assertFalse(self.home.exists())

    def test_forget_removes_record_only_with_active_model_guard(self):
        r = self.r
        r.models_dir().mkdir(parents=True)
        record = r.model_record("owner/model")
        record.write_text(json.dumps({"id": "owner/model"}))
        shared = Path(self.temporary.name) / "shared-weight"
        shared.write_text("keep")
        self.assertEqual(r.local_models(), ["owner/model"])
        with patch.object(r, "require_supported"):
            for bad in ["../bad", "alias", "owner/../../bad"]:
                with self.assertRaises(ValueError):
                    r.delete_model(bad, {})
            with self.assertRaises(ValueError):
                r.delete_model("owner/model", {"model": {"provider": "rapid-mlx", "default": "owner/model"}})
            r.delete_model("owner/model", {})
        self.assertEqual(shared.read_text(), "keep")
        self.assertFalse(record.exists())

    def test_process_identity_uses_birth_private_exe_and_exact_cli(self):
        r = self.r
        child = SimpleNamespace(create_time=lambda: 30., is_running=lambda: True, status=lambda: "running",
                                exe=lambda: str(r.cli().with_name("python3.12")),
                                cmdline=lambda: [str(r.cli().with_name("python3.12")), str(r.cli()), "serve"])
        r.cli().parent.mkdir(parents=True)
        r._write_json(r.cli().parents[1] / "hermes-install.json", {"interpreter_executable": child.exe()})
        state = {"pid": 11, "create_time": 30., "owner_pid": 10, "owner_create_time": 20.,
                 "executable": child.exe(), "base_url": r.base_url()}
        with patch.object(r.psutil, "Process", return_value=child):
            self.assertIs(r._process(state), child)
            for invalid in [{"create_time": 31}, {"owner_pid": None}, {"executable": "/bin/sh"}]:
                self.assertIsNone(r._process({**state, **invalid}))
            child.cmdline = lambda: [child.exe(), "/tmp/other.py", "serve"]
            self.assertIsNone(r._process(state))

    def test_silent_cancel_reaps_real_child_without_publishing(self):
        r = self.r
        r.cli().parent.mkdir(parents=True)
        r.cli().touch()
        entered, cancel = Event(), Event()
        real_popen = subprocess.Popen
        children, errors = [], []
        def spawn(*args, **kwargs):
            child = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            children.append(child)
            entered.set()
            return child
        def run():
            try:
                r.download("owner/model", cancel_event=cancel)
            except BaseException as exc:
                errors.append(exc)
        with patch.object(r, "require_supported"), patch.object(r.subprocess, "Popen", side_effect=spawn):
            worker = Thread(target=run)
            worker.start()
            self.assertTrue(entered.wait(3))
            cancel.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], CancelledError)
        self.assertIsNotNone(children[0].poll())
        self.assertEqual(r.local_models(), [])

    def test_alias_pull_publishes_only_after_cancellation_cutoff(self):
        r = self.r
        r.cli().parent.mkdir(parents=True)
        r.cli().touch()
        real_popen = subprocess.Popen
        def spawn(*args, **kwargs):
            return real_popen([sys.executable, "-c", 'print(\'{"id":"owner/model","size_bytes":20}\')'], **kwargs)
        def refuse():
            raise CancelledError()
        with patch.object(r, "require_supported"), patch.object(r.subprocess, "Popen", side_effect=spawn):
            with self.assertRaises(CancelledError):
                r.download("alias", before_publish=refuse)
            self.assertEqual(r.local_models(), [])
            self.assertEqual(r.download("alias"), "owner/model")
        self.assertEqual(r.local_models(), ["owner/model"])

    def test_use_checks_advertisement_and_generation(self):
        r = self.r
        with patch.object(r, "local_models", return_value=["owner/model"]), patch.object(r, "read_state", return_value={}), patch.object(r, "start") as start, patch.object(r, "request") as request:
            request.side_effect = [{"data": [{"id": "owner/model"}]}, {"choices": []}]
            with self.assertRaisesRegex(RuntimeError, "no text"):
                r.validate_model("owner/model")
            start.assert_called_once_with("owner/model")
            request.side_effect = [{"data": [{"id": "owner/model"}]}, {"choices": [{"message": {"content": "OK"}}]}]
            r.validate_model("owner/model")

    def test_installer_hashes_private_wheels_and_idempotent_receipt(self):
        r = self.r
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            if "venv" in args:
                r.cli().parent.mkdir(parents=True)
                r.cli().touch()
            return SimpleNamespace(returncode=0)
        with patch.object(r, "require_supported"), patch.object(r.shutil, "which", return_value=sys.executable), patch.object(r.subprocess, "run", side_effect=run), patch.object(r, "_record_interpreter", side_effect=lambda path, digest: r._write_json(path, {"version": r.VERSION, "requirements_sha256": digest, "interpreter_executable": sys.executable})):
            r.install()
            self.assertIn("--copies", calls[0])
            self.assertIn("--require-hashes", calls[1])
            self.assertIn("--only-binary=:all:", calls[1])
            self.assertIn("--isolated", calls[1])
            self.assertEqual(r._read_json(r.cli().parents[1] / "hermes-install.json")["version"], "1.0.2")
            r.install()
            self.assertEqual(len(calls), 4)

    def test_foreign_listener_never_adopted_or_spawned(self):
        r = self.r
        r.cli().parent.mkdir(parents=True)
        r.cli().touch()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            with patch.object(r, "require_supported"), patch.object(r, "enabled", return_value=True), patch.object(r, "installed", return_value=True), patch.object(r, "local_models", return_value=["owner/model"]), patch.object(r, "base_url", return_value=f"http://127.0.0.1:{port}/v1"), patch.object(r.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "unowned server"):
                    r.start("owner/model")
                spawn.assert_not_called()

    def test_peer_cannot_stop_a_live_owners_verified_process(self):
        r = self.r
        owner = SimpleNamespace(pid=10, create_time=lambda: 20.)
        peer = SimpleNamespace(pid=12, create_time=lambda: 25., is_running=lambda: True)
        state = {"owner_pid": 12, "owner_create_time": 25.}
        with patch.object(r, "require_supported"), patch.object(r, "read_state", return_value=state), patch.object(r, "_process", return_value=object()), patch.object(r.psutil, "Process", side_effect=lambda pid=None: owner if pid is None else peer), patch.object(r, "_stop_process") as stop:
            with self.assertRaisesRegex(RuntimeError, "Another live Hermes process"):
                r.stop()
            stop.assert_not_called()

    def test_incomplete_runtime_is_not_advertised_as_installed(self):
        r = self.r
        r.cli().parent.mkdir(parents=True)
        r.cli().touch()
        self.assertFalse(r.installed())
        with patch.object(r, "require_supported"), patch.object(r, "enabled", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Install the managed"):
                r.start("owner/model")

    def test_interpreter_receipt_observes_real_executable(self):
        r = self.r
        r.cli().parent.mkdir(parents=True)
        r.cli().with_name("python3.12").symlink_to(sys.executable)
        receipt = r.cli().parents[1] / "hermes-install.json"
        r._record_interpreter(receipt, "test-lock-digest")
        data = r._read_json(receipt)
        self.assertEqual(data["requirements_sha256"], "test-lock-digest")
        self.assertEqual(Path(data["interpreter_executable"]), Path(r.psutil.Process().exe()).resolve())
