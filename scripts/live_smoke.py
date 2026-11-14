"""Opt-in native test. Run with Hermes's Python in an ISOLATED HERMES_HOME.

Uses port 8480 by default; does not inspect or contact port 8477.
"""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
from fastapi import FastAPI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-checkout", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--port", type=int, default=8480, choices=range(8480, 8490))
    args = parser.parse_args()
    sys.path.insert(0, str(args.hermes_checkout.resolve()))
    from providers import get_provider_profile
    profile = get_provider_profile("rapid-mlx")
    assert profile is not None, "Install and enable the plugin in the isolated Hermes home first"
    module = sys.modules[type(profile).__module__]
    runtime = module.runtime
    print("Loaded provider:", profile.name, flush=True)
    runtime.configure_port(args.port)
    path = Path(__file__).resolve().parents[1] / "dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("rapid_live_api", path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    app = FastAPI()
    app.include_router(api.create_router(runtime))
    def request(method, path, body=None):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.request(method, path, json=body)
                response.raise_for_status()
                return response.json()
        return asyncio.run(run())
    def job(path, body=None):
        record = request("POST", path, body)
        deadline = time.monotonic() + 1800
        while record["status"] == "running" and time.monotonic() < deadline:
            time.sleep(.25)
            record = request("GET", "/jobs/" + record["id"])
        assert record["status"] == "complete", record
        print(path, "complete", record.get("result"), flush=True)
        return record.get("result")
    pid = None
    try:
        job("/install")
        model = job("/download", {"repo": args.model})
        assert model in runtime.local_models()
        print("Library:", runtime.local_models(), flush=True)
        runtime.set_enabled(True)
        runtime.start(model)
        pid = runtime.read_state()["pid"]
        print("Started verified PID", pid, "at", runtime.base_url(), flush=True)
        print("/v1/models:", json.dumps(runtime.request("/models")["data"]), flush=True)
        # Auth check does not disclose the generated key.
        with httpx.Client(trust_env=False) as client:
            response = client.get(runtime.base_url() + "/models")
            assert response.status_code in {401, 403}, response.status_code
            print("Unauthenticated /v1/models refused:", response.status_code, flush=True)
        job("/activate", {"model_id": model})
        from hermes_cli.config import load_config_readonly
        selection = load_config_readonly()["model"]
        assert selection["provider"] == "rapid-mlx" and selection["default"] == model
        print("Use persisted profile selection:", selection["provider"], selection["default"], flush=True)
        client = profile.create_client(max_retries=0)
        try:
            reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say hello in five words."}], max_tokens=96)
            print("Guarded provider response:", reply.choices[0].message.content or reply.choices[0].message.model_extra, flush=True)
        finally:
            client.close()
        command = [sys.executable, str(args.hermes_checkout / "hermes"), "chat", "--provider", "rapid-mlx", "-m", model,
                   "--toolsets", "", "--max-turns", "1", "-q", "Reply with a short greeting. Do not use tools."]
        print("Hermes CLI command: hermes chat --provider rapid-mlx -m", model, "--toolsets '' --max-turns 1 -q <greeting>", flush=True)
        child = subprocess.run(command, cwd=args.hermes_checkout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
        print(child.stdout, flush=True)
        assert child.returncode == 0, f"Hermes chat exited {child.returncode}"
        request("POST", "/stop")
        print("Turn off completed; restarting immediately", flush=True)
        job("/start")
        pid = runtime.read_state()["pid"]
        print("Restarted verified PID:", pid, flush=True)
    finally:
        if runtime.owned_process() is not None:
            runtime.set_enabled(False)
            runtime.stop()
        print("Stopped managed PID:", pid, "state:", runtime.read_state(), flush=True)
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", args.port)) != 0, "Test port still listening"
            print("Test port is closed", flush=True)


if __name__ == "__main__":
    main()
