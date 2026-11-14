"""Foreground CLI management, using the active Hermes profile and public APIs."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

# Execute with the Hermes environment's Python. No shell or global Rapid-MLX needed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "status", "pull", "use", "run", "forget"])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    from hermes_cli.config import load_config_readonly
    from hermes_cli.model_setup import persist_model_selection
    if args.port is not None:
        runtime.configure_port(args.port)
    if args.action == "setup":
        runtime.install()
    elif args.action == "status":
        print(json.dumps(runtime.status(load_config_readonly()), indent=2))
    elif args.action == "pull":
        print(runtime.download(args.model))
    elif args.action == "forget":
        runtime.delete_model(args.model, load_config_readonly())
    elif args.action in {"use", "run"}:
        runtime.set_enabled(True)
        if args.action == "use":
            runtime.validate_model(args.model)
            persist_model_selection(args.model, "rapid-mlx", base_url="", api_mode="chat_completions")
            print("Validated and saved the profile default. Existing chats keep their selection.")
        else:
            runtime.start(args.model)
            print(f"Managed Rapid-MLX ready at {runtime.base_url()} (Ctrl-C stops its verified PID)", flush=True)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            try:
                while runtime.owned_process() is not None:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                runtime.stop()


if __name__ == "__main__":
    main()
