"""Public-only Rapid-MLX pull worker; runs solely in the private pinned environment."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    from rapid_mlx.model_aliases import resolve_hub_model, resolve_profile
    from huggingface_hub import snapshot_download

    executable, requested = sys.argv[1:]
    profile = resolve_profile(requested)
    if profile is not None and (getattr(profile, "modality", "text") or "text") != "text":
        raise ValueError("This plugin manages text chat models only")
    # The upstream CLI owns alias resolution, disk checks, and resumable Hub transfers.
    subprocess.run([executable, "pull", requested], stdout=sys.stderr, stderr=sys.stderr, check=True)
    repo = resolve_hub_model(requested)
    snapshot = Path(snapshot_download(repo, local_files_only=True, token=False))
    if not (snapshot / "config.json").is_file() or not any(snapshot.glob("*.safetensors")):
        raise ValueError("Only completed safetensors chat-model snapshots can enter this library")
    print(json.dumps({"id": repo, "requested": requested,
                      "size_bytes": sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
