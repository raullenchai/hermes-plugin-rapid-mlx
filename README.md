# Rapid-MLX for Hermes

A managed [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) local runtime, model library, and named `rapid-mlx` inference provider for Hermes Desktop and CLI. MIT licensed; Rapid-MLX is Apache-2.0.

Requires **Hermes 0.23.0+**, **macOS 15+ on Apple Silicon**, and an existing **Python 3.12** (`python3.12` on PATH) on the inference host. The tested host is macOS 26.5.2 arm64. The provider needs in-process execution because it returns a live OpenAI SDK client; strict host-process isolation is unsupported.

## Install

After this repository is published:

```sh
hermes plugins install https://github.com/raullenchai/hermes-plugin-rapid-mlx
hermes plugins enable rapid-mlx
```

After catalog acceptance, `hermes plugins install rapid-mlx` also works. For local development, copy this tree to the active `$HERMES_HOME/plugins/rapid-mlx/`, then enable it with Hermes. Do not copy `.git`, caches, or virtual environments.

Enable the Desktop half in **Capabilities → Plugins**, then open **Rapid-MLX** in plugin settings. Both the Python plugin and Desktop half must be enabled in the intended host/profile. If a running gateway has not loaded the backend, restart it from an external terminal after ongoing work finishes.

1. Choose **Install**. This creates the private, hash-locked Rapid-MLX 1.0.2 environment; it does not install a global runtime or modify Hermes's Python packages.
2. Enter a Rapid-MLX alias such as `qwen3.5-4b-4bit`, or an exact public MLX chat repository such as `mlx-community/Qwen3-0.6B-4bit`, then choose **Download**. This invokes `rapid-mlx pull` and records the resolved repository only after success.
3. Choose **Use**. This starts the authenticated loopback runtime, checks `/v1/models`, performs a real short generation, and only then saves the profile default through Hermes's model-selection API.
4. **Turn off** stops the verified owned PID and disables automatic restart. **Turn on** restarts the last served model. After explicit enablement, a provider request may restart the runtime; opening settings, resolving credentials, listing models, and constructing clients never start it.

**Use** changes the profile default, not an existing chat's selection. Use the ordinary chat model picker to select `rapid-mlx` and the repository ID for an existing chat. This version serves one model at a time; switching models requires stopping the old managed server. It refuses to stop another live Hermes process's server: use that owner's settings window or terminal.

## CLI

Run `manage.py` with the **Hermes environment's Python**, with Hermes importable and the intended `HERMES_HOME` selected. For a source checkout, the documented `scripts/run-in-hermes-env` wrapper supplies that environment:

```sh
# From a Hermes checkout; set plugin to this repository's absolute path.
plugin=/absolute/path/to/hermes-plugin-rapid-mlx
scripts/run-in-hermes-env python "$plugin/manage.py" setup
scripts/run-in-hermes-env python "$plugin/manage.py" pull mlx-community/Qwen3-0.6B-4bit
scripts/run-in-hermes-env python "$plugin/manage.py" use mlx-community/Qwen3-0.6B-4bit
scripts/run-in-hermes-env python "$plugin/manage.py" run mlx-community/Qwen3-0.6B-4bit
# In another terminal using the same Hermes profile:
hermes chat --provider rapid-mlx -m mlx-community/Qwen3-0.6B-4bit -q 'Hello'
```

`run` stays in the foreground; Ctrl-C or SIGTERM shuts down its verified server. `use` validates and saves the default, then cleans up the server on exit. Hermes chat can also start it after `use` has enabled the runtime. `status` is read-only; `forget owner/model` removes a library record under the same guards as Desktop. `setup --port 8480` selects an explicit port while stopped. Otherwise the port is deterministic per profile. Port 8477 is reserved and refused by this plugin.

## Storage and downloads

Runtime, receipts, library records and logs live under the active profile's durable `plugin-data/rapid-mlx/`. A private child HOME keeps Rapid-MLX preferences and prefix caches there. Updates do not replace this data. The plugin never adopts an existing Rapid-MLX installation or external listener.

Model files use the **standard shared Hugging Face cache**, normally `~/.cache/huggingface/hub`, honoring the host's `HF_HOME`, `HF_HUB_CACHE`, and `HF_XET_CACHE`. There is no private model cache or duplicate snapshot copy. Pulling an already-cached model reuses it. **Remove** forgets only the library record; it never deletes shared model files. The active default and models while a server record exists cannot be removed. Use the Hugging Face cache tools separately if you intentionally want to reclaim shared storage.

Pull jobs are polled, cancellable, and bounded to four hours. Cancellation signals the owned process group and waits for the pull worker to exit before settling; upstream partial cache files remain for retry. A cancellation cutoff protects library-record publication. Progress is deliberately **indeterminate**: this version exposes the native `pull.log`, completion and cancellation state, without claiming byte counts, transfer speed, or ETA. Installation is bounded to 30 minutes plus interpreter preparation and checks. See `install.log`, `pull.log`, and `server.log` in plugin data for native diagnostics.

## Operational and security disclosure

- **Downloads:** explicit setup uses the existing Python 3.12 stdlib `venv --copies`, then its `python -I -m pip --isolated install --require-hashes --only-binary=:all: --no-cache-dir` against `https://pypi.org/simple`. It fetches Rapid-MLX **1.0.2** and every exact dependency in [runtime-requirements.txt](runtime-requirements.txt) from PyPI/`files.pythonhosted.org`; all allowed artifacts have committed SHA-256 hashes. The base upstream distribution includes native MLX/Metal and vision/audio-related dependencies, even though this plugin exposes chat only. It does not fetch Python, Homebrew, uv, a shell installer, or optional extras. No wheels or native executables are committed here.
- **Model network:** `rapid-mlx pull <alias-or-repo>` uses Rapid-MLX's official `https://models.rapidmlx.com` model mirror with Hugging Face/CDN fallback and the standard Hub cache. Alias-declared companion artifacts may also be pulled. Models have their own licenses. No external account credentials are required; private/gated models are unsupported. The child's implicit HF token use and telemetry are disabled. Already-cached files can originate from another local tool.
- **Processes:** setup invokes existing Python and private pip; pulls invoke the private `rapid-mlx pull`; inference runs the private `rapid-mlx serve <repo> --host 127.0.0.1 --port N --api-key SECRET`. Arguments are lists, never shell interpolation. The server is long-running while its Hermes owner is alive; model loading consumes unified memory and pulls consume disk/bandwidth. Native libraries execute with the user's OS privileges.
- **Ownership:** before requests or lifecycle actions, verifies PID birth time, the interpreter observed during setup, exact private CLI path and `serve` subcommand, profile, and origin. macOS framework Python may execute its installed framework binary; it is recorded in the installation receipt. A busy port or unverifiable ownership fails closed. Another live Hermes owner's server can serve same-profile requests but cannot be stopped by a peer. Normal owner exit cleans up its runtime.
- **Environment:** children receive a small OS allowlist, the standard cache locations, and privacy switches; no inherited API keys, HF tokens, proxies, or approval-bypass variables. Child HOME is private. Neither credentials nor environment files are read directly by this plugin.
- **Local credential:** a random bearer key is created through Hermes's secure credential writer and read with profile-scoped `get_secret`. Hermes stores it as `HERMES_RAPID_MLX_LOCAL_API_KEY` in that profile's `.env`. It is never returned by REST or written to plugin data. The masked `local_api_key` setting is optional; changing it requires restarting the server. Like the upstream `--api-key` interface, the child process's argument list contains the local key.
- **Inference:** server startup uses cached models only (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, mirror disabled); if files were removed externally, pull again explicitly. The client permits only authenticated `http://127.0.0.1:<owned-port>/v1/…` traffic is permitted. Proxies and redirects are disabled; SDK copies retain the guard. Startup errors happen at request time, outside Hermes's provider-construction fallback. Local key material and prompts are not sent to arbitrary listeners.
- **Hermes changes:** only documented provider registration, profile storage/config/secret/model-selection APIs, the authenticated plugin backend mount, and `@hermes/plugin-sdk`. **Use** rewrites the profile's default model in `config.yaml`. No core patches, internal renderer imports, auto-approvals, self-updating plugin code, or telemetry.

This is trusted local plugin code, not a sandbox. Interrupted installs retain a visibly incomplete environment for inspection; stop the runtime and remove only its incomplete `plugin-data/rapid-mlx/runtime/1.0.2` directory before retrying. The installer never deletes an existing environment automatically.

## Development and verification

Use an isolated Hermes home. From the Hermes checkout:

```sh
plugin=/absolute/path/to/hermes-plugin-rapid-mlx
scripts/run_tests.sh "$plugin/tests" -q -c "$plugin/pytest.ini" --confcutdir="$plugin/tests"
hermes plugins validate "$plugin" --install-deps
node --check "$plugin/desktop/plugin.js"
# Opt-in real native install/pull/start/Use/chat/stop; isolated profile only:
scripts/run-in-hermes-env python "$plugin/scripts/live_smoke.py" --hermes-checkout "$PWD" --port 8480
```

The native smoke script requires the plugin already copied and enabled in that isolated profile. It captures actual registered-provider and Hermes CLI responses and stops the test PID in `finally`. See [verification](docs/verification.md) and [CLI transcript](docs/cli-transcript.txt) for executed checks and limitations. No benchmark or untested audio/embedding support is claimed.

## Provenance

Derived from the MIT-licensed [oMLX community plugin](https://github.com/vitaliy-blprnt/hermes-plugin-omlx/tree/eb748caf0a3080cd3b8d9426eb995d2722ec3406), cataloged in [Hermes PR #150815](https://github.com/NousResearch/hermes-agent/pull/150815). Its provider transport, profile isolation, background-job API, Desktop SDK settings structure and associated tests are retained/adapted with original copyright notices. Rapid-MLX uses a different runtime and architecture: a hash-locked PyPI venv, native alias-aware `pull`, a shared Hub cache, and single-model lifecycle, rather than an oMLX DMG or private snapshot library. This plugin is maintained separately; it is not an oMLX rebrand or a Hermes fork.
