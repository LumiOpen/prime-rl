# Running prime-rl on LUMI

How to get the `lumi/upstream-main` branch running on LUMI and verify it with the
reverse-text smoke test (a 0.6B model on 2 GPUs, ~15 min). Everything runs inside the
site container `/appl/local/laifs/containers/lumi-multitorch-latest.sif` — we do not
build images.

Project: `project_462001516`. Put your clone under
`/scratch/project_462001516/users/$USER/`.

## 1. Clone

```bash
cd /scratch/project_462001516/users/$USER
git clone --recurse-submodules -b lumi/upstream-main https://github.com/LumiOpen/prime-rl.git
cd prime-rl
```

## 2. Fix the paths

Three files hardcode the clone location. Replace `khakala` with your own username:

| File | What to change |
|---|---|
| `env.sh` line 2 | `BASE=` → your clone path |
| `run_reverse_text_test.sbatch` line 33 | `BASE=` → your clone path |
| `run_reverse_text_test.sbatch` lines 14–15 | `--output` / `--error` → your clone's `logs/` |

```bash
sed -i "s|users/khakala/lib/prime-rl-upstream-main|users/$USER/prime-rl|g" env.sh run_reverse_text_test.sbatch
```

**Do not change the wheel paths** in `pyproject.toml` (`vllm = ...`, `flash-attn = ...`).
They point at `/scratch/project_462001516/users/khakala/wheels/`, which is group-readable
by everyone in the project — so you get the prebuilt ROCm wheels and never have to run
`build_vllm_wheel.sbatch` / `build_flash_attn_wheel.sbatch` yourself. Those two builders
are only needed when the torch version moves; each wheel is valid for exactly one
(torch, python, GPU arch) triple.

## 3. Get the amdsmi bindings

`pyproject.toml` pins `amdsmi = { path = "amd_smi" }`, and that directory is **not in
git** — AMD publishes no wheel, and the bindings are ctypes wrappers that must match the
`libamd_smi.so` inside the container you actually run in. Copy them out of the image:

```bash
./scripts/setup_amdsmi.sh
```

Re-run it if the container image is updated. A stale copy fails at import with
`undefined symbol: amdsmi_get_gpu_enumeration_info`.

## 4. Create the venv

Inside the container, on a login node (this only resolves and installs, no GPU needed):

```bash
./start_shell.sh                    # drops you into the container
cd /scratch/project_462001516/users/$USER/prime-rl
source env.sh
uv sync --all-extras
uv pip install --no-deps -e deps/verifiers/environments/reverse_text
exit
```

Notes:

- `env.sh` redirects every cache (uv, pip, HF, torch, triton, vLLM) into `<clone>/cache/`.
  Source it before any `uv` command. **`$HOME` has a 100K inode quota that is nearly
  full** — never let a cache land there.
- `uv sync --all-extras` is the right command here; the CUDA-only extras have been
  removed from this branch, so "all" is safe.
- The second line is needed because `uv sync` does not install workspace members, and
  the smoke test's environment is one.
- `start_shell.sh` needs no edits, but it only binds `/scratch/project_462001516`. That
  is enough to build the venv, not to run the test — see below.

## 5. Run the smoke test

```bash
sbatch run_reverse_text_test.sbatch
```

One `dev-g` node, 2 GPUs, ~16 min wall clock for 5 steps (~6 min of that is vLLM loading
off Lustre). Output lands in `logs/reverse_text_<jobid>.out`. Success looks like:

```
torch 2.13.0+rocm7.2 sees 2 GPU(s): ['AMD Instinct MI250X', 'AMD Instinct MI250X']
...
SUCCESS Training finished!
```

Ignore these — they appear in every run and are harmless: torchao `_C_mxfp8` /
`_C_cutlass_90a` `.so` load failures, `Can't initialize amdsmi - Error code: 34`, and
`Peak FLOPS undefined for AMD Instinct MI250X. Falling back to A100`. That last one means
the reported MFU is meaningless on LUMI; throughput in tokens/s is still real.

### Running it by hand instead

```bash
salloc -p dev-g -A project_462001516 -N1 --gpus-per-node=2 -c28 --hint=multithread -t 1:00:00
srun --pty singularity exec --cleanenv \
    -B /pfs,/scratch,/projappl,/project,/flash,/appl \
    -B "$HOME:$HOME:ro" -B "$PWD/cache/home_cache:$HOME/.cache:rw" -B /tmp:/tmp:rw \
    /appl/local/laifs/containers/lumi-multitorch-latest.sif bash
```

then inside, `source env.sh` and run the body of `run_reverse_text_test.sbatch` (the part
between the two `RUNEOF` markers). The four extra binds over `start_shell.sh` all matter:

| Bind | Why |
|---|---|
| `/pfs,/projappl,/project,/flash,/appl` | the container image itself lives under `/appl`, and Lustre paths resolve through `/pfs` |
| `$HOME:ro` | `uv` and its CPython live in `$HOME/.local`; read-only protects the inode quota |
| `$PWD/cache/home_cache` → `$HOME/.cache:rw` | prime-rl and verifiers both derive cache roots from `Path.home()` with no env override, and the env subprocess runtime *must* be able to write there |
| `/tmp:rw` | MIOpen/comgr SQLite caches; their locking does not work on Lustre and the failure mode is a silent hang |

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Read-only file system: .../.cache/verifiers` | missing the `$HOME/.cache` bind (above) |
| `undefined symbol: amdsmi_get_gpu_enumeration_info` | stale `amd_smi/`; re-run `scripts/setup_amdsmi.sh` |
| `Distribution not found at: file:///.../wheels/...` | the wheel pin in `pyproject.toml` was edited or the wheels dir is unreadable |
| Startup hangs forever with no error | MIOpen cache on Lustre; `MIOPEN_USER_DB_PATH` must be under `/tmp` |
| `No space left` / `Disk quota exceeded` during `uv sync` | caches leaking into `$HOME`; `source env.sh` first |
