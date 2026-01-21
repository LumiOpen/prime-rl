# Installation on TW
It is recommended to install things inside a compute node using the correct container.

1. Start terminal on a compute node.
```bash
srun --partition=amd-tw-verification --nodes=1 --ntasks=3 --cpus-per-task=16 --exclusive=user --mem=480G --gpus-per-node=8  --time=08:00:00 --pty
```

2. Clone this repository and switch to the tw branch.
```bash
git clone git@github.com:LumiOpen/prime-rl.git
cd prime-rl
git checkout tw
```

3. Start container and set needed env variables. Modify this to your liking, currently hardcoded mounts and env paths!
```bash
./start_shell.sh
. env.sh
```

4. Install the uv virtual env (you will need uv installed, pip install uv works). This will build flash-attn from source, which takes a long time.

TODO: build a flash-attn wheel to make this faster in the future.

```bash
uv sync -v
```

# Starting multinode inference engine manually
For manually testing multinode inference, start at least 2 nodes. These instructions are for 2 nodes with DP=16.

In a similar fashion to the installation, the container and env can be set with:
```bash
./start_shell.sh
. env.sh
```
Do this on both nodes.

Starting the head node:
```bash
uv run inference @ configs/deepscaler/stage1/infer.toml \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --data-parallel-size-local 8 \
    --data-parallel-address <INSERT_MASTER_NODE> \
    --data-parallel-rpc-port 29500
```

On the headless node:
```bash
uv run inference @ configs/deepscaler/stage1/infer.toml \
    --data-parallel-size 16 \
    --tensor-parallel-size 1 \
    --data-parallel-size-local 8 \
    --data-parallel-address <INSERT_MASTER_NODE> \
    --data-parallel-rpc-port 29500 \
    --data-parallel-start-rank 8 \
    --headless
```

__Hint:__ use
```bash
export VLLM_LOGGING_LEVEL=DEBUG
```
to see what vLLM is doing.