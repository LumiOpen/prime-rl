import torch


def is_rocm() -> bool:
    """Whether torch is a ROCm build (AMD GPUs).

    ROCm torch exposes AMD cards through the CUDA API, so ``torch.cuda`` works and
    ``get_device_capability()`` returns an NVIDIA-shaped tuple — gfx90a reports
    ``(9, 0)``, numerically identical to Hopper. Anything branching on the compute
    capability alone will misidentify an MI250X as an H100, so check this first.

    NVML (``pynvml``, ``libnvidia-ml.so``) does not exist on these machines at all.
    """
    return torch.version.hip is not None
