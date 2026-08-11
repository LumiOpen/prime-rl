from unittest.mock import patch

import pytest

from prime_rl.configs.shared import GPU_VISIBILITY_ENV_VARS, reject_protected_env_vars
from prime_rl.entrypoints.rl import get_gpu_visibility, with_gpu_visibility


def _clear_gpu_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in GPU_VISIBILITY_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


@pytest.mark.parametrize("variable", GPU_VISIBILITY_ENV_VARS)
def test_get_gpu_visibility_accepts_platform_variables(variable: str, monkeypatch: pytest.MonkeyPatch):
    _clear_gpu_visibility(monkeypatch)
    monkeypatch.setenv(variable, "GPU-deadbeef, 2")

    assert get_gpu_visibility() == (variable, ["GPU-deadbeef", "2"])


def test_get_gpu_visibility_preserves_cuda_precedence(monkeypatch: pytest.MonkeyPatch):
    _clear_gpu_visibility(monkeypatch)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert get_gpu_visibility() == ("CUDA_VISIBLE_DEVICES", ["0"])


def test_get_gpu_visibility_preserves_empty_visibility(monkeypatch: pytest.MonkeyPatch):
    _clear_gpu_visibility(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with patch("prime_rl.entrypoints.rl.pynvml") as pynvml:
        assert get_gpu_visibility() == ("CUDA_VISIBLE_DEVICES", [])
        pynvml.nvmlInit.assert_not_called()


def test_get_gpu_visibility_falls_back_to_nvml(monkeypatch: pytest.MonkeyPatch):
    _clear_gpu_visibility(monkeypatch)

    with patch("prime_rl.entrypoints.rl.pynvml") as pynvml:
        pynvml.nvmlDeviceGetCount.return_value = 2

        assert get_gpu_visibility() == ("CUDA_VISIBLE_DEVICES", ["0", "1"])
        pynvml.nvmlInit.assert_called_once_with()


def test_with_gpu_visibility_removes_conflicting_masks():
    env = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "ROCR_VISIBLE_DEVICES": "2,3",
        "HIP_VISIBLE_DEVICES": "4,5",
        "OTHER": "value",
    }

    assert with_gpu_visibility(env, "ROCR_VISIBLE_DEVICES", ["2"]) == {
        "ROCR_VISIBLE_DEVICES": "2",
        "OTHER": "value",
    }


@pytest.mark.parametrize("variable", GPU_VISIBILITY_ENV_VARS)
def test_component_env_cannot_override_gpu_visibility(variable: str):
    with pytest.raises(ValueError, match=variable):
        reject_protected_env_vars({variable: "0"})
