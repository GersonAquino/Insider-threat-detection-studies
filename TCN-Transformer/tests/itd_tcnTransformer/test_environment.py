"""Synthetic tests for training-environment provenance capture."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.fixture(scope="module")
def modules():
    """Import training settings without requiring a CUDA-capable test host."""
    with patch.object(torch.cuda, "is_available", return_value=True):
        environment = importlib.import_module("src.itd_tcnTransformer.utils.environment")
        logging_utils = importlib.import_module("src.itd_tcnTransformer.utils.logging_utils")
        train = importlib.import_module("src.itd_tcnTransformer.training.train")
    return environment, logging_utils, train


def _fake_torch(device_count=2):
    properties = [
        SimpleNamespace(name="Mock GPU 0", total_memory=8_000_000_000),
        SimpleNamespace(name="Mock GPU 1", total_memory=16_000_000_000),
    ]
    return SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: device_count,
            get_device_properties=lambda index: properties[index],
        ),
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 9010, deterministic=True, benchmark=False, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: True,
    )


def _patch_common_sources(monkeypatch, environment, *, release="6.8.0", torch_module=None):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setattr(environment.subprocess, "run", Mock(side_effect=AssertionError("Unexpected external query")))
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "release", lambda: release)
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.12.3")
    monkeypatch.setattr(environment, "_os_release", lambda: "Ubuntu 24.04")
    monkeypatch.setattr(environment, "_cpu_model", lambda: "Mock CPU")
    monkeypatch.setattr(environment, "_ram_total_gb", lambda: 64.0)
    monkeypatch.setattr(environment.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(environment, "_package_version", lambda package: {"torch": "2.11.0", "numpy": "2.2.0", "scikit-learn": "1.6.0"}[package])
    monkeypatch.setattr(environment, "_torch_module", lambda: torch_module or _fake_torch())
    monkeypatch.setattr(environment, "_nvidia_smi_versions", lambda: ("610.74", "13.3"))


def test_collects_linux_visible_logical_gpus_flags_and_is_json_serialisable(monkeypatch, modules):
    environment, _, _ = modules
    _patch_common_sources(monkeypatch, environment)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setenv("PYTHONHASHSEED", "44")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    result = environment.collect_training_environment("cuda:0", "17")

    assert result["os"] == "Linux"
    assert result["windows_version"] is None
    assert result["unavailable_reasons"]["windows_version"] == "not running under WSL"
    assert result["gpu_count"] == 2
    assert result["gpus"] == [
        {"gpu_index": 0, "gpu_name": "Mock GPU 0", "gpu_vram_gb": 8.0},
        {"gpu_index": 1, "gpu_name": "Mock GPU 1", "gpu_vram_gb": 16.0},
    ]
    assert result["selected_device"] == "cuda:0"
    assert result["cuda_visible_devices"] == "3,7"
    assert result["pythonhashseed"] == "44"
    assert result["pythonhashseed_at_entry"] == "17"
    assert result["torch_deterministic_algorithms"] is True
    assert result["torch_deterministic_algorithms_warn_only"] is True
    assert result["cudnn_deterministic"] is True
    assert result["cudnn_benchmark"] is False
    assert result["cuda_matmul_allow_tf32"] is False
    assert result["cudnn_allow_tf32"] is True
    assert result["nvidia_driver_version"] == "610.74"
    assert result["cuda_driver_version"] == "13.3"
    assert json.loads(json.dumps(result))["gpus"][1]["gpu_index"] == 1


def test_collects_wsl_host_metadata_only_when_running_under_wsl(monkeypatch, modules):
    environment, _, _ = modules
    _patch_common_sources(monkeypatch, environment, release="6.6.1-microsoft-standard-WSL2", torch_module=_fake_torch(0))
    monkeypatch.setattr(environment, "_windows_version", lambda: "Windows 11 10.0 build 26200")
    monkeypatch.setattr(environment, "_wsl_version", lambda: "2.5.10")

    result = environment.collect_training_environment("cpu", None)

    assert result["windows_version"] == "Windows 11 10.0 build 26200"
    assert result["wsl_version"] == "2.5.10"
    assert result["gpu_count"] == 0
    assert result["gpus"] == []
    assert result["unavailable_reasons"]["pythonhashseed_at_entry"] == "environment variable or caller entry value was not set"


def test_wsl_version_accepts_localised_utf16_output_and_partial_nvidia_metadata(monkeypatch, modules):
    environment, _, _ = modules
    _patch_common_sources(monkeypatch, environment, release="6.6.1-microsoft-standard-WSL2")
    monkeypatch.setattr(
        environment,
        "_run_command",
        lambda command: "Versão do WSL: 2.5.10\nVersão do kernel: 6.6.1" if command[0] == "wsl.exe" else "",
    )
    assert environment._decode_command_output("Versão do WSL: 2.5.10".encode("utf-16")) == "Versão do WSL: 2.5.10"
    assert environment._wsl_version() == "2.5.10"
    monkeypatch.setattr(environment, "_nvidia_smi_versions", lambda: ("610.74", None))

    result = environment.collect_training_environment("cuda", "44")

    assert result["configured_device"] == "cuda"
    assert result["nvidia_driver_version"] == "610.74"
    assert result["cuda_driver_version"] is None
    assert result["unavailable_reasons"]["cuda_driver_version"] == "nvidia-smi did not report a usable CUDA version"


def test_native_windows_uses_host_details_and_marks_linux_fields_inapplicable(monkeypatch, modules):
    environment, _, _ = modules
    _patch_common_sources(monkeypatch, environment, torch_module=_fake_torch(0))
    monkeypatch.setattr(
        environment,
        "os",
        SimpleNamespace(name="nt", environ=os.environ, cpu_count=lambda: 24),
    )
    monkeypatch.setattr(
        environment,
        "_windows_system_details",
        lambda: {"windows_version": "Windows 11", "cpu_model": "Mock Windows CPU", "cpu_logical_cores": 24, "ram_total_gb": 64.0},
    )

    result = environment.collect_training_environment("cpu", None)

    assert result["windows_version"] == "Windows 11"
    assert result["cpu_model"] == "Mock Windows CPU"
    assert result["linux_distribution"] is None
    assert result["unavailable_reasons"]["linux_distribution"] == "not applicable when Python runs on native Windows"


def test_per_gpu_property_failure_preserves_other_gpu_metadata(monkeypatch, modules):
    environment, _, _ = modules
    fake_torch = _fake_torch()
    get_properties = Mock(side_effect=[
        SimpleNamespace(name="GPU 0", total_memory=8_000_000_000),
        SimpleNamespace(name="GPU 0", total_memory=8_000_000_000),
        RuntimeError("GPU 1 unavailable"),
        RuntimeError("GPU 1 unavailable"),
    ])
    fake_torch.cuda.get_device_properties = get_properties
    _patch_common_sources(monkeypatch, environment, torch_module=fake_torch)

    result = environment.collect_training_environment("cuda:0", "44")

    assert result["gpus"][0] == {"gpu_index": 0, "gpu_name": "GPU 0", "gpu_vram_gb": 8.0}
    assert result["gpus"][1] == {"gpu_index": 1, "gpu_name": None, "gpu_vram_gb": None}
    assert "RuntimeError: GPU 1 unavailable" in result["unavailable_reasons"]["gpus[1].gpu_name"]


def test_cpu_only_and_unavailable_observations_keep_nulls_and_reasons(monkeypatch, modules):
    environment, _, _ = modules
    _patch_common_sources(monkeypatch, environment, torch_module=_fake_torch(0))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(environment, "_nvidia_smi_versions", Mock(side_effect=FileNotFoundError("nvidia-smi")))
    monkeypatch.setattr(environment, "_package_version", Mock(side_effect=LookupError("not installed")))

    result = environment.collect_training_environment("cpu", None)

    assert result["gpu_count"] == 0
    assert result["gpus"] == []
    assert result["pytorch_version"] is None
    assert result["nvidia_driver_version"] is None
    assert result["cuda_driver_version"] is None
    assert result["unavailable_reasons"]["pytorch_version"] == "LookupError: not installed"
    assert result["unavailable_reasons"]["nvidia_driver_version"] == "FileNotFoundError: nvidia-smi"
    assert result["unavailable_reasons"]["cuda_visible_devices"] == "environment variable or caller entry value was not set"


def test_command_helper_uses_bounded_shell_free_queries_and_propagates_failures(monkeypatch, modules):
    environment, _, _ = modules
    completed = SimpleNamespace(stdout=" output\n")
    run = Mock(return_value=completed)
    monkeypatch.setattr(environment.subprocess, "run", run)

    assert environment._run_command(["nvidia-smi"]) == "output"
    _, kwargs = run.call_args
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 5
    assert kwargs["check"] is True

    monkeypatch.setattr(
        environment.subprocess,
        "run",
        Mock(side_effect=subprocess.TimeoutExpired(["nvidia-smi"], 5)),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        environment._run_command(["nvidia-smi"])


def _reproducibility_report(entry_value="entry-seed"):
    return {
        "random_seed": 44,
        "deterministic_training": True,
        "deterministic_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
        "torch_initial_seed": 44,
        "cuda_available": False,
        "cuda_device_count": 0,
        "torch_version": "2.11.0",
        "pythonhashseed_at_entry": entry_value,
    }


def test_timeout_is_recorded_without_interrupting_collection(monkeypatch, modules):
    environment, _, _ = modules
    _patch_common_sources(monkeypatch, environment)
    monkeypatch.setattr(environment, "_nvidia_smi_versions", Mock(
        side_effect=subprocess.TimeoutExpired(["nvidia-smi"], 5)))
    result = environment.collect_training_environment("cpu", None)
    assert result["nvidia_driver_version"] is None
    assert "TimeoutExpired" in result["unavailable_reasons"]["nvidia_driver_version"]
    assert result["cpu_model"] == "Mock CPU"


def test_nvidia_parser_retains_driver_when_cuda_is_unavailable(monkeypatch, modules):
    environment, _, _ = modules
    monkeypatch.setattr(environment, "_run_command", lambda _: "Driver Version: 610.74 | CUDA Version: N/A |")
    assert environment._nvidia_smi_versions() == ("610.74", None)


def test_linux_memory_uses_decimal_gb(monkeypatch, modules):
    environment, _, _ = modules
    monkeypatch.setattr(environment, "_read_text", lambda _: "MemTotal:       1000000 kB\n")
    assert environment._ram_total_gb() == 1.024


@pytest.mark.parametrize("phase", ["single_stage", "pretrain", "finetune"])
def test_build_training_run_log_collects_environment_for_each_training_phase(monkeypatch, modules, phase):
    _, _, train = modules
    environment = {"selected_device": "cpu", "phase_capture": phase}
    collector = Mock(return_value=environment)
    monkeypatch.setattr(train, "collect_training_environment", collector)
    monkeypatch.setattr(train, "_get_batch_shapes", lambda *_: ([2, 30, 3], [2, 30]))
    trainer = SimpleNamespace(
        checkpoint_path=Path("checkpoint.pt"), scaler_path=None, feature_columns=["a", "b", "c"],
        n_features=3, learning_rate=0.001, weight_decay=0.0, gradient_clip_norm=None,
        threshold=0.5, early_stopping_patience=3, device="cpu", loss_hyperparameters=None,
    )
    model = SimpleNamespace(
        hidden_dim=128, dropout=0.1,
        transformer_encoder=SimpleNamespace(num_heads=8, feedforward_dim=256),
    )

    log = train._build_training_run_log(
        trainer, model, object(), object(), phase, "validation", 2,
        _reproducibility_report(), {},
    )

    collector.assert_called_once_with(device="cpu", pythonhashseed_at_entry="entry-seed")
    assert log.training_phase == phase
    assert log.environment == environment
    assert json.loads(json.dumps(log, default=lambda value: value.__dict__))["environment"] == environment


def test_main_captures_pythonhashseed_before_reproducibility_setup(monkeypatch, modules):
    _, _, train = modules
    monkeypatch.setattr(train, "TRAINING_MODE", "single_stage")
    monkeypatch.setenv("PYTHONHASHSEED", "before")
    calls = []

    def set_reproducibility(**kwargs):
        calls.append(kwargs)
        os.environ["PYTHONHASHSEED"] = "after"

    report = _reproducibility_report(entry_value=None)
    captured = {}
    monkeypatch.setattr(train, "set_reproducibility", set_reproducibility)
    monkeypatch.setattr(train, "build_reproducibility_report", lambda **_: report.copy())
    monkeypatch.setattr(train, "create_sequence_dataloaders", lambda **_: {"train": object()})
    monkeypatch.setattr(train, "_run_single_stage_training", lambda _dataloaders, reproducibility_report, _params: captured.setdefault("report", reproducibility_report) or "history")

    train.main()

    assert calls
    assert captured["report"]["pythonhashseed_at_entry"] == "before"
    assert os.environ["PYTHONHASHSEED"] == "after"


def test_training_run_log_keeps_legacy_construction_serialisable_without_environment(modules, tmp_path):
    _, logging_utils, _ = modules
    log = logging_utils.TrainingRunLog(
        training_mode="single_stage", split_strategy="temporal", training_phase="single_stage",
        model=None, loss=None, evaluation=None, optimization=None, reproducibility=None,
        artifacts=None, shapes=None,
    )
    output_path = tmp_path / "training_run_log.json"
    logging_utils.save_training_run_log(log, output_path)

    assert log.environment is None
    assert json.loads(output_path.read_text(encoding="utf-8"))["environment"] is None
