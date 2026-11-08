"""Tests for the process re-exec used to establish training environment variables."""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock

import pytest

from src.itd_tcnTransformer.utils import training_bootstrap


@pytest.mark.parametrize(
    ("pythonhashseed", "cublas_workspace_config"),
    [("different", ":4096:8"), ("17", ":16:8")],
)
def test_prepare_training_process_reexecutes_when_configuration_differs(
    monkeypatch, pythonhashseed, cublas_workspace_config
):
    monkeypatch.setenv("PYTHONHASHSEED", pythonhashseed)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", cublas_workspace_config)
    monkeypatch.setattr(sys, "argv", ["pytest", "--config", "custom.yaml"])
    execve = Mock()
    monkeypatch.setattr(training_bootstrap.os, "execve", execve)

    training_bootstrap.prepare_training_process(17, ":4096:8")

    execve.assert_called_once()
    executable, argv, environment = execve.call_args.args
    assert executable == sys.executable
    assert argv == [
        sys.executable,
        "-m",
        "src.itd_tcnTransformer.scripts.06_train_model",
        "--config",
        "custom.yaml",
    ]
    assert environment["PYTHONHASHSEED"] == "17"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["PATH"] == os.environ["PATH"]


def test_prepare_training_process_does_not_reexec_when_configuration_matches(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    execve = Mock()
    monkeypatch.setattr(training_bootstrap.os, "execve", execve)

    result = training_bootstrap.prepare_training_process(17, ":4096:8")

    assert result is None
    execve.assert_not_called()


def test_prepare_training_process_preserves_environment_and_all_cli_arguments(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "wrong")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "wrong")
    monkeypatch.setenv("TRAINING_TEST_MARKER", "preserve-me")
    monkeypatch.setattr(sys, "argv", ["launcher", "--seed", "17", "--", "value with spaces"])
    execve = Mock()
    monkeypatch.setattr(training_bootstrap.os, "execve", execve)

    training_bootstrap.prepare_training_process(17, ":4096:8")

    executable, argv, environment = execve.call_args.args
    assert executable == sys.executable
    assert argv[3:] == ["--seed", "17", "--", "value with spaces"]
    assert environment["TRAINING_TEST_MARKER"] == "preserve-me"
    assert environment["PYTHONHASHSEED"] == "17"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

