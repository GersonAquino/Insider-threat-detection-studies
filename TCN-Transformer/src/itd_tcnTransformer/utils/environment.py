"""Best-effort capture of the runtime environment used for model training.

The collector deliberately has no mandatory third-party dependencies.  Missing
tools, unsupported hardware, and partially configured CUDA installations are
represented in the returned metadata instead of raising during training.
"""

from __future__ import annotations

import datetime as _datetime
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Callable


_COMMAND_TIMEOUT_SECONDS = 5


def _run_command(command: list[str]) -> str:
    """Run a small system query without a shell and return stripped stdout."""
    options: dict[str, Any] = {
        "capture_output": True,
        "check": True,
        "timeout": _COMMAND_TIMEOUT_SECONDS,
        "shell": False,
    }
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        options["startupinfo"] = startupinfo
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return _decode_command_output(subprocess.run(command, **options).stdout).strip()


def _decode_command_output(output: str | bytes | None) -> str:
    """Decode normal and UTF-16 Windows command output without locale assumptions."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if output.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in output[:64]:
        return output.decode("utf-16", errors="replace")
    return output.decode("utf-8", errors="replace")


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _os_release() -> str:
    values: dict[str, str] = {}
    for line in _read_text("/etc/os-release").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values.get("PRETTY_NAME") or values.get("NAME") or ""


def _cpu_model() -> str:
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return ""


def _ram_total_gb() -> float:
    for line in _read_text("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return round((kib * 1024) / 1_000_000_000, 3)
    raise ValueError("MemTotal is absent from /proc/meminfo")


def _package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _windows_version() -> str:
    # This command is intentionally used only through powershell.exe in WSL.
    output = _run_command(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem | "
            "ForEach-Object { \"$($_.Caption) $($_.Version) build $($_.BuildNumber)\" })",
        ]
    )
    return output.strip()


def _windows_system_details() -> dict[str, Any]:
    """Read host details when the training process itself runs on Windows."""
    output = _run_command(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$os=Get-CimInstance Win32_OperatingSystem; $cpu=Get-CimInstance Win32_Processor; "
            "[pscustomobject]@{windows_version=\"$($os.Caption) $($os.Version) build $($os.BuildNumber)\"; "
            "cpu_model=($cpu | Select-Object -First 1 -ExpandProperty Name); "
            "cpu_logical_cores=(($cpu | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum); "
            "ram_total_gb=[math]::Round(((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1000000000),3)} | ConvertTo-Json -Compress",
        ]
    )
    details = json.loads(output)
    if not isinstance(details, dict):
        raise ValueError("PowerShell did not return a system-details object")
    return details


def _wsl_version() -> str:
    """Return the installed WSL package version, never the Linux kernel version."""
    output = _run_command(["wsl.exe", "--version"])
    for line in output.splitlines():
        # WSL output is localised (for example, "Versão do WSL") and may use
        # a UTF-16 encoding.  The first label containing WSL is its package
        # version; kernel lines are deliberately ignored.
        normalised = line.casefold()
        if "wsl" in normalised and "kernel" not in normalised and ":" in line:
            return line.split(":", 1)[1].strip()
    raise ValueError("WSL version was not present in 'wsl.exe --version' output")


def _nvidia_smi_versions() -> tuple[str | None, str | None]:
    output = _run_command(["nvidia-smi"])
    driver = re.search(r"Driver Version:\s*([^|\s]+)", output)
    cuda = re.search(r"CUDA Version:\s*([^|\s]+)", output)
    driver_value = driver.group(1) if driver else None
    cuda_value = cuda.group(1) if cuda else None
    if driver_value and driver_value.casefold() == "n/a":
        driver_value = None
    if cuda_value and cuda_value.casefold() == "n/a":
        cuda_value = None
    if not driver_value and not cuda_value:
        raise ValueError("nvidia-smi header did not contain usable driver or CUDA versions")
    return driver_value, cuda_value


def _torch_module() -> Any:
    # Import only when the collector is called, keeping non-training imports light.
    import torch

    return torch


def _unavailable_reason(error: Exception | None = None) -> str:
    if error is None:
        return "not available in this runtime"
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def collect_training_environment(
    device: str, pythonhashseed_at_entry: str | None
) -> dict[str, Any]:
    """Collect JSON-serialisable training-runtime metadata without raising.

    ``device`` is retained verbatim as the device selected by the caller.  The
    result describes what was observable at collection time; it does not alter
    PyTorch deterministic or TF32 settings.
    """
    data: dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp": _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "os": None,
        "windows_version": None,
        "wsl_version": None,
        "linux_distribution": None,
        "kernel_version": None,
        "python_version": None,
        "pytorch_version": None,
        "numpy_version": None,
        "sklearn_version": None,
        "cpu_model": None,
        "cpu_logical_cores": None,
        "ram_total_gb": None,
        "gpu_count": None,
        "gpus": None,
        "configured_device": device,
        "selected_device": device,
        "nvidia_driver_version": None,
        "cuda_driver_version": None,
        "pytorch_cuda_version": None,
        "cudnn_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pythonhashseed_at_entry": pythonhashseed_at_entry,
        "torch_deterministic_algorithms": None,
        "torch_deterministic_algorithms_warn_only": None,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "cuda_matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
    }
    sources = {
        "schema_version": "environment collector schema",
        "timestamp": "Python UTC clock",
        "os": "platform.system",
        "windows_version": "powershell.exe -NoProfile -NonInteractive Get-CimInstance Win32_OperatingSystem",
        "wsl_version": "wsl.exe --version (installed WSL package)",
        "linux_distribution": "/etc/os-release",
        "kernel_version": "platform.release",
        "python_version": "platform.python_version",
        "pytorch_version": "importlib.metadata (torch)",
        "numpy_version": "importlib.metadata (numpy)",
        "sklearn_version": "importlib.metadata (scikit-learn)",
        "cpu_model": "/proc/cpuinfo",
        "cpu_logical_cores": "os.cpu_count",
        "ram_total_gb": "/proc/meminfo MemTotal",
        "gpu_count": "torch.cuda.device_count (visible logical devices)",
        "gpus": "torch.cuda.get_device_properties (visible logical devices)",
        "configured_device": "training caller argument",
        "selected_device": "resolved training device (torch.cuda.current_device for unindexed CUDA)",
        "nvidia_driver_version": "nvidia-smi header",
        "cuda_driver_version": "nvidia-smi header CUDA Version (maximum supported CUDA version)",
        "pytorch_cuda_version": "torch.version.cuda",
        "cudnn_version": "torch.backends.cudnn.version",
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES environment variable",
        "pythonhashseed": "PYTHONHASHSEED environment variable",
        "cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG environment variable",
        "pythonhashseed_at_entry": (
            "training caller argument captured before reproducibility configuration; "
            "a runtime assignment does not change interpreter hash randomisation"
        ),
        "torch_deterministic_algorithms": "torch.are_deterministic_algorithms_enabled",
        "torch_deterministic_algorithms_warn_only": "torch.is_deterministic_algorithms_warn_only_enabled",
        "cudnn_deterministic": "torch.backends.cudnn.deterministic",
        "cudnn_benchmark": "torch.backends.cudnn.benchmark",
        "cuda_matmul_allow_tf32": "torch.backends.cuda.matmul.allow_tf32",
        "cudnn_allow_tf32": "torch.backends.cudnn.allow_tf32",
    }
    unavailable: dict[str, str] = {}

    def capture(field: str, getter: Callable[[], Any]) -> None:
        try:
            value = getter()
            if value is None or value == "":
                unavailable[field] = "source returned no value"
            else:
                data[field] = value
        except Exception as error:  # Environment collection must not halt training.
            unavailable[field] = _unavailable_reason(error)

    capture("os", platform.system)
    capture("python_version", platform.python_version)
    capture("pytorch_version", lambda: _package_version("torch"))
    capture("numpy_version", lambda: _package_version("numpy"))
    capture("sklearn_version", lambda: _package_version("scikit-learn"))
    if os.name == "nt":
        unavailable["kernel_version"] = "not applicable when Python runs on native Windows"
        unavailable["linux_distribution"] = "not applicable when Python runs on native Windows"
        sources.update({
            "cpu_model": "powershell.exe Win32_Processor.Name",
            "cpu_logical_cores": "powershell.exe Win32_Processor.NumberOfLogicalProcessors",
            "ram_total_gb": "powershell.exe Win32_ComputerSystem.TotalPhysicalMemory",
        })
        try:
            windows_details = _windows_system_details()
            for field in ("windows_version", "cpu_model", "cpu_logical_cores", "ram_total_gb"):
                value = windows_details.get(field)
                if value is None or value == "":
                    unavailable[field] = "PowerShell returned no value"
                else:
                    data[field] = value
        except Exception as error:
            reason = _unavailable_reason(error)
            for field in ("windows_version", "cpu_model", "cpu_logical_cores", "ram_total_gb"):
                unavailable[field] = reason
    else:
        capture("kernel_version", platform.release)
        capture("linux_distribution", _os_release)
        capture("cpu_model", _cpu_model)
        capture("cpu_logical_cores", lambda: os.cpu_count())
        capture("ram_total_gb", _ram_total_gb)

    try:
        is_wsl = "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ
    except Exception:
        # Failure to identify WSL must not affect the other independent fields.
        is_wsl = "WSL_DISTRO_NAME" in os.environ
    if is_wsl:
        capture("windows_version", _windows_version)
        capture("wsl_version", _wsl_version)
    elif os.name != "nt":
        unavailable["windows_version"] = "not running under WSL"
        unavailable["wsl_version"] = "not running under WSL"
    else:
        unavailable["wsl_version"] = "not applicable when Python runs on native Windows"

    torch: Any | None = None
    try:
        torch = _torch_module()
    except Exception as error:
        reason = _unavailable_reason(error)
        for field in (
            "gpu_count", "gpus", "pytorch_cuda_version", "cudnn_version",
            "torch_deterministic_algorithms", "torch_deterministic_algorithms_warn_only",
            "cudnn_deterministic", "cudnn_benchmark", "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        ):
            unavailable[field] = reason
        if str(device).strip().lower() == "cuda":
            data["selected_device"] = None
            unavailable["selected_device"] = reason

    if torch is not None:
        # ``cuda`` is resolved to the visible logical device actually selected
        # by PyTorch, while the caller's original setting remains available.
        if str(device).strip().lower() == "cuda":
            try:
                data["selected_device"] = f"cuda:{torch.cuda.current_device()}"
            except Exception as error:
                data["selected_device"] = None
                unavailable["selected_device"] = _unavailable_reason(error)
        capture("gpu_count", lambda: torch.cuda.device_count())

        def gpu_properties() -> list[dict[str, Any]]:
            count = torch.cuda.device_count()
            devices: list[dict[str, Any]] = []
            for index in range(count):
                entry: dict[str, Any] = {"gpu_index": index, "gpu_name": None, "gpu_vram_gb": None}
                try:
                    entry["gpu_name"] = torch.cuda.get_device_properties(index).name
                except Exception as error:
                    unavailable[f"gpus[{index}].gpu_name"] = _unavailable_reason(error)
                try:
                    properties = torch.cuda.get_device_properties(index)
                    entry["gpu_vram_gb"] = round(properties.total_memory / 1_000_000_000, 3)
                except Exception as error:
                    unavailable[f"gpus[{index}].gpu_vram_gb"] = _unavailable_reason(error)
                devices.append(entry)
            return devices

        capture("gpus", gpu_properties)
        capture("pytorch_cuda_version", lambda: torch.version.cuda)
        capture("cudnn_version", lambda: torch.backends.cudnn.version())
        capture("torch_deterministic_algorithms", lambda: torch.are_deterministic_algorithms_enabled())
        capture(
            "torch_deterministic_algorithms_warn_only",
            lambda: torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        capture("cudnn_deterministic", lambda: torch.backends.cudnn.deterministic)
        capture("cudnn_benchmark", lambda: torch.backends.cudnn.benchmark)
        capture("cuda_matmul_allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32)
        capture("cudnn_allow_tf32", lambda: torch.backends.cudnn.allow_tf32)

    try:
        driver, cuda_driver = _nvidia_smi_versions()
        if driver is None:
            unavailable["nvidia_driver_version"] = "nvidia-smi did not report a usable driver version"
        else:
            data["nvidia_driver_version"] = driver
        if cuda_driver is None:
            unavailable["cuda_driver_version"] = "nvidia-smi did not report a usable CUDA version"
        else:
            data["cuda_driver_version"] = cuda_driver
    except Exception as error:
        reason = _unavailable_reason(error)
        unavailable["nvidia_driver_version"] = reason
        unavailable["cuda_driver_version"] = reason

    for field in ("cuda_visible_devices", "pythonhashseed", "cublas_workspace_config", "pythonhashseed_at_entry"):
        if data[field] is None:
            unavailable[field] = "environment variable or caller entry value was not set"

    # A zero-device CUDA runtime is still a successful observation, so keep an
    # empty GPU list rather than treating it as missing metadata.
    if data["gpu_count"] == 0 and data["gpus"] is None:
        data["gpus"] = []

    return {
        **data,
        "sources": sources,
        "unavailable_reasons": unavailable,
        "schema_notes": (
            "All values describe the process at collection time. GPU indices are "
            "logical indices after CUDA_VISIBLE_DEVICES filtering. RAM is the total "
            "memory exposed to the execution OS (the Linux guest under WSL). "
            "pythonhashseed_at_entry is observed at training main entry, not verified "
            "interpreter startup. Setting PYTHONHASHSEED during execution does not "
            "retroactively change the running interpreter's hash randomisation."
        ),
        "units": {
            "ram_total_gb": "decimal GB (10^9 bytes)",
            "gpus[].gpu_vram_gb": "decimal GB (10^9 bytes)",
            "timestamp": "UTC ISO 8601",
        },
    }
