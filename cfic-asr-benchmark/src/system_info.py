"""System and hardware inspection utility for CFIC x VoiceShield ASR Benchmark."""

import os
import sys
import platform
import shutil
import subprocess
import json
from typing import Dict, Any

def get_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "python_executable": sys.executable,
    }

    # CPU info
    info["processor"] = platform.processor()
    info["logical_cpus"] = os.cpu_count() or 0
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            info["cpu_model"] = cpu_name.strip()
        except Exception:
            info["cpu_model"] = platform.processor()
    else:
        info["cpu_model"] = platform.processor()

    # Memory info via psutil or ctypes fallback
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 2)
        info["available_ram_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["total_ram_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
            info["available_ram_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
        except Exception as e:
            info["ram_error"] = str(e)

    # Disk info
    try:
        total, used, free = shutil.disk_usage(".")
        info["disk_total_gb"] = round(total / (1024**3), 2)
        info["disk_free_gb"] = round(free / (1024**3), 2)
    except Exception as e:
        info["disk_error"] = str(e)

    # GPU / CUDA detection
    info["cuda_available"] = False
    info["gpu_model"] = "None"
    info["gpu_vram_gb"] = 0.0

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_model"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except ImportError:
        info["torch_version"] = "not_installed"

    if info["gpu_model"] == "None" and platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}")
            i = 0
            adapters = []
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, i)
                    if subkey_name.isdigit():
                        skey = winreg.OpenKey(key, subkey_name)
                        desc, _ = winreg.QueryValueEx(skey, "DriverDesc")
                        adapters.append(desc)
                    i += 1
                except OSError:
                    break
            if adapters:
                info["gpu_model"] = "; ".join(adapters)
        except Exception:
            pass

    # Git status
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        info["git_commit"] = res.stdout.strip()
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        info["git_dirty"] = bool(status_res.stdout.strip())
    except Exception:
        info["git_commit"] = "unknown"
        info["git_dirty"] = False

    return info

if __name__ == "__main__":
    data = get_system_info()
    print(json.dumps(data, indent=2))
