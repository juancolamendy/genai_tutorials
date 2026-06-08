# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psutil",
#   "py-cpuinfo",
# ]
# ///

import platform
import psutil
import cpuinfo


def print_system_info():
    uname = platform.uname()
    print(f'System: {uname.system}')
    print(f'Node: {uname.node}')
    print(f'Release: {uname.release}')
    print(f"Processor: {cpuinfo.get_cpu_info()['brand_raw']}")
    print(f'CPU Cores: {psutil.cpu_count(logical=True)}')
    print(f'Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB')
    print(f"Disk Total: {psutil.disk_usage('/').total / (1024**3):.2f} GB")


if __name__ == '__main__':
    print_system_info()
