# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psutil",
# ]
# ///

import psutil


TOP_N = 5


def get_processes() -> list[dict]:
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_info']):
        try:
            info = proc.info
            processes.append({
                'pid': info['pid'],
                'name': info['name'],
                'status': info['status'],
                'cpu_percent': info['cpu_percent'] or 0.0,
                'memory_mb': (info['memory_info'].rss / (1024**2)) if info['memory_info'] else 0.0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def print_top(label: str, processes: list[dict], key: str, unit: str) -> None:
    print(f'\n--- Top {TOP_N} by {label} ---')
    for rank, p in enumerate(
        sorted(processes, key=lambda x: x[key], reverse=True)[:TOP_N], start=1
    ):
        print(f'  {rank}. [{p["pid"]:>6}] {p["name"]:<30} {p[key]:>8.1f} {unit}')


def print_summary(processes: list[dict]) -> None:
    total = len(processes)
    by_status: dict[str, int] = {}
    for p in processes:
        by_status[p['status']] = by_status.get(p['status'], 0) + 1

    print(f'Total processes: {total}')
    for status, count in sorted(by_status.items()):
        print(f'  {status:<12}: {count}')


def main() -> None:
    # Trigger cpu_percent collection with a brief interval measurement
    for proc in psutil.process_iter(['cpu_percent']):
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    import time
    time.sleep(0.5)

    processes = get_processes()

    print('=== Running Processes Summary ===')
    print_summary(processes)
    print_top('CPU Usage', processes, 'cpu_percent', '%')
    print_top('Memory Usage', processes, 'memory_mb', 'MB')


if __name__ == '__main__':
    main()
