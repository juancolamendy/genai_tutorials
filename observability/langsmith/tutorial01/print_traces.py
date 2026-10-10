"""
Print LangSmith traces/threads for the project configured in the environment.

Usage:
    python print_traces.py
    python print_traces.py --thread-id <thread-id>
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

PROJECT = os.getenv("LANGSMITH_PROJECT", "tutorial01")


def print_thread(client: Client, thread_id: str, project_name: str) -> None:
    """Print root turns and all spans for a specific thread."""
    print("\nALL SPANS")
    for run in client.read_thread(
        thread_id=thread_id,
        project_name=project_name,
        is_root=False,
        order="asc",
    ):
        print(run.id, run.run_type, run.name, run.status)


def print_recent_traces(client: Client, project_name: str, days: int = 1) -> None:
    """Print root traces from the last N days for the project."""
    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(days=days)
    run_filter = (
        f'and(gt(start_time, "{start_time.isoformat()}"), '
        f'lt(end_time, "{end_time.isoformat()}"))'
    )

    print(f"Traces from last {days} day(s) in project {project_name}")
    for run in client.list_runs(
        project_name=project_name,
        is_root=True,
        filter=run_filter,
    ):
        print(run.id, run.name, run.status, run.start_time)


def main():
    parser = argparse.ArgumentParser(
        description="Print LangSmith traces/threads for the configured project."
    )
    parser.add_argument(
        "--thread-id",
        "--thread_id",
        dest="thread_id",
        help="Thread ID to print traces for. If omitted, prints traces from the last day.",
    )
    args = parser.parse_args()

    client = Client()

    if args.thread_id:
        print_thread(client, args.thread_id, PROJECT)
    else:
        print_recent_traces(client, PROJECT)


if __name__ == "__main__":
    main()
