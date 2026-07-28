import asyncio
import click
from celery_app import app
from tasks import send_email, process_video, generate_report


@click.group()
def cli():
    """Producer CLI — enqueues Celery tasks onto Valkey."""
    pass


@cli.command("send-email")
@click.option("--email", required=True)
@click.option("--subject", required=True)
@click.option("--message", required=True)
def send_email_cmd(email, subject, message):
    result = send_email.delay(email, subject, message)
    click.echo(f"queued task_id={result.id}")


@cli.command("process-video")
@click.option("--video-id", required=True)
@click.option("--quality", default="720p")
def process_video_cmd(video_id, quality):
    result = process_video.delay(video_id, quality)
    click.echo(f"queued task_id={result.id}")


@cli.command("generate-report")
@click.option("--report-type", required=True)
@click.option("--date-range", required=True)
def generate_report_cmd(report_type, date_range):
    result = generate_report.delay(report_type, date_range)
    click.echo(f"queued task_id={result.id}")


# ---- status & inspection: these benefit from async, see explanation below ----

async def _fetch_status(task_id: str) -> dict:
    loop = asyncio.get_running_loop()
    ar = app.AsyncResult(task_id)
    # AsyncResult.state / .result do a blocking network round trip to Valkey;
    # push that off the event loop so it doesn't block.
    state = await loop.run_in_executor(None, lambda: ar.state)
    payload = {"task_id": task_id, "status": state}
    if state == "SUCCESS":
        payload["result"] = await loop.run_in_executor(None, lambda: ar.result)
    elif state == "FAILURE":
        payload["error"] = str(await loop.run_in_executor(None, lambda: ar.info))
    return payload


@cli.command("task-status")
@click.argument("task_ids", nargs=-1, required=True)
@click.option("--watch", is_flag=True, help="Poll every 2s until all tasks finish")
def task_status_cmd(task_ids, watch):
    """Check one or more task IDs. Checking several concurrently is where
    async actually pays off — N blocking Valkey round trips become one await."""

    async def run():
        terminal = {"SUCCESS", "FAILURE"}
        while True:
            results = await asyncio.gather(*(_fetch_status(tid) for tid in task_ids))
            for r in results:
                click.echo(r)
            if not watch or all(r["status"] in terminal for r in results):
                break
            await asyncio.sleep(2)

    asyncio.run(run())


@cli.command("list-active-tasks")
def list_active_tasks_cmd():
    """Ask every connected worker what it's currently running.
    control.inspect() fans out one RPC per worker over the broker;
    gathering them concurrently avoids serial latency as workers grow."""

    async def run():
        loop = asyncio.get_running_loop()
        inspect = app.control.inspect()
        active = await loop.run_in_executor(None, inspect.active)
        if not active:
            click.echo("no workers connected")
            return
        for worker, tasks in active.items():
            click.echo(f"{worker}: {len(tasks)} active")
            for t in tasks:
                click.echo(f"  - {t['id']} {t['name']} args={t['args']}")

    asyncio.run(run())


if __name__ == "__main__":
    cli()
