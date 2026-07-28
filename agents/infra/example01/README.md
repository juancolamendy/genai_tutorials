# Celery + Valkey

## Run
### Terminal 1 — Valkey
valkey-server

### Terminal 2 — Worker 1, listens to the "default" queue
uv run celery -A celery_app worker --loglevel=info --concurrency=2 -Q default -n worker1@%h

### Terminal 3 — Worker 2, listens to "background"
uv run celery -A celery_app worker --loglevel=info --concurrency=2 -Q background -n worker2@%h

### Terminal 4 — Worker 3, listens to "priority"
uv run celery -A celery_app worker --loglevel=info --concurrency=1 -Q priority -n worker3@%h

### Terminal 5 — Producer CLI
uv run cli.py process-video --video-id abc

uv run cli.py send-email --email alice@example.com --subject "Welcome" --message "Hi there!"
# queued task_id=3f9a1e2c-... queue=default

uv run cli.py generate-report --report-type sales --date-range 2024-01
# queued task_id=8b7d4f10-... queue=priority

## CLI
### See how many tasks are waiting in each queue (queue = a Valkey list)
127.0.0.1:6379> LLEN default
(integer) 3
127.0.0.1:6379> LLEN background
(integer) 1

### Peek at the raw (still-queued) task payload without removing it
127.0.0.1:6379> LINDEX default 0
"{\"body\": \"...\", \"headers\": {...}, \"properties\": {...}}"

### Look up a task's stored result (DB 1, since result_backend = redis://localhost:6379/1)
127.0.0.1:6379> SELECT 1
OK
127.0.0.1:6379[1]> GET celery-task-meta-<task_id>
"{\"status\": \"SUCCESS\", \"result\": {...}, \"date_done\": \"...\"}"

### See the TTL Celery set via result_expires
127.0.0.1:6379[1]> TTL celery-task-meta-<task_id>
(integer) 85999

### List all currently-stored result keys (careful with KEYS on production data — it's O(n) and blocks)
127.0.0.1:6379[1]> SCAN 0 MATCH celery-task-meta-*
