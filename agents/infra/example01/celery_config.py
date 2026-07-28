class Config:
    # Broker: use the redis:// scheme even though the server is Valkey —
    # Valkey speaks the Redis protocol, and Kombu only knows the "redis" transport name.
    broker_url = "redis://localhost:6379/0"

    # Result backend: separate logical DB (index 1) so results don't collide with queue data
    result_backend = "redis://localhost:6379/1"

    task_serializer = "json"
    accept_content = ["json"]
    result_serializer = "json"
    timezone = "UTC"

    task_track_started = True
    task_time_limit = 30 * 60
    task_soft_time_limit = 25 * 60

    worker_prefetch_multiplier = 1
    worker_max_tasks_per_child = 1000

    # Queue topology — explained in detail below
    task_default_queue = "default"
    task_routes = {
        "tasks.send_email": {"queue": "default"},
        "tasks.process_video": {"queue": "background"},
        "tasks.generate_report": {"queue": "priority"},
    }
    task_queues = {
        "default": {"exchange": "default", "routing_key": "default"},
        "priority": {"exchange": "priority", "routing_key": "priority"},
        "background": {"exchange": "background", "routing_key": "background"},
    }

    result_expires = 86400  # seconds; auto-expire result keys in Valkey
