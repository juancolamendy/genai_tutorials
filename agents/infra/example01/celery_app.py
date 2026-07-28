from celery import Celery
from celery_config import Config

app = Celery("celery_valkey_demo", include=["tasks"])
app.config_from_object(Config)

