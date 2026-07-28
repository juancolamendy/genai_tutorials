import time
from celery_app import app


@app.task(name="tasks.send_email", bind=True)
def send_email(self, email, subject, message):
    print(f"[{self.request.id}] Sending email to {email}: {subject}")
    time.sleep(2)
    return {"status": "success", "email": email, "task_id": self.request.id}


@app.task(name="tasks.process_video", bind=True)
def process_video(self, video_id, quality="720p"):
    print(f"[{self.request.id}] Processing {video_id} @ {quality}")
    for i in range(5):
        time.sleep(1)
        print(f"[{self.request.id}] progress {(i + 1) * 20}%")
    return {"status": "completed", "video_id": video_id, "quality": quality}


@app.task(name="tasks.generate_report", bind=True)
def generate_report(self, report_type, date_range):
    print(f"[{self.request.id}] Generating {report_type} report for {date_range}")
    time.sleep(3)
    return {
        "status": "completed",
        "report_type": report_type,
        "date_range": date_range,
        "filename": f"{report_type}_{date_range}.pdf",
    }
