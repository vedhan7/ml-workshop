"""
Scheduler configuration using APScheduler.
"""

from datetime import datetime

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import get_settings

logger = structlog.get_logger(__name__)

_scheduler: BackgroundScheduler = None


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the scheduler."""
    global _scheduler
    
    if _scheduler is not None:
        return _scheduler
    
    _scheduler = BackgroundScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 60,
        },
    )
    
    return _scheduler


def start_scheduler() -> BackgroundScheduler:
    """Start the scheduler with configured jobs."""
    scheduler = create_scheduler()
    
    if scheduler.running:
        logger.info("scheduler_already_running")
        return scheduler
    
    from src.orchestration.tasks import (
        daily_data_ingestion,
        weekly_model_retrain,
        hourly_prediction_batch,
        data_quality_check,
    )
    
    # Daily data ingestion at 1 AM UTC
    scheduler.add_job(
        daily_data_ingestion,
        trigger=CronTrigger(hour=1, minute=0),
        id="daily_data_ingestion",
        name="Daily Data Ingestion",
        replace_existing=True,
    )
    
    # Weekly model retraining on Sundays at 3 AM UTC
    scheduler.add_job(
        weekly_model_retrain,
        trigger=CronTrigger(day_of_week="sun", hour=3, minute=0),
        id="weekly_model_retrain",
        name="Weekly Model Retrain",
        replace_existing=True,
    )
    
    # Hourly batch predictions
    scheduler.add_job(
        hourly_prediction_batch,
        trigger=IntervalTrigger(hours=1),
        id="hourly_predictions",
        name="Hourly Batch Predictions",
        replace_existing=True,
    )
    
    # Daily data quality check at 6 AM UTC
    scheduler.add_job(
        data_quality_check,
        trigger=CronTrigger(hour=6, minute=0),
        id="data_quality_check",
        name="Daily Data Quality Check",
        replace_existing=True,
    )
    
    scheduler.start()
    logger.info("scheduler_started", jobs=len(scheduler.get_jobs()))
    
    return scheduler


def stop_scheduler() -> None:
    """Stop the scheduler."""
    global _scheduler
    
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=True)
        logger.info("scheduler_stopped")


def get_scheduler() -> BackgroundScheduler:
    """Get the scheduler instance."""
    global _scheduler
    
    if _scheduler is None:
        _scheduler = create_scheduler()
    
    return _scheduler


def list_jobs() -> list[dict]:
    """List all scheduled jobs."""
    scheduler = get_scheduler()
    
    return [
        {
            "id": job.id,
            "name": job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
        }
        for job in scheduler.get_jobs()
    ]


def run_job_now(job_id: str) -> bool:
    """Manually trigger a job to run immediately."""
    scheduler = get_scheduler()
    
    job = scheduler.get_job(job_id)
    if job is None:
        return False
    
    # Run immediately
    job.modify(next_run_time=datetime.utcnow())
    logger.info("job_triggered_manually", job_id=job_id)
    return True
