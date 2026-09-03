from apscheduler.schedulers.blocking import BlockingScheduler
from retrain import get_champion_accuracy, train_new_model, promote_if_better
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrain_job():
    logger.info("Starting retraining job...")
    champion_acc = get_champion_accuracy()
    logger.info(f"Champion accuracy: {champion_acc}")
    run_id, new_acc = train_new_model()
    logger.info(f"New model accuracy: {new_acc}")
    promote_if_better(run_id, new_acc, champion_acc)
    logger.info("Retraining job completed.")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(retrain_job, "interval", minutes=1)
    logger.info("Scheduler started. Retraining every 1 minute...")
    logger.info("Press Ctrl+C to stop")
     # Run once immediately before waiting
    retrain_job()
    
    scheduler.start()
