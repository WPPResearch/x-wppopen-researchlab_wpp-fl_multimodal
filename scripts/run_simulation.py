import logging
import os
import random
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
from flwr.simulation import run_simulation
from loguru import logger

from src.client_app import get_client_fn
from src.server_app import get_server_fn
from src.settings import settings

# Load .env FIRST
load_dotenv()

# Set Ray environment variables BEFORE any other imports to ensure they take effect
os.environ["RAY_metrics_export_port"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_ENABLE_PROMETHEUS_EXPORT"] = "0"
os.environ["GLOG_minloglevel"] = "3"  # 0:INFO, 1:WARN, 2:ERROR, 3:FATAL
os.environ["RAY_BACKEND_LOG_LEVEL"] = "fatal"
os.environ["FL_PROJECT_ROOT"] = str(Path(__file__).parent.parent)

# Silence all Python warnings
warnings.filterwarnings("ignore")


def simulate() -> None:
    try:
        start_time = time.time()
        random.seed(settings.general.random_seed)
        # Start simulation
        run_simulation(
            server_app=get_server_fn(),
            client_app=get_client_fn(),
            num_supernodes=settings.client.num_clients,
            backend_config={
                "client_resources": {
                    "num_cpus": settings.backend.client_resources["num_cpus"],
                    "num_gpus": settings.backend.client_resources["num_gpus"],
                },
                "ray_init_args": {
                    "include_dashboard": False,
                    "logging_level": logging.FATAL,
                },
            },
        )
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time.__round__(2)} sec")

    except Exception as e:
        logger.error(f"Error in {settings.model.name} Federated Scenario, processing: {str(e)}")
        raise
