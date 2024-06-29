import logging
import logging.config
from pathlib import Path
import yaml
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

def initialize_logger(config_path: str = "logging.yaml", logs_dir_name: str = "logs") -> logging.Logger: 
    try: 
        config_path_parent = Path(config_path).parent
        logs_dir = config_path_parent / logs_dir_name
        logs_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path, "rt") as f:
            config = yaml.safe_load(f.read())
 
        config["disable_existing_loggers"] = False 
        logging.config.dictConfig(config)

    except FileNotFoundError:
        logger.warning(f"No logging configuration file found at: {config_path}. Setting logging level to INFO.")
        logging.basicConfig(level=logging.INFO)


def load_env_vars(): 
    load_dotenv(find_dotenv())