import logging, os
from torch.utils.tensorboard import SummaryWriter

def get_logger(name: str = "ddpmppo"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(h)
    return logger

def get_tb_writer(log_dir: str = "runs"):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)
