import logging
import os

def set_logger(name, out_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.DEBUG)
    ## don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ## file
    handler = logging.FileHandler(os.path.join(out_dir, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    ## console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger