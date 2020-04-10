import logging

def set_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("output/log.txt") ## file
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler() ## console
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger