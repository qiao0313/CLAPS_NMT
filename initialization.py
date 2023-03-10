import logging
import sys
import random
import numpy as np
import torch


def set_seed(random_seed=999):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)


def set_device(deviceId):
    if deviceId < 0:
        device = torch.device("cpu")
    else:
        assert torch.cuda.device_count() >= deviceId + 1
        device = torch.device("cuda:%d" % (deviceId))
        torch.backends.cudnn.enabled = False
    return device


def set_logger(exp_path, testing=False):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s')  # ('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    if testing:
        fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
    else:
        fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger

