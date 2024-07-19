import logging
import sys
import datetime
import os
from pathlib import Path

def set_logs(log_level, name= 'logs.log'):
    _logLevel = {
        'CRITICAL': logging.CRITICAL,
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARNING,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    level = _logLevel.get(log_level.upper(),logging.NOTSET)
    path = Path('.logs')
    path.mkdir(parents=True, exist_ok=True)
    # Configure and create logger
    logging.basicConfig(filename=str(path/name), format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    # Setting the threshold of logger
    logger.setLevel(level)
    return logger

def update_collection_cmd_log(collection_folder):

    log_line = ' '.join([datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), ' '.join(sys.argv)])

    with open(os.path.join(collection_folder, 'collection_cmd_log.txt'), 'a') as fp:

        fp.write(log_line+'\n')

    return log_line
