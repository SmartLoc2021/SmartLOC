import logging
import time
import os
import socket
from multiprocessing import Process, Lock


def set_up_log(args, sys_argv):
    log_dir = args.log_dir    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    file_path = os.path.join(log_dir, '{}.log'.format(str(time.time())))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger


def save_performance_result(args, logger, metrics, tag = 'GNN'):
    '''
    Docs:
    Save result and parameters.

    time, tag('GNN'/'LOC'), version_name, model_name, seed, early stop epoch, best test data result (in val data), best test data result (in test data), log name, hostname
    '''
    summary_file = args.summary_file
    if summary_file != 'test':
        summary_file = os.path.join(args.log_dir, summary_file)
    else:
        return
    # dataset = args.dataset
    epoch, val_metric, no_val_metric = metrics
    model_name = '-'.join([args.model, args.feature, str(args.prop_depth)])
    seed = args.seed
    log_name = os.path.split(logger.handlers[1].baseFilename)[-1]
    server = socket.gethostname()
    line = '\t'.join([str(time.asctime( time.localtime(time.time()))), tag, args.version_name, model_name, str(seed), str(epoch), str(round(val_metric, 4)), str(round(no_val_metric, 4)), log_name, server]) + '\n'
    with open(summary_file, 'a') as f:
        f.write(line)  # WARNING: process unsafe!





