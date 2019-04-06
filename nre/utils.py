# -*- coding:utf-8 -*-
import logging
import numpy as np
import tensorflow as tf
import random
import os
import shutil
import codecs
import json
import sys

def init_logger(log_file):
    logger = logging.getLogger('FAST-NRE')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def process_info(info):
    sys.stdout.write(info + '\r')
    sys.stdout.flush()


def add_log(message):
    logger = logging.getLogger('RE-IDCNN')
    logger.info(message)
    

def load_config(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('\r\n', '')
        config = json.loads(content)
        return config


def ensure_dir(dir_path):
    """
    If dir_path not exists, make this directory
    """
    abs_path = os.path.abspath(dir_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)


def clean_model():
    """
    Remove saved model and training log
    """
    if os.path.isfile("log/train.log"):
        os.remove('log/train.log')

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isdir('runs'):
        shutil.rmtree('runs')


def create_model(session, Model_class, is_train, pretrain_model, word2vec):
    model = Model_class(is_train)
    model.__assign_word2vec__(word2vec)
    model.build_model()

    # Reuse parameters if exists
    if pretrain_model != 'None':
        # model.saver.restore(session, checkpoint_prefix)
        add_log('restore model from {}'.format(pretrain_model))
        model.saver.restore(session, pretrain_model)
    else:
        session.run(tf.global_variables_initializer())
        # session.run(model.word_embedding.assign(word2vec))
    return model


def reload_model(session, model, checkpoint_prefix, step):
    ckpt = checkpoint_prefix + '-' + str(step)
    add_log('reload model from {}'.format(ckpt))
    model.saver.restore(session, ckpt)


def save_model(session, model, checkpoint_prefix, step):
    return model.saver.save(session, checkpoint_prefix, global_step=step)
