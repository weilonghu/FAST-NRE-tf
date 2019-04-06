# -*- coding:utf-8 -*-
import  tensorflow as tf
import os
from collections import OrderedDict

from init_data import DataInitializer
from dataset import DataManager
from utils import create_model, init_logger, add_log, save_model, clean_model, load_config, ensure_dir
from model import Model
from engine import Engine

config = load_config('data/origin_data/config.json')

flags = tf.app.flags
flags.DEFINE_boolean('clean', True, 'Clean train folder')
flags.DEFINE_boolean('is_train', True, 'Whether train the model or evaluate')
flags.DEFINE_boolean('preprocess', False, 'Preprocess the data')

flags.DEFINE_integer('embedding_dim', config['embedding_dim'], 'Embedding size of words')
flags.DEFINE_integer('vocabulary_size', config['vocabulary_size'], 'Vocabulary size of corpus')
flags.DEFINE_float('dropout_keep', 0.5, 'Dropout rate')
flags.DEFINE_integer('batch_size', 160, 'Batch size')
flags.DEFINE_float('lr', 0.5, 'Learning rate')
flags.DEFINE_string('optimizer', 'sgd', 'Optimizer for training')
flags.DEFINE_integer('num_epochs', 30, 'Maximum training epochs')
flags.DEFINE_integer('steps_check', 5, 'Steps per checkpoint')
flags.DEFINE_integer('save_epoch', 2, 'Save model after n epochs')
flags.DEFINE_string('epoch_range',  '(3,  30)', 'checkpoint epoch range', )

flags.DEFINE_integer('pos_num', config['pos_num']*2 + 1, 'The number of all position')
flags.DEFINE_integer('pos_size', 5, 'Position embedding size')
flags.DEFINE_integer('type_num', 4, 'The number of all types')
flags.DEFINE_integer('type_size', 5, 'Type embedding size')
flags.DEFINE_integer('classes_num', 53, 'The number of all relation classes')
flags.DEFINE_integer('filter_width', 3, 'Windows size of filter')
flags.DEFINE_integer('filter_num', 230, 'The number of filters')
flags.DEFINE_integer('repeat_times', 1, 'Repeat times')
flags.DEFINE_boolean('use_type', False, 'Whether use entity types')
flags.DEFINE_integer('num_steps', config['num_steps'], 'Length of sentences')

flags.DEFINE_string('encoder', 'cnn1', 'Encoder type for extraction')
flags.DEFINE_string('attention', 'att', 'Attention type for selection')

flags.DEFINE_string('models_dir', 'models', 'Directory for storing models')
flags.DEFINE_string('ckpt_path', 'checkpoints', 'Path to save checkpoints')
flags.DEFINE_string('summary_path', 'summary', 'Path to store summaries')
flags.DEFINE_string('log_file', 'train.log', 'File for logging')
flags.DEFINE_string('res_path', 'res', 'Path to save result')
flags.DEFINE_string('pretrain_model', 'None', 'Pretrain model prefix')

#Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS
assert 0 <= FLAGS.dropout_keep < 1, 'dropout rate between 0 and 1'
assert FLAGS.lr > 0, 'learning rate must larger than zero'
assert FLAGS.optimizer in ['adam', 'sgd', 'adagrad']


def preprocess():
    data_initializer = DataInitializer('data/origin_data/', 'data/processed/')
    data_initializer.get_Binary_Data()


def main(_):
    ensure_dir(FLAGS.models_dir)

    if FLAGS.is_train and FLAGS.clean:
        clean_model()

    init_logger(os.path.join(FLAGS.models_dir, FLAGS.log_file))

    if FLAGS.preprocess:
        preprocess()

    data_manager = DataManager()

    out_dir = os.path.join('.', FLAGS.models_dir)
    checkpoint_dir = os.path.join(out_dir, FLAGS.ckpt_path)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    ensure_dir(checkpoint_dir)

    gpu_options = tf.GPUOptions(allow_growth=True)
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options)
    
    with tf.Session(config=session_conf) as sess:
        wordembedding = data_manager.get_wordembedding()
        model = create_model(sess, Model, FLAGS.is_train, FLAGS.pretrain_model, wordembedding)
        model_engine = Engine(data_manager)

        if FLAGS.is_train:
            add_log('\nstart training...\n')
            summary_op, summary_writer = summary(out_dir, sess, model)
            model_engine.start_train(model, sess, summary_op, summary_writer, checkpoint_prefix)
            #model_engine.start_train(model, sess, summary_op, summary_writer)

        else:
            add_log('\nstart evaluating...\n')
            #reload_model(sess, model, checkpoint_dir)
            model_engine.start_test(model, sess, checkpoint_prefix)
            #model_engine.start_test(model, sess)


def summary(out_dir, sess, model):
    tf.summary.scalar('loss', model.final_loss)
    tf.summary.scalar('accuracy', model.accuracy)

    # Train summaries
    train_summary_op = tf.summary.merge_all()
    train_summary_dir = os.path.join(out_dir, FLAGS.summary_path)
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    return train_summary_op, train_summary_writer


if __name__ == '__main__':
    tf.app.run(main)
