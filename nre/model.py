# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from layers.encoder import Encoder
from layers.selector import Selector

FLAGS = tf.app.flags.FLAGS

class Model(object):
    def __init__(self, is_train = True):

        self.is_train = is_train
        self.encoder = Encoder(FLAGS.encoder)
        self.selector = Selector(FLAGS.attention)

        # Placeholders for input
        # each time input 'batch_size' bags, every bag has several sentences with length of num_steps
        # so we assume the sum of sentences in bags is 'total_sentences'
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='input_pos2')
        self.input_type = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='input_type')
        self.input_lens = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_lens')
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.num_steps], name='input_mask')
        self.input_scope = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size+1], name='input_scope')
        self.label = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
        self.label_for_select = tf.placeholder(dtype=tf.int32, shape=[None], name='label_for_select')
        self.input_weights = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size], name='input_weights')
        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep_prob')

    def __assign_word2vec__(self, wordvec):
        self.word_vec = wordvec

    def build_model(self):
        self.global_step = tf.Variable(0, trainable=False)

        """embedding"""
        embedding_output = self.embedding_layer(use_type=FLAGS.use_type)

        """encoding"""
        encoder_out = self.encoder_layer(encoder_input=embedding_output)

        """attention"""
        self.logits, _ = self.attention_layer(encoder_out=encoder_out)

        """loss"""
        self.loss_layer(project_logits=self.logits)

        """model initilization"""
        with tf.variable_scope("optimizer"):
            optimizer = FLAGS.optimizer
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(FLAGS.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(FLAGS.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(FLAGS.lr)
            else:
                raise KeyError

            # Define training procedure
            grads_vars = self.opt.compute_gradients(self.final_loss)
            """
            capped_grads_vars = [[tf.clip_by_value(g, -FLAGS.clip, FLAGS.clip), v]
                                for g, v in grads_vars]
            """
            self.train_op = self.opt.apply_gradients(grads_vars, self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    def embedding_layer(self, use_type):
        """
        :return: [total_sentences, num_steps, emb_size]
        """
        with tf.name_scope('embedding'):
            temp_word_embedding = tf.get_variable(initializer=self.word_vec, name='temp_word_embedding', dtype=tf.float32)
            unk_word_embedding = tf.get_variable('unk_embedding', [FLAGS.embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.word_embedding = tf.concat(
                [temp_word_embedding,
                tf.reshape(unk_word_embedding, [1, FLAGS.embedding_dim]),
                tf.reshape(tf.constant(np.zeros([FLAGS.embedding_dim], dtype=np.float32)), [1, FLAGS.embedding_dim])],
                axis=0
            )
            self.pos1_embedding = tf.concat(
                [tf.get_variable('pos1_embedding', shape=[FLAGS.pos_num, FLAGS.pos_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)), [1, FLAGS.pos_size])],
                axis=0)
            self.pos2_embedding = tf.concat(
                [tf.get_variable('pos2_embedding', shape=[FLAGS.pos_num, FLAGS.pos_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()),
                tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)), [1, FLAGS.pos_size])],
                axis=0)

            if use_type:
                self.type_embedding = tf.get_variable('type_embedding', [FLAGS.type_num, FLAGS.type_size])

                embedded_chars = tf.concat(
                    [tf.nn.embedding_lookup(self.word_embedding,self.input_word),
                    tf.nn.embedding_lookup(self.pos1_embedding,self.input_pos1),
                    tf.nn.embedding_lookup(self.pos2_embedding,self.input_pos2),
                    tf.nn.embedding_lookup(self.type_embedding,self.input_type)],
                    2)
            else:
                embedded_chars = tf.concat(
                    [tf.nn.embedding_lookup(self.word_embedding,self.input_word),
                    tf.nn.embedding_lookup(self.pos1_embedding,self.input_pos1),
                    tf.nn.embedding_lookup(self.pos2_embedding,self.input_pos2)],
                    2)

        return embedded_chars

    def encoder_layer(self, encoder_input):
        """
        Encode instances
        """
        with tf.name_scope('encoder'):
            return self.encoder.encode(self.is_train, self.input_mask, encoder_input)
                
    def attention_layer(self, encoder_out):
        """
        Attention mechanism for bag level prediction
        """
        with tf.name_scope('attention'):
            return self.selector.select(self.is_train, encoder_out, self.input_scope, self.label_for_select)

    def loss_layer(self, project_logits, name=None):
        """
        Explainatin:
            In training process, the relation of a bag(with bag_id) is one-hot vector. But in testing process, it should be N-hot vector.
            However, we don't care about the losses in testing process, we don't need to use sigmoid_cross_entropy
        """
        with tf.name_scope('loss'):
            onehot_label = tf.one_hot(indices=self.label, depth=FLAGS.classes_num, dtype=tf.int32)
            # losses = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=project_scores, multi_class_labels=onehot_label))
            self.final_loss = tf.losses.softmax_cross_entropy(logits=project_logits, onehot_labels=onehot_label, weights=self.input_weights)

        with tf.name_scope('accuracy'):
            self.prediction = tf.argmax(project_logits, axis=1, name='prediction')
            corrent_predictions = tf.equal(self.prediction, tf.cast(self.label, dtype=tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(corrent_predictions, 'float'), name='accuracy')

    def run_step(self, session, is_train, feed_dict, summary_op=None):
        if is_train:
            global_step, _, summaries, loss, accuracy = session.run(
                [self.global_step, self.train_op, summary_op, self.final_loss, self.accuracy],
                feed_dict
            )
            return global_step, summaries, loss, accuracy
        else:
            scores = session.run(
                [self.logits],
                feed_dict
            )
            return scores
