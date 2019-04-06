import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Encoder(object):
    def __init__(self, encoder_type):
        self.encoders = {
            'cnn1': self.cnn1_layer,
            'cnn2': self.cnn2_layer,
            'pcnn': self.pcnn_layer,
            'birnn': self.birnn_layer,
        }

        self.selected_encoder = self.encoders[encoder_type]

    def encode(self, is_train, mask, model_inputs, length):
        """Interface for encoding"""
        return self.selected_encoder(is_train, mask, model_inputs, length)

    def cnn1_layer(self, is_train, mask, model_inputs, length, name=None):
        with tf.name_scope('cnn'):
            model_inputs = tf.expand_dims(model_inputs, axis=1)
            model_inputs = tf.layers.conv2d(
                inputs=model_inputs,
                filters = FLAGS.filter_num,
                kernel_size=[1, FLAGS.filter_width],
                strides=[1, 1],
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            
            x = tf.reshape(model_inputs, [-1, FLAGS.num_steps, FLAGS.filter_num])
            x = tf.reduce_max(x, axis=1)
            x = tf.reshape(x, [-1, FLAGS.filter_num])
            x = tf.nn.relu(x)
            return tf.layers.dropout(x, rate=FLAGS.dropout_keep, training=is_train)

    def cnn2_layer(self, is_train, mask, model_inputs, length, name=None):
        with tf.name_scope('CNN'):
            model_inputs = tf.expand_dims(model_inputs, axis=1)
            if FLAGS.use_type:
                embedding_size = FLAGS.embedding_dim + 2*FLAGS.pos_size + FLAGS.type_size
            else:
                embedding_size = FLAGS.embedding_dim + 2*FLAGS.pos_size

            filter_shape = [1, FLAGS.filter_width, embedding_size, FLAGS.filter_num]
            model_inputs = tf.nn.conv2d(
                model_inputs, 
                tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name="cnn_W"),
                strides=[1,1,1,1], 
                padding='SAME', 
                name='conv')
            x = tf.reshape(model_inputs, [-1, FLAGS.num_steps, FLAGS.filter_num])
            x = tf.reduce_max(x, axis=1)
            x = tf.reshape(x, [-1, FLAGS.filter_num])
            x = tf.nn.relu(x)
            return tf.layers.dropout(x, rate=FLAGS.dropout_keep, training=is_train)

    def pcnn_layer(self, is_train, mask, model_inputs, length, name=None):
        with tf.name_scope('pcnn'):
            model_inputs = tf.expand_dims(model_inputs, axis=1)
            # embedding_size = FLAGS.embedding_dim + 2*FLAGS.pos_size
            mask_embedding = tf.constant([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
            mask = tf.nn.embedding_lookup(mask_embedding, mask)
            model_inputs = tf.layers.conv2d(
                inputs=model_inputs,
                filters=FLAGS.filter_num,
                kernel_size=[1, FLAGS.filter_width],
                strides=[1, 1],
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
            )
            x = tf.reshape(model_inputs, [-1, FLAGS.num_steps, FLAGS.filter_num, 1])
            x = tf.reduce_max(
                tf.reshape(mask, [-1, FLAGS.num_steps, 1, 3]) * 100 + x, axis=1
            ) - 100
            x = tf.reshape(x, [-1, FLAGS.filter_num * 3])
            x = tf.nn.relu(x)
            return tf.layers.dropout(x, rate=FLAGS.dropout_keep, training=is_train)

    def birnn_layer(self, is_train, mask, model_inputs, length, name=None):
        with tf.name_scope('birnn'):
            model_inputs = tf.layers.dropout(model_inputs, rate=FLAGS.dropout_keep, training=is_train)
            fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size, state_is_tuple=True)
            bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size, state_is_tuple=True)
            _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sequence_length=length, dtype=tf.float32, scope='dynamic-bi-rnn')
            fw_states, bw_states = states
            return tf.concat([fw_states, bw_states], axis=1)
