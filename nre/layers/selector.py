import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Selector(object):
    def __init__(self, selector_type):
        self.selectors = {
            'avg': self.avg_attention,
            'max': None,
            'att': self.attention,
        }

        self.selected_selector = self.selectors[selector_type]

    def select(self, is_train, encoder_out, input_scope, input_label):
        return self.selected_selector(is_train, encoder_out, input_scope, input_label)

    def avg_attention(self, is_train, encoder_out, input_scope, input_label, name=None):
        """
        param: idcnn_out: [total_sentences, encoder_output_width]
        return: [batch_size, encoder_output_width]
        """

        tower_repre = []
        for i in range(FLAGS.batch_size):
            # size = self.input_scope[i+1] - self.input_scope[i]
            sen_matrix = encoder_out[input_scope[i]:input_scope[i+1]]
            final_repre = tf.reduce_mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        
        attention_out = tf.reshape(tower_repre, [FLAGS.batch_size, -1])
        logits = self.__project_logits__(attention_out, 'average_logits', False)
        if not is_train:
            logits = tf.nn.softmax(logits)
        return logits, attention_out

    def attention(self, is_train, encoder_out, input_scope, input_label):
        if is_train:
            attention_logit = self.__attention_train_logits__(encoder_out, input_label, 'attention_logits', False) 
            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = encoder_out[input_scope[i] : input_scope[i+1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[input_scope[i]: input_scope[i+1]], [1, -1]))
                final_repre = tf.squeeze(tf.matmul(attention_score, sen_matrix))
                tower_repre.append(final_repre)

            stack_repre = tf.stack(tower_repre)
            return self.__project_logits__(stack_repre, 'attention_logits'), stack_repre
        
        else:
            test_attention_logit = self.__attention_test_logits__(encoder_out, 'attention_logits', False)
            test_tower_output = []
            test_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = encoder_out[input_scope[i] : input_scope[i+1]]
                test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[input_scope[i]:input_scope[i+1], :]))
                final_repre = tf.matmul(test_attention_score, sen_matrix)
                logits = self.__project_logits__(final_repre, 'attention_logits')
                test_repre.append(final_repre)
                test_tower_output.append(tf.diag_part(tf.nn.softmax(logits)))

            test_repre = tf.reshape(tf.stack(test_repre), [input_scope.shape[0]-1, FLAGS.classes_num, -1])
            test_output = tf.reshape(tf.stack(test_tower_output), [input_scope[0]-1, FLAGS.classes_num])
            return test_output, test_repre

    def __attention_train_logits__(self, x, query, var_scope = None, reuse = None):
        with tf.variable_scope(var_scope or 'attention_logits', reuse = reuse):
            relation_matrix = tf.get_variable('relation_matrix', [FLAGS.classes_num, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # bias = tf.get_variable('bias', [FLAGS.classes_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            current_attention = tf.nn.embedding_lookup(relation_matrix, query)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
        return attention_logit

    def __attention_test_logits__(self, x, var_scope = None, reuse = None):
        with tf.variable_scope(var_scope or 'attention_logits', reuse = reuse):
            relation_matrix = tf.get_variable('relation_matrix', [FLAGS.classes_num, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            # bias = tf.get_variable('bias', [FLAGS.classes_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        return tf.matmul(x, tf.transpose(relation_matrix))
        
    def __project_logits__(self, attention_out, var_scope=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(var_scope or 'logits', reuse=reuse):
            relation_matrix = tf.get_variable('relation_matrix', [FLAGS.classes_num, attention_out.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', [FLAGS.classes_num], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            logits = tf.matmul(attention_out, tf.transpose(relation_matrix)) + bias

        return logits
