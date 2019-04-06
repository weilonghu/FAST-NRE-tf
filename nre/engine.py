# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import sklearn.metrics
from utils import add_log, save_model, ensure_dir, process_info, reload_model

FLAGS = tf.app.flags.FLAGS

class Engine(object):
    def __init__(self, data_manager):
        self.train_set = data_manager.get_train_data()
        self.test_set = data_manager.get_test_data()

    def start_train(self, model, session, summary_op, summary_writer, checkpoint_prefix):
        for epoch in range(FLAGS.num_epochs):
            for batch in self.iter_batch(self.train_set, True):
                feed_dict = self.create_feed_dict(model, batch, True)
                step, summaries, loss, accuracy = model.run_step(session, True, feed_dict, summary_op)
                summary_writer.add_summary(summaries, step)

                if step % FLAGS.steps_check == 0:
                    log_msg = 'epoch: {}/{}, step: {}, acc: {:g}, loss: {:g}'.format(epoch+1, FLAGS.num_epochs, step, accuracy, loss)
                    process_info(log_msg)

            if (epoch+1) % FLAGS.save_epoch == 0:
                path = save_model(session, model, checkpoint_prefix, epoch)
                log_msg = 'epoch {} is over. Saved model checkpoint to {}\n'.format(epoch+1, path)
                add_log(log_msg)

    def start_test(self, model, session, checkpoint_prefix):
        epoch_range = eval(FLAGS.epoch_range)
        epoch_range = range(epoch_range[0], epoch_range[1])
        save_x = None
        save_y = None
        best_auc = 0
        best_epoch = -1

        for epoch in epoch_range:
            if not os.path.exists('{}-{}.index'.format(checkpoint_prefix, epoch)):
                continue

            add_log('start testing model-{}'.format(epoch))
            reload_model(session, model, checkpoint_prefix, epoch)

            test_result = []
            total_recall = 0

            for i, batch in enumerate(self.iter_batch(self.test_set, False)):
                feed_dict = self.create_feed_dict(model, batch, False)
                test_output = model.run_step(session, False, feed_dict)[0]

                for j in range(len(test_output)):
                    pred = test_output[j]
                    entity_pair = self.test_set.instance_entity[j + batch.start_index]
                    for rel in range(1, len(pred)):
                        flag = int(((entity_pair[0], entity_pair[1], rel) in self.test_set.instance_triple))
                        total_recall += flag
                        test_result.append([(entity_pair[0], entity_pair[1], rel), flag, pred[rel]])

                if i % 100 == 0:
                    process_info('predicting {} / {}\r'.format(i, batch.num_batches))

            add_log('\nevaluating...')

            sorted_test_result = sorted(test_result, key=lambda x: x[2])
            # Reference url: https://blog.csdn.net/zk_j1994/article/details/78478502
            pr_result_x = [] # recall
            pr_result_y = [] # precision
            correct = 0
            for i, item in enumerate(sorted_test_result[::-1]):
                if item[1] == 1: # flag == 1
                    correct += 1
                pr_result_y.append(float(correct) / (i+1))
                pr_result_x.append(float(correct) / total_recall)

            auc = sklearn.metrics.auc(x=pr_result_x, y=pr_result_y)
            prec_mean = (pr_result_y[100] + pr_result_y[200] + pr_result_y[300]) / 3

            add_log('auc: {:g}\np@100: {:g}\np@200: {:g}\np@300: {:g}\np@(100,200,300) mean: {:g}\n'.format(
                auc, pr_result_y[100], pr_result_y[200], pr_result_y[300], prec_mean))

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                save_x = pr_result_x
                save_y = pr_result_y

        ensure_dir(os.path.join(FLAGS.models_dir, FLAGS.res_path))

        new_res_path = os.path.join(FLAGS.models_dir, FLAGS.res_path)
        np.save(os.path.join(new_res_path, 'model_x.npy'), save_x)
        np.save(os.path.join(new_res_path, 'model_y.npy'), save_y)
        add_log('best_auc: {:g}; best_epoch: {}'.format(best_auc, best_epoch))

    def create_feed_dict(self, model, batch, is_train):
        """
        Args:
            model: model with placeholders
            batch: returned by __generate_batch
        """
        feed_dict = {
            model.input_word: batch.word,
            model.input_pos1: batch.pos1,
            model.input_pos2: batch.pos2,
            model.input_type: batch.type,
            model.input_lens: batch.lens,
            model.input_mask: batch.mask, 
            model.input_scope: batch.scope,
            model.label:  batch.label,
            model.label_for_select: batch.label_for_select
        }

        if is_train:
            feed_dict[model.input_weights] = batch.weights
            feed_dict[model.dropout_keep] = FLAGS.dropout_keep
        else:
            feed_dict[model.dropout_keep] = 1.0
        
        return feed_dict


    def __generate_batch(self, indexs, scope, label, label_for_select, weights, start_index, num_batches, dataset):
        """
        A single training step
        """
        word_batch = dataset.sen_word[indexs, :]
        pos1_batch = dataset.sen_pos1[indexs, :]
        pos2_batch = dataset.sen_pos2[indexs, :]
        type_batch = dataset.sen_type[indexs, :]
        mask_batch = dataset.sen_mask[indexs, :]
        lens_batch = dataset.sen_len[indexs]
        
        # Ten elements
        batch = Batch(
            word = word_batch,
            pos1 = pos1_batch,
            pos2 = pos2_batch,
            type = type_batch,
            mask = mask_batch,
            lens = lens_batch,
            label = label,
            label_for_select = label_for_select,
            scope = np.array(scope),
            weights = weights,
            start_index = start_index,
            num_batches = num_batches
        )
        return batch


    def iter_batch(self, dataset, is_train):
        data_size = len(dataset.instance_scope)

        num_batches_per_epoch = int(data_size/FLAGS.batch_size)

        if is_train:
            # Randomly shuffle data
            shuffle_indices = np.random.permutation(np.arange(data_size))
        else:
            shuffle_indices = np.arange(data_size)

        for batch_num  in range(num_batches_per_epoch):
            start_index = batch_num * FLAGS.batch_size
            end_index = (batch_num+1) * FLAGS.batch_size

            # Get bags according to the number of batch_size
            input_scope = np.take(dataset.instance_scope, shuffle_indices[start_index:end_index], axis=0)
            index = []
            scope = [0]
            weights = []
            label = []

            for num in input_scope:
                # A list contains all index of instances in batch_size bags
                index = index + list(range(num[0], num[1]+1))
                label.append(dataset.sen_label[num[0]])
                # A list contains start_index of each bag
                scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
                if is_train:
                    weights.append(dataset.reltot[dataset.sen_label[num[0]]])

            yield self.__generate_batch(index, scope, label, dataset.sen_label[index], weights, start_index, num_batches_per_epoch, dataset)


class Batch(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
