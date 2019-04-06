import numpy as np
import init_data

class Dataset(object):
    def __init__(self, instance_triple, instance_scope, sen_len, sen_word, sen_pos1, sen_pos2, sen_type, sen_mask, sen_label):
        self.instance_triple = instance_triple
        self.instance_scope = instance_scope
        self.sen_len = sen_len
        self.sen_word = sen_word
        self.sen_pos1 = sen_pos1
        self.sen_pos2 = sen_pos2
        self.sen_type = sen_type
        self.sen_mask = sen_mask
        self.sen_label = sen_label

        self.instance_entity = None
        self.instance_entity_no_bag = None
        self.reltot = None

    def set_extral_data(self, instance_entity, instance_entity_no_bag):
        self.instance_entity = instance_entity
        self.instance_entity_no_bag = instance_entity_no_bag

    def set_reltot(self, reltot):
        self.reltot = reltot


class DataManager(object):
    
    def get_wordembedding(self):
        """
        Load wordembedding
        """
        return np.load('data/processed/vec.npy')

    def get_train_data(self):
        """
        Load train data
        """
        train_instance_triple = self.load_data('train_instance_triple')
        train_instance_scope = self.load_data('train_instance_scope')
        train_len = self.load_data('train_len')
        train_word = self.load_data('train_word')
        train_pos1 = self.load_data('train_pos1')
        train_pos2 = self.load_data('train_pos2')
        train_type = self.load_data('train_type')
        train_mask = self.load_data('train_mask')
        train_label = self.load_data('train_label')

        reltot = {}
        for _, i in enumerate(train_label):
            if not i in reltot:
                reltot[i] = 1.0
            else:
                reltot[i] += 1.0
        for i in reltot:
            reltot[i] = 1 / (reltot[i] ** (0.05))

        train_set = Dataset(train_instance_triple, train_instance_scope, train_len, train_word, train_pos1, train_pos2, train_type, train_mask, train_label)
        train_set.set_reltot(reltot)

        return train_set

    def get_test_data(self):
        test_instance_entity = self.load_data('test_instance_entity')
        test_instance_entity_no_bag = self.load_data('test_instance_entity_no_bag')
        instance_triple = self.load_data('test_instance_triple')
        test_instance_triple = set()
        for item in instance_triple:
            tup = (item[0], item[1], int(item[2]))
            test_instance_triple.add(tup)
        test_instance_scope = self.load_data('test_instance_scope')
        test_len = self.load_data('test_len')
        test_word = self.load_data('test_word')
        test_pos1 = self.load_data('test_pos1')
        test_pos2 = self.load_data('test_pos2')
        test_type = self.load_data('test_type')
        test_mask = self.load_data('test_mask')
        test_label = self.load_data('test_label')

        test_set = Dataset(test_instance_triple, test_instance_scope, test_len, test_word, test_pos1, test_pos2, test_type, test_mask, test_label)
        test_set.set_extral_data(test_instance_entity, test_instance_entity_no_bag)

        return test_set

    def load_data(self, data_name):
        file = 'data/processed/{}.npy'.format(data_name)
        return np.load(file)
