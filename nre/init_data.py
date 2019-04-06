from __future__ import print_function
import numpy as np
import os
import codecs
import json
from utils import ensure_dir, load_config

class DataInitializer(object):
    def __init__(self, origin_path, processed_path):
        # txt files directory, default: data/origin_data/
        self.origin_path = origin_path
        # npy files directory, default: data/processed/
        self.processed_path = processed_path

        config = load_config(self.origin_path+'config.json')

        self.sentence_length = config['num_steps']
        self.position_length = config['pos_num']

    def get_Binary_Data(self):
        """
        Transform dataset from txt to binary data
        """

        ensure_dir(self.processed_path)

        self.word2id_f()
        self.relation2id_f()
        self.type2id_f()
        self.gather_bag('train.txt')
        self.gather_bag('test.txt')

        instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_type, train_mask = self.read_train_files('bag_train.txt')
        self.save_array('train_instance_triple', instance_triple)
        self.save_array('train_instance_scope', instance_scope)
        self.save_array('train_len', train_len)
        self.save_array('train_label', train_label)
        self.save_array('train_word', train_word)
        self.save_array('train_pos1', train_pos1)
        self.save_array('train_pos2', train_pos2)
        self.save_array('train_type', train_type)
        self.save_array('train_mask', train_mask)
        print('Length of train_instance_scope: {}'.format(len(instance_scope)))

        instance_entity, instance_entity_no_bag, instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_type, test_mask = self.read_test_files('bag_test.txt')
        self.save_array('test_instance_entity', instance_entity)
        self.save_array('test_instance_entity_no_bag', instance_entity_no_bag)
        self.save_array('test_instance_triple', instance_triple)
        self.save_array('test_instance_scope', instance_scope)
        self.save_array('test_len', test_len)
        self.save_array('test_label', test_label)
        self.save_array('test_word', test_word)
        self.save_array('test_pos1', test_pos1)
        self.save_array('test_pos2', test_pos2)
        self.save_array('test_type', test_type)
        self.save_array('test_mask', test_mask)
        print('Length of test_instance_scope: {}'.format(len(instance_scope)))

        # Remove cached files
        bag_train_file = '{}bag_train.txt'.format(self.origin_path)
        bag_test_file = '{}bag_test.txt'.format(self.origin_path)
        if os.path.isfile(bag_train_file):
            os.remove(bag_train_file)
        if os.path.isfile(bag_test_file):
            os.remove(bag_test_file)

    def word2id_f(self):
        print('reading word embedding data...')
        # a map, key is word, value is its id
        word2id = {}

        with codecs.open(self.origin_path+'vec.txt', 'r', encoding='utf-8') as f:
            total, size = f.readline().strip().split()[:2]
            total = (int)(total)
            size = (int)(size)
            vec = np.ones((total, size), dtype=np.float32)

            for i in range(total):
                content = f.readline()
                content = content.strip().split()
                word2id[content[0]] = len(word2id)
                for j in range(size):
                    vec[i][j] = np.float32(content[j+1])

            word2id['UNK'] = len(word2id)
            word2id['BLANK'] = len(word2id)

            np.save(self.processed_path+'vec.npy',vec)
        
        self.word2id = word2id

    def relation2id_f(self):
        """
        Read relation to id
        """
        print('reading relation to id')
        relation2id = {}

        with codecs.open(self.origin_path+'relation2id.txt','r', encoding='utf-8') as f:
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                relation2id[content[0]] = int(content[1])

        self.relation2id = relation2id

    def type2id_f(self):
        """
        Read type to id
        """
        print("reading type to id")
        type2id = {}

        with codecs.open(self.origin_path+'type2id.txt', 'r', encoding='utf-8') as f:
            while True:
                content = f.readline()
                if content == '':
                    break
                content = content.strip().split()
                type2id[content[0]] = int(content[1])

        self.type2id = type2id

    def gather_bag(self, file):
        """
        Gather instances into bags
        bag_id: (en1_id, en2_id, relation)
        Hence a bag only contains one relation
        """
        bags = {}
        total = 0

        with codecs.open(self.origin_path+file, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                total += 1

                content = line.strip().split()
                en1_id = content[0]
                en2_id = content[1]
                relation = content[6]
                if relation in self.relation2id:
                    relation = self.relation2id[relation]
                else:
                    relation = self.relation2id['NA']
                bag_id = '{}#{}#{}'.format(en1_id, en2_id, relation)

                if bag_id not in bags:
                    bags[bag_id] = []
                bags[bag_id].append(line)

        with codecs.open(self.origin_path+'bag_'+file, 'w', encoding='utf-8') as r:
            r.write('%d\n'%(total))
            for bag_id in bags:
                for instance in bags[bag_id]:
                    r.write(instance)

    def read_train_files(self, file):
        print('reading train data ...')

        f = codecs.open(self.origin_path+file, 'r', encoding='utf-8')

        total = (int)(f.readline().strip())

        # length of sentence is 70
        fixlen = self.sentence_length
        # max length of position embedding is self.position_length (-60~+60)
        # maxlen = self.position_length

        word2id = self.word2id
        relation2id = self.relation2id
        type2id = self.type2id

        sen_word = np.zeros((total, fixlen), dtype=np.int32)
        sen_pos1 = np.zeros((total, fixlen), dtype=np.int32)
        sen_pos2 = np.zeros((total, fixlen), dtype=np.int32)
        sen_mask = np.zeros((total, fixlen), dtype=np.int32)
        sen_type = np.zeros((total, fixlen), dtype=np.int32)
        # Actual length of sentence
        sen_len = np.zeros((total), dtype=np.int32)
        # Relation id
        sen_label = np.zeros((total), dtype=np.int32)
        # Bag scope
        instance_scope = []
        # Bag triplet
        instance_triple = []

        for s in range(total):
            content = f.readline()
            content = content.strip().split()

            relation = 0
            if content[6] not in relation2id:
                relation = relation2id['NA']
            else:
                relation = relation2id[content[6]]

            sentence = content[7:-1]
            # entits' positions
            en1 = content[2]
            en2 = content[3]
            en1pos = 0
            en2pos = 0
            
            for i in range(len(sentence)):
                if sentence[i] == en1:
                    en1pos = i
                if sentence[i] == en2:
                    en2pos = i

            for i in range(fixlen):
                sen_word[s][i] = word2id['BLANK']
                sen_pos1[s][i] = self.pos_embed(i - en1pos)
                sen_pos2[s][i] = self.pos_embed(i - en2pos)
                mask = 0
                if i >= len(sentence):
                    mask = 0
                elif i - en1pos <= 0:
                    mask = 1
                elif i - en2pos <= 0:
                    mask = 2
                else:
                    mask = 3
                sen_mask[s][i] = mask
                sen_type[s][i] = type2id['NA']

            for i in range(min(fixlen,len(sentence))):
                word = 0
                if sentence[i] not in word2id:
                    word = word2id['UNK']
                else:
                    word = word2id[sentence[i]]
                
                sen_word[s][i] = word

            # entity type
            if en1pos < fixlen:
                sen_type[s][en1pos] = type2id[content[4]]
            if en2pos < fixlen:
                sen_type[s][en2pos] = type2id[content[5]]
            
            sen_len[s] = min(fixlen, len(sentence))
            sen_label[s] = relation

            # tup is bag_id
            tup = (content[0], content[1], relation)
            # If a new bag start
            if len(instance_triple) == 0 or instance_triple[len(instance_triple)-1] != tup:
                instance_triple.append(tup)
                instance_scope.append([s, s])
            # Update the scope
            instance_scope[len(instance_triple)-1][1] = s

        f.close()

        # so the length of instance_scope equals that of instance_triple
        # scopes of bags(en1_id, en2_id, relation)
        instance_scope = np.array(instance_scope)
        # list of bag_id
        instance_triple = np.array(instance_triple)
        return instance_triple, instance_scope, sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_type, sen_mask

    def read_test_files(self, file):
        print('reading test data ...')

        f = codecs.open(self.origin_path+file, 'r', encoding='utf-8')

        total = (int)(f.readline().strip())

        # length of sentence is 70
        fixlen = self.sentence_length
        # max length of position embedding is self.position_length (-60~+60)
        # maxlen = self.position_length

        word2id = self.word2id
        relation2id = self.relation2id
        type2id = self.type2id

        sen_word = np.zeros((total, fixlen), dtype=np.int32)
        sen_pos1 = np.zeros((total, fixlen), dtype=np.int32)
        sen_pos2 = np.zeros((total, fixlen), dtype=np.int32)
        sen_mask = np.zeros((total, fixlen), dtype=np.int32)
        sen_type = np.zeros((total, fixlen), dtype=np.int32)
        # Actual length of sentence
        sen_len = np.zeros((total), dtype=np.int32)
        # Relation id
        sen_label = np.zeros((total), dtype=np.int32)
        # Bag scope
        instance_scope = []
        # Bag triplet
        instance_triple = []
        instance_scope_with_NA = []
        instance_entity = []
        instance_entity_no_bag = []

        instances = []
        for _ in range(total):
            content = f.readline()
            content = content.strip().split()
            en1 = content[2]
            en2 = content[3]
            en1_id = content[0]
            en2_id = content[1]
            en1_type = content[4]
            en2_type = content[5]
            sentence = content[7:-1]
            relation = 0
            if content[6] not in relation2id:
                relation = relation2id['NA']
            else:
                relation = relation2id[content[6]]

            tup = (str(en1_id)+'\t'+str(en2_id)+'\t'+str(relation), sentence, en1_id, en2_id, en1, en2, en1_type, en2_type, relation)
            instances.append(tup)
        # sorted by bag_id
        instances = sorted(instances, key=lambda x:x[0])

        for s in range(total):
            #unique_id, sentence, en1_id, en2_id, en1_name, en2_name, relation = instances[s]
            _, sentence, en1_id, en2_id, en1_name, en2_name, en1_type, en2_type, relation = instances[s]

            en1pos = 0
            en2pos = 0
            
            for i in range(len(sentence)):
                if sentence[i] == en1_name:
                    en1pos = i
                if sentence[i] == en2_name:
                    en2pos = i

            for i in range(fixlen):
                sen_word[s][i] = word2id['BLANK']
                sen_pos1[s][i] = self.pos_embed(i - en1pos)
                sen_pos2[s][i] = self.pos_embed(i - en2pos)
                mask = 0
                if i >= len(sentence):
                    mask = 0
                elif i - en1pos <= 0:
                    mask = 1
                elif i - en2pos <= 0:
                    mask = 2
                else:
                    mask = 3
                sen_mask[s][i] = mask
                sen_type[s][i] = type2id['NA']

            for i in range(min(fixlen,len(sentence))):
                word = 0
                if sentence[i] not in word2id:
                    word = word2id['UNK']
                else:
                    word = word2id[sentence[i]]

                sen_word[s][i] = word

            # entity type
            if en1pos < fixlen:
                sen_type[s][en1pos] = type2id[en1_type]
            if en2pos < fixlen:
                sen_type[s][en2pos] = type2id[en2_type]

            sen_len[s] = min(fixlen, len(sentence))
            sen_label[s] = relation

            tup = (en1_id, en2_id, relation)
            # tup[:2] = (en1_id, en2_id)
            instance_entity_no_bag.append(tup[:2])

            # instance_scope_with_NA contains bags with relation == 0
            # if len(instance_scope_with_NA) == 0 or the last element of it != tup. It means a new bag begins
            if len(instance_scope_with_NA) == 0 or instance_scope_with_NA[len(instance_scope_with_NA)-1] != tup:

                # It means a new entity pair begins
                if len(instance_scope_with_NA) == 0 or instance_scope_with_NA[len(instance_scope_with_NA)-1][:2] != tup[:2]:
                    instance_scope.append([s, s])
                    instance_entity.append(tup[:2])

                instance_scope_with_NA.append(tup)

                # Add triple without relation of NA
                if tup[2] != 0:
                    instance_triple.append(tup)
            
            # When begin a new entity pairs, instance_scope adds a new elements, otherwise only updates the scope
            # So it represents the scope of entity pairs. Hence within a scope there exists mutiple relations
            instance_scope[len(instance_scope)-1][1] = s
        
        f.close()

        # so the length of instance_scope doesn't equals to that of instance_triple

        # entity pairs
        instance_entity = np.array(instance_entity)
        instance_entity_no_bag = np.array(instance_entity_no_bag)
        # triples without relation of NA
        instance_triple = np.array(instance_triple)
        # scopes of entity pairs
        instance_scope = np.array(instance_scope)

        return instance_entity, instance_entity_no_bag, instance_triple, instance_scope, sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_type, sen_mask

    def pos_embed(self, x):
        """
        get position embedding of x
        """
        return max(0, min(x + self.position_length, 2*self.position_length + 1))

    def find_index(self, x,y):
        """
        find the index of x in y, if x not in y, return -1
        """
        for index, item in enumerate(y):
            if x == item:
                return index
            return -1

    def save_array(self, file_name, data):
        file = '{}{}.npy'.format(self.processed_path, file_name)
        np.save(file, data)

    def save_config(self, config_file, config_dict):
        with codecs.open(config_file, 'w', encoding='utf-8') as f:
            json_str = json.dumps(config_dict, indent=4)
            f.write(json_str)

if __name__ == '__main__':
    initializer = DataInitializer('data/origin_data/', 'data/processed/')
    initializer.get_Binary_Data()
