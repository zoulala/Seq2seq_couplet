import os,re
import numpy as np
import pickle
import random


def batch_generator( en_arrs, zh_arrs, batchsize):
    '''产生训练batch样本'''
    assert len(en_arrs) == len(zh_arrs), 'error: incorrect length english&chinese samples'
    n = len(en_arrs)
    print('samples number:',n)
    samples = [en_arrs[i] + zh_arrs[i] for i in range(n)]

    while True:
        random.shuffle(samples)  # 打乱顺序
        for i in range(0, n, batchsize):
            batch_samples = samples[i:i + batchsize]
            batch_en = []
            batch_en_len = []
            batch_zh = []
            batch_zh_len = []
            batch_zh_label = []
            for sample in batch_samples:
                batch_en.append(sample[0])
                batch_en_len.append(sample[1])
                batch_zh.append(sample[2][:-1])
                batch_zh_len.append(sample[3] - 1)
                batch_zh_label.append(sample[2][1:])
            yield np.array(batch_en), np.array(batch_en_len), np.array(batch_zh), np.array(batch_zh_len), np.array(
                batch_zh_label)

class TextConverter(object):
    def __init__(self, vocab_dir=None, max_vocab=5000 , seq_length = 20):
        self.vocab = []
        if os.path.exists(vocab_dir):
            with open(vocab_dir, 'r', encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    self.vocab.append(line)
        else:
            raise('error: vocabs file not exist')
        if len(self.vocab)>max_vocab:
            self.vocab = self.vocab[:max_vocab]
        self.seq_length = seq_length  # 样本序列最大长度
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_en_to_arr(self, text):
        arr = []
        last_num = len(self.vocab)
        query_len = len(text)
        for word in text:
            arr.append(self.word_to_int(word))

        # padding
        if query_len < self.seq_length:
            arr += [last_num] * (self.seq_length - query_len)
        else:
            arr = arr[:self.seq_length]
            query_len = self.seq_length
        if query_len == 0:
            query_len = 1
        return np.array(arr), np.array(query_len)

    def text_de_to_arr(self, text):
        arr = []
        last_num = len(self.vocab)
        query_len = len(text)
        for word in text:
            arr.append(self.word_to_int(word))

        # padding
        if query_len < self.seq_length+1:
            arr += [last_num] * (self.seq_length+1 - query_len)
        else:
            arr = arr[:self.seq_length+1]
            query_len = self.seq_length
        if query_len == 0:
            query_len = 1
        return np.array(arr), np.array(query_len)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def get_en_arrs(self, file_path):
        arrs_list = []  #
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split()
                arr, arr_len = self.text_en_to_arr(line)
                arrs_list.append([arr, arr_len])
        return arrs_list

    def get_de_arrs(self, file_path):
        arrs_list = []  #
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = ['<s>']+line.split()+['</s>']
                arr, arr_len = self.text_de_to_arr(line)
                arrs_list.append([arr, arr_len])
        return arrs_list




if __name__ == '__main__':
    pass
    # loadConversations('data/xiaohuangji50w_fenciA.conv')


