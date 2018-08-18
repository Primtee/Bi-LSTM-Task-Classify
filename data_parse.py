import re
import os
import sys
import pickle
import numpy as np
from copy import deepcopy
from random import sample


class DataParse:
    def __init__(self, train_data_path, task2id_id2task_path, sentence_task_path, vocb_path):

        self.train_data_path = train_data_path  # 训练数据路径
        self.task2id_id2task = task2id_id2task_path  # 任务的映射
        self.sentence_task_path = sentence_task_path  # 训练数据
        self.vocb_path = vocb_path  # 字典
        self.stop = ['”', '“', '、', '。', '，', '──', '……', '（', '）', '？', '《', '》', '<', '>',
                     '！', '......', '.', ',', '；', ';', '%']

    @property
    def stopwords(self):
        return self.stop

    def check_p(self, p, p_txt):
        """
        主要是针对正则索引进行分词
        :param p: 正则结果
        :param p_txt: 正则文本
        :return:  分词list
        """
        if len(p) > 0:
            copy_txt = deepcopy(p_txt)
            index_margin = []
            res = ''
            for cur_p in p:
                start_end_index = re.search(cur_p, copy_txt)
                if start_end_index:
                    start_end_index = start_end_index.span()
                    index_margin.append(start_end_index)  # 左开右闭的区间
                    tmp = copy_txt[:copy_txt.index(cur_p) + len(cur_p)].replace(cur_p, ''.join(['*'] * len(cur_p)))
                    copy_txt = tmp.__add__(copy_txt[copy_txt.index(cur_p) + len(cur_p):])
            index_margin = sorted(index_margin)
            if len(index_margin) == 1:
                # '体面2019好么'  （2,5）
                res = list(p_txt[:index_margin[0][0]])
                res.append(p_txt[index_margin[0][0]: index_margin[0][1]])
                res.extend(p_txt[index_margin[0][1]:])
                return res
            else:
                for cur_index in range(0, len(index_margin) - 1, 1):
                    start = index_margin[cur_index][0]
                    end = index_margin[cur_index][1]
                    begin = index_margin[cur_index + 1][0]
                    if len(res) == 0:
                        res = list(p_txt[: start])
                    res.append(p_txt[start: end])
                    res.append(p_txt[end: begin])  # 2015年的8月2日和2016年的8月2日你在哪里那么 2018呢还有 i love you 2019
                res.append(p_txt[index_margin[-1][0]: index_margin[-1][1]])
                res.append(p_txt[index_margin[-1][1]:])
                # # 可能存在重复 去重
                results = []
                for cur_item_index in range(0, len(res), 1):
                    cur_e = re.findall(r'[a-zA-Z]+', res[cur_item_index])
                    cur_n = re.findall(r'[0-9]+', res[cur_item_index])
                    if (len(cur_e) == 0) and (len(cur_n) == 0):
                        results.extend(list(res[cur_item_index]))
                    else:
                        results.append(res[cur_item_index])
                return results
        else:
            return list(p_txt)

    def clean_txt(self, txt):
        for cur_stop_words in self.stopwords:
            if cur_stop_words in txt:
                txt = txt.replace(cur_stop_words, '')
        p = re.findall(r'[0-9]+', txt)
        p_e = re.findall(r'[a-zA-Z]+', txt)
        return self.check_p(p + p_e, txt)

    def read_corpus(self, corpus_path=None, save=True):
        """
            获取语料，路径
        """
        if corpus_path is None:
            assert 'path empty'
        try:
            with open(corpus_path, mode='r', encoding='utf-8') as fr:
                lines = fr.readlines()
            task_sentense = []
            task2id_id2task = {}
            task2id = {}
            id2task = {}
            for line in lines:
                # [char, label] = line.strip().split()
                task, sentence = line.strip().split('\t')[:]
                sentence = self.clean_txt(sentence.strip())
                sentence = list(filter(lambda x: not x.__eq__(' '), sentence))  # 过滤多余空格
                task_sentense.append((sentence, [task]))  # 句子--任务
                if task2id.get(task, 10000) == 10000:
                    task2id[task] = len(task2id)
                    id2task[len(id2task)] = task

            task2id_id2task['task2id'] = task2id
            task2id_id2task['id2task'] = id2task
            print(task2id_id2task)
            if save:
                pickle.dump(task_sentense, open(self.sentence_task_path, mode='wb'))
                pickle.dump(task2id_id2task, open(self.task2id_id2task, mode='wb'))
            return task_sentense, task2id_id2task
        except Exception:
            raise Exception

    def old_creat_vocab(self, data=None, save=True):
        if not os.path.exists(self.vocb_path):
            vocab = {}
            vocab['<UNK>'] = 1  # 未知的词汇
            vocab['<PAD>'] = 0  # 需要被填充的标记
            if data is not None:
                assert isinstance(data, list) and isinstance(data[0], tuple)
                for sentence, _, _ in data:
                    sentence = sentence.split()
                    print('sentence', sentence)
                    for cut_word in sentence:
                        if vocab.get(cut_word, 0) == 0:
                            vocab[cut_word] = len(vocab)
                if save:
                    pickle.dump(vocab, open(self.vocb_path, mode='wb'))
                return vocab
            else:
                print('data empty......')
        else:
            sys.stdout.write('vocab exists......')
            return pickle.load(open(self.vocb_path, mode='rb'))

    def creat_vocab(self, data=None, save=True):
        if not os.path.exists(self.vocb_path):
            vocab = {}  # 创建字典
            vocab['<UNK>'] = 1  # 未知的词汇
            vocab['<PAD>'] = 0  # 需要被填充的标记
            if data is not None:
                # assert isinstance(data, list) and isinstance(data[0], str)
                assert isinstance(data, list) and isinstance(data[0], tuple)
                for task_sentence in data:
                    all_splited = task_sentence[0]  # 得到任务 和 句子
                    print('任务—句子', all_splited)
                    for cut_word in all_splited:
                        if vocab.get(cut_word, -1) == -1:
                            vocab[cut_word] = len(vocab)
                if save:
                    pickle.dump(vocab, open(self.vocb_path, mode='wb'))
                return vocab
            else:
                print('data empty......')
        else:
            sys.stdout.write('vocab exists......')
            return pickle.load(open(self.vocb_path, mode='rb'))

    def random_embedding(self, embedding_dim, word_num):
        """
        随机的生成word的embedding，这里如果有语料充足的话，可以直接使用word2vec蓄念出词向量，这样词之间的区别可能更大。
        :param embedding_dim:  词向量的维度。
        :return: numpy format array. shape is : (vocab, embedding_dim)
        """
        # if vocb_paths is None:
        #     vocab_creatation = pickle.load(open(self.vocb_path, mode='rb'))
        # else:
        #     vocab_creatation = pickle.load(open(vocb_paths, mode='rb'))
        embedding_mat = np.random.uniform(-0.25, 0.25, (word_num, embedding_dim))
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat


class Data_Inter:
    """
    生成训练数据
    """
    def __init__(self, batch_size, task_sentence_path, task2id_id2task, vocb_path):
        self.task_sentence_path = task_sentence_path  # 任务—句子
        self.task2id_id2task = task2id_id2task  # 任务的映射
        self.vocb_path = vocb_path  # 字典路径
        self.batch_size = batch_size  # 批的大小
        self.index = 0
        # self.initializer()
        if os.path.exists(self.task_sentence_path):  # 读取 任务—句子对
            # 格式[['Hei], 'fun', 'lei'], ['O', 'B', 'O'], ['Love']]------> rowtext, slots, intent
            self.task_sentence = np.array(pickle.load(open(self.task_sentence_path, mode='rb')))
            self.end = len(self.task_sentence)
            print('self.end:', self.end)
            self.num_batches = self.end // self.batch_size
            self.shuffle = sample(range(0, self.end, 1), self.end)
        else:
            print('train data is empty......')

        if os.path.exists(self.vocb_path):  # 读取字典
            self.vocab = pickle.load(open(self.vocb_path, mode='rb'))
        else:
            print('vocab is empty......')

        if os.path.exists(self.task2id_id2task):  # 读取任务映射
            self.task2id = pickle.load(open(self.task2id_id2task, mode='rb'))['task2id']
        else:
            print('intent mapping must be provided......')

    def next(self):
        sentence = []
        task = []
        if self.index + self.batch_size <= self.end:
            it_data = self.task_sentence[self.shuffle[self.index: self.index + self.batch_size], :]  # 迭代数据
            self.index = self.end - self.index - self.batch_size
        if self.index + self.batch_size == self.end:
            self.shuffle = sample(range(0, self.end, 1), self.end)
        if self.index + self.batch_size > self.end:
            it_data = self.task_sentence[self.shuffle[self.index: self.end], :]  # 随机选取
            self.index = 0
            remain = self.task_sentence[self.shuffle[self.index: self.index + self.batch_size], :]  # 剩余
            it_data = np.concatenate((it_data, remain), axis=0)
        for cur_sentences, cur_task in it_data:
            sentence.append(self.sentence2index(cur_sentences, self.vocab))  # 句子
            task.append(self.task2id[cur_task[-1]])  # 任务
        return np.array(sentence), np.array(task)

    def sentence2index(self, sen, vocab):
        # print('sen sne:', sen)
        # sen = sen.split()
        assert isinstance(sen, list) and len(sen) > 0
        assert isinstance(vocab, dict) and len(vocab) > 0
        sen2id = []
        for cur_sen in sen:
            sen2id.append(vocab.get(cur_sen, 0))  # 如果找不到，就用0代替。
        return sen2id

    def task2index(self, cur_tasks, mapping):
        assert isinstance(cur_tasks, list) and len(cur_tasks) > 0 and hasattr(cur_tasks, '__len__')
        assert isinstance(mapping, dict) and len(mapping) > 0
        cur_task2index_mapping = []
        for cur_task in cur_tasks:
            cur_task2index_mapping.append(mapping[cur_task])
        return cur_task2index_mapping


if __name__ == '__main__':

     so = DataParse(train_data_path=r'../../data/train_test_data/new_train.txt',
                    task2id_id2task_path='task2id_id2task',
                    sentence_task_path='sentence_task',
                    vocb_path='vocb')
     so.read_corpus(so.train_data_path)
     row_datas = pickle.load(open(so.sentence_task_path, mode='rb'))
     so.creat_vocab(row_datas)