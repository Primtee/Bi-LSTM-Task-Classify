import os
import re
import pickle
import numpy as np
import tensorflow as tf
from copy import deepcopy
from task_quick_dection.bi_lstm_model import Bi_LSTM_Task
from task_quick_dection.data_parse.data_parse import DataParse


class Config:
    def __init__(self):
        # self.root_data_path = 'data_path'
        self.root_data_path = r'model_save'
        self.batch_size = 32
        self.epoch = 100
        self.hidden_dim = 100
        self.optimizer = 'Adam'
        self.lr = 0.001
        self.clip = 5.0  # 防止梯度爆炸
        self.dropout = 0.9
        self.update_embedding = True  # 训练的时候更新映射
        self.pretrain_embedding = False  # 词向量的初始化方式，随机初始化
        self.embedding_dim = 100  # 词向量的维数
        self.shuffle = True  # 打乱训练数据
        self.log = 'train_log.txt'

    @property
    def get_log(self):
        return self.log

    @property
    def get_root_data_path(self):
        return self.root_data_path

    @property
    def get_batch_size(self):
        return self.batch_size

    @property
    def get_epoch(self):
        return self.epoch

    @property
    def get_hidden_dim(self):
        return self.hidden_dim

    @property
    def get_optimizer(self):
        return self.optimizer

    @property
    def get_lr(self):
        return self.lr

    @property
    def get_clip(self):
        return self.clip

    @property
    def get_dropout(self):
        return self.dropout

    @property
    def get_update_embedding(self):
        return self.update_embedding

    @property
    def get_pretrain_embedding(self):
        return self.pretrain_embedding

    @property
    def get_embedding_dim(self):
        return self.embedding_dim

    @property
    def get_shuffle(self):
        return self.shuffle


class Train:

    def __init__(self, vocb_path, task2id_id2task_path, model_name, sentence_task_path):
        self.config = Config()
        self.dataparse = DataParse(train_data_path = None,
                                   task2id_id2task_path='task_quick_dection.data_parse.data_parse/task2id_id2task',
                                   sentence_task_path='task_quick_dection.data_parse.data_parse/sentence_task',
                                   vocb_path='task_quick_dection.data_parse.data_parse/vocb')
        self.vocb_path = vocb_path
        self.task2id_id2task_path = task2id_id2task_path
        self.sentence_task_path = sentence_task_path
        self.model_name = model_name
        self.word2id = pickle.load(open(vocb_path, mode='rb'))  # 获取本地存放的字典。
        self.task2id = pickle.load(open(task2id_id2task_path, mode='rb'))['task2id']
        self.id2task = pickle.load(open(task2id_id2task_path, mode='rb'))['id2task']
        if not self.config.get_pretrain_embedding:
            self.embeddings = self.dataparse.random_embedding(self.config.get_embedding_dim, len(self.word2id))
        else:
            pre_trained_word_model_path = os.path.join('data_path', 'pre_trained_word.pkl')
            assert os.path.exists(pre_trained_word_model_path), '暂时没有预训练好的词向量'
            self.embeddings = np.array(np.load(pre_trained_word_model_path), dtype='float32')  # 否则加载预训练好词向量。
        self.model_same_path = self.config.get_root_data_path.__add__('/').__add__(model_name).__add__('/')  # 模型路径

    def train(self):  # 越界。
        model = Bi_LSTM_Task(param_config=self.config,
                             embeddings=self.embeddings,
                             vocab=self.word2id,
                             model_save_path=self.model_same_path,
                             sentence_task_path=self.sentence_task_path,
                             task2id_id2task=self.task2id_id2task_path,
                             vocb_path=self.vocb_path,
                             task2id=self.task2id
                            )
        model.build_graph()  # 创建网络时，就已经计算了网络的损失
        print('net created......')
        out_put_log = open(self.config.get_log, mode='w', encoding='utf-8', newline='')
        model.train(log_file=out_put_log)

    def predict(self):
        model = Bi_LSTM_Task(param_config=self.config,
                             embeddings=self.embeddings,
                             vocab=self.word2id,
                             model_save_path=self.model_same_path,
                             sentence_task_path=self.sentence_task_path,
                             task2id_id2task=self.task2id_id2task_path,
                             vocb_path=self.vocb_path,
                             task2id=self.task2id,
                             id2task=self.id2task
                            )
        model.build_graph()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt_file = tf.train.latest_checkpoint(self.config.get_root_data_path.__add__('/task/'))
            saver.restore(sess, ckpt_file)

            while True:
                txt = input()
                if txt == '' or txt.isspace():
                    break
                else:
                    # task detect result
                    # row_text 进行过滤
                    task_detect_txt = self.dataparse.clean_txt(txt)
                    print('测试 task_detect_txt:', task_detect_txt)
                    model.predict(sess, [task_detect_txt], predicted=True)


if __name__ == '__main__':

    train_pro = Train(vocb_path='data_parse/vocb',
                      task2id_id2task_path='data_parse/task2id_id2task',
                      model_name='task',
                      sentence_task_path='data_parse/sentence_task')
    train_pro.train()
    # train_pro.predict()
