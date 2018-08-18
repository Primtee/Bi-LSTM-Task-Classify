import time
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from task_quick_dection.data_parse.data_parse import Data_Inter
from task_quick_dection.utils import check_multi_path


class Bi_LSTM_Task:
    def __init__(self, param_config,
                 embeddings,
                 vocab,
                 model_save_path,
                 sentence_task_path,
                 task2id_id2task,
                 vocb_path,
                 task2id=None,
                 id2task=None):
        self.batch_size = param_config.get_batch_size
        self.epoch_num = param_config.get_epoch
        self.hidden_dim = param_config.get_hidden_dim
        self.embeddings = embeddings
        self.update_embedding = param_config.get_update_embedding
        self.dropout_keep_prob = param_config.get_dropout
        self.optimizer = param_config.get_optimizer
        self.lr = param_config.get_lr
        self.clip_grad = param_config.get_clip
        self.vocab = vocab
        self.shuffle = param_config.get_shuffle
        self.model_path = model_save_path
        self.data_inter = Data_Inter(batch_size=self.batch_size,
                                     task_sentence_path=sentence_task_path,
                                     task2id_id2task=task2id_id2task,
                                     vocb_path=vocb_path)  # 迭代器。
        if task2id is not None:
            self.task2id = task2id
            self.task_counts = len(self.task2id)
        if id2task is not None:
            self.id2task = id2task

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.loss_op()
        self.trainstep_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.task_targets = tf.placeholder(tf.int32, [self.batch_size],  # 真正的意图
                                             name='intent_targets')  # 16

    def lookup_layer_op(self):
        """
        将词的one-hot形式表示成词向量的形式，词向量这里采用随机初始化的形式，显然可以使用w2c预训练的词向量。
        """
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)  # 只有当训练的时候droup才会起作用。

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=self.word_embeddings,
                        sequence_length=self.sequence_lengths,
                        dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
            encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        with tf.variable_scope("proj"):

            w_task = tf.get_variable(name="W_task",
                                     shape=[2 * self.hidden_dim, self.task_counts],  # intent的个数
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32)
            b_task = tf.get_variable(name="b_task",
                                     shape=[self.task_counts],
                                     initializer=tf.zeros_initializer(),
                                     dtype=tf.float32)

            # task
            intent_logits = tf.add(tf.matmul(encoder_final_state_h, w_task), b_task)  # 得到意图的识别
            self.softmax_score = tf.nn.softmax(intent_logits)
            self.task = tf.argmax(intent_logits, axis=1)
            # 定义task的分类的损失
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.task_targets,
                                                                                      depth=self.task_counts,
                                                                                      dtype=tf.float32),
                                                                    logits=intent_logits)
            self.loss_task = tf.reduce_mean(cross_entropy)

    def loss_op(self):
            self.loss = self.loss_task  # 任务识别的损失

    def trainstep_op(self):
        """
        训练节点.
        """
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)  # 全局训批次的变量，不可训练。
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def pad_sequences(self, sequences, pad_mark=0, predict=False):
        """
        批量的embedding，其中rowtext embedding的长度要与slots embedding的长度一致，不然使用crf时会出错。
        :param sequences: 批量的文本格式[[], [], ......, []]，其中子项[]里面是一个完整句子的embedding（索引。）
        :param pad_mark:  长度不够时，使用何种方式进行padding
        :param predict:  是否是测试
        :return:
        """
        # print('sequences:', sequences)
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            # print('传进来的数据:', seq)
            if predict:
                seq = list(map(lambda x: self.vocab.get(x, 0), seq))
            seq_ = seq[:len(seq)] + [pad_mark] * max(max_len - len(seq), 0)  # 求得最大的索引长度。
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    def train(self, log_file=None):
        """
            数据由一个外部迭代器提供。
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_index in range(0, self.epoch_num, 1):
                batches_recording = 0
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                batches_recording += 1
                while batches_recording <= self.data_inter.num_batches:  # 迭代器内部是没有设置结束标志。
                    batches_recording += 1
                    sentence, tasks = self.data_inter.next()  # 迭代器，每次取出一个batch块.
                    # print('all tasks is:', tasks)
                    feed_dict, _ = self.get_feed_dict(sentence, tasks, self.lr, self.dropout_keep_prob)
                    _, loss_train, step_num_ = sess.run([self.train_op,
                                                         self.loss,
                                                         self.global_step],
                                                        feed_dict=feed_dict)
                    if batches_recording % 2 == 0:
                        if log_file is not None:
                            log_file.write('time:'.__add__(start_time).__add__('\tepoch: ').
                                           __add__(str(epoch_index + 1)).__add__('\tstep:').
                                           __add__(str(batches_recording + epoch_index * self.data_inter.num_batches)).
                                           __add__('\tloss:').__add__(str(loss_train)).__add__('\n'))

                        print('time {} epoch {}, step {}, loss: {:.4}'.
                              format(start_time, epoch_index + 1, batches_recording + epoch_index *
                                     self.data_inter.num_batches, loss_train))

                    check_multi_path(self.model_path)
                    saver.save(sess, self.model_path, global_step=self.data_inter.num_batches * (epoch_index + 1))
            if log_file is not None:
                log_file.close()

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None, predicted=False):
        """

        :param seqs:  训练的batch块
        :param labels:  实体标签
        :param lr:  学利率
        :param dropout:  活跃的节点数，全连接层
        :return: feed_dict  训练数据
        :return: predicted  测试标志
        """
        word_ids, seq_len_list = self.pad_sequences(seqs, pad_mark=0, predict=predicted)
        feed_dict = {self.word_ids: word_ids,  # embedding到同一长度
                     self.sequence_lengths: seq_len_list,  # 实际长度。
                     }
        if labels is not None:
            feed_dict[self.task_targets] = labels
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, seq_len_list

    def predict(self, sess, seqs, predicted=False):
        """

        :param sess:
        :param seqs:
        :param predicted:
        :return: label_list
                 seq_len_list
        """
        feed_dict, _ = self.get_feed_dict(seqs, dropout=1.0, predicted=predicted)

        cur_task, intent_logits = sess.run([self.task, self.softmax_score], feed_dict=feed_dict)
        task_id = int(cur_task[0])
        print('predicted results:', intent_logits)
        print('predicted results:', self.id2task[task_id])
        return cur_task