import os
import time
import numpy as np
import tensorflow as tf


class Config(object):
    """RNN配置参数"""
    file_name = 'lstm_c'  #保存模型文件

    embedding_dim = 100      # 词向量维度
    seq_length = 30        # 序列长度
    # num_classes = 2        # 类别数
    vocab_size = 5000       # 词汇表达小


    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    # rnn = 'gru'             # lstm 或 gru
    share_emb_and_softmax = False  # 是否共享词向量层和sorfmax层的参数。（共享能减少参数且能提高模型效果）

    train_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 32  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config):
        self.config = config

        # 待输入的数据
        self.en_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='encode_input')
        self.en_length = tf.placeholder(tf.int32, [None], name='ec_length')

        self.zh_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='decode_input')
        self.zh_length = tf.placeholder(tf.int32, [None], name='zh_length')
        self.zh_seqs_label = tf.placeholder(tf.int32, [None, self.config.seq_length], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(3, dtype=tf.float32, trainable=False, name="global_loss")

        # seq2seq模型
        self.seq2seq()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def seq2seq(self):
        """seq2seq模型"""

        # 词嵌入层
        en_embedding = tf.get_variable('en_emb', [self.config.vocab_size, self.config.embedding_dim])
        zh_embedding = tf.get_variable('zh_emb', [self.config.vocab_size, self.config.embedding_dim])
        embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
        self.en_embedding = tf.concat([en_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
        self.zh_embedding = tf.concat([zh_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值

        embed_en_seqs = tf.nn.embedding_lookup(self.en_embedding, self.en_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
        embed_zh_seqs = tf.nn.embedding_lookup(self.zh_embedding, self.zh_seqs)

        # 在词嵌入上进行dropout
        embed_en_seqs = tf.nn.dropout(embed_en_seqs, keep_prob=self.keep_prob)
        embed_zh_seqs = tf.nn.dropout(embed_zh_seqs, keep_prob=self.keep_prob)

        def get_mul_cell(hidden_dim, num_layers):
            # 创建多层lstm
            def get_en_cell(hidden_dim):
                # 创建单个lstm
                enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
                return enc_base_cell
            return tf.nn.rnn_cell.MultiRNNCell([get_en_cell(hidden_dim) for _ in range(num_layers)])


        with tf.variable_scope("encoder"):
            # 构建双向lstm
            encode_cell_fw = get_mul_cell(self.config.hidden_dim, self.config.num_layers)
            encode_cell_bw = get_mul_cell(self.config.hidden_dim, self.config.num_layers)
            bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(  cell_fw=encode_cell_fw,
                                                                                    cell_bw=encode_cell_bw,
                                                                                    inputs=embed_en_seqs,
                                                                                    sequence_length=self.en_length,
                                                                                    dtype=tf.float32,
                                                                                    time_major=False)
            # concat encode output and state
            enc_output = tf.concat(bi_encoder_output, -1)  # fw,bw 输出拼接
            encoder_state = []
            for layer_id in range(self.config.num_layers):
                encoder_state.append(bi_encoder_state[0][layer_id])
                encoder_state.append(bi_encoder_state[1][layer_id])
            self.enc_state = tuple(encoder_state)  # (f_c, b_c, f_h, b_h)

        with tf.variable_scope("decoder_attention"):
            self.dec_cell =  get_mul_cell(self.config.hidden_dim, self.config.num_layers)
            # 选择注意力权重计算模型，BahdanauAttention是使用一个隐藏层的前馈网络，memory_sequence_length是一个维度[batch_size]的张量，代表每个句子的长度
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.config.hidden_dim, enc_output, memory_sequence_length=self.en_length)
            # attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(self.config.hidden_dim, enc_output, self.en_length, normalize=True)
            # 将解码器和注意力模型一起封装成更高级的循环网络
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism,attention_layer_size = self.config.hidden_dim)

            # 通过dynamic_rnn对cell展开时间维度，没有指定init_state,完全依赖注意力作为信息来源
            dec_output, self.dec_state = tf.nn.dynamic_rnn(attention_cell,
                                              inputs=embed_zh_seqs,
                                              sequence_length=self.zh_length,
                                              # initial_state=self.enc_state,  # 没有使用encoder层的隐状态,自动为0状态
                                              time_major=False,
                                              dtype=tf.float32)

            # from tensorflow.python.layers import core as layers_core
            # helper = tf.contrib.seq2seq.TrainingHelper(
            # embed_zh_seqs, self.zh_length, time_major=False)
            # projection_layer = layers_core.Dense(self.config.vocab_size, use_bias=False)
            # decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper,
            #                               self.enc_state, output_layer=projection_layer)
            # outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            #                                       maximum_iterations=100)
            # dec_output = outputs.rnn_output

        with tf.name_scope("sorfmax_weights"):
            if self.config.share_emb_and_softmax:
                self.softmax_weight = tf.transpose(self.zh_embedding)
            else:
                self.softmax_weight = tf.get_variable("weight",[self.config.hidden_dim, self.config.vocab_size+1])  #+1是因为对未知的可能输出
            self.softmax_bias = tf.get_variable("bias",[self.config.vocab_size+1])


        with tf.name_scope("loss"):
            out_put = tf.reshape(dec_output, [-1, self.config.hidden_dim])
            logits = tf.matmul(out_put, self.softmax_weight) + self.softmax_bias
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.zh_seqs_label,[-1]), logits=logits)
            # 计算平均损失时，需要将填充位置权重设置为0，以免无效位置预测干扰模型训练
            label_weights = tf.sequence_mask(self.zh_length, maxlen=tf.shape(self.zh_seqs_label)[1], dtype=tf.float32)
            label_weights = tf.reshape(label_weights, [-1])
            self.mean_loss = tf.reduce_mean(loss*label_weights)

        with tf.name_scope("pres"):
            self.output_id = tf.argmax(logits, axis=1, output_type=tf.int32)[0]

        with tf.name_scope("optimize"):
            # 优化器
            # tvars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
            # train_op = tf.train.AdamOptimizer(self.config.learning_rate)
            # self.optim = train_op.apply_gradients(zip(grads, tvars),global_step=self.global_step)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss, global_step=self.global_step)

    def train(self, batch_train_g, model_path):
        with self.session as sess:
            for batch_en, batch_en_len, batch_zh, batch_zh_len, batch_zh_label in batch_train_g:
                start = time.time()
                feed = {self.en_seqs: batch_en,
                        self.en_length: batch_en_len,
                        self.zh_seqs: batch_zh,
                        self.zh_length: batch_zh_len,
                        self.zh_seqs_label: batch_zh_label,
                        self.keep_prob: self.config.train_keep_prob}
                _, mean_loss = sess.run([self.optim, self.mean_loss ], feed_dict=feed)
                end = time.time()

                # control the print lines
                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {}... '.format(mean_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % self.config.save_every_n == 0):
                    self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)
                if self.global_step.eval() >= self.config.max_steps:
                    break

    def test(self, test_g, model_path, zt):
        batch_en, batch_en_len = test_g
        feed = {self.en_seqs: batch_en,
                self.en_length: batch_en_len,
                self.keep_prob:1.0}
        enc_state = self.session.run(self.enc_state, feed_dict=feed)

        output_ids = []
        dec_state = enc_state
        dec_input, dec_len = zt.text_to_arr(['<s>',])  # decoder层初始输入
        dec_input = np.array([dec_input[:-1], ])
        dec_len = np.array([dec_len, ])
        for i in range(self.config.seq_length):  # 最多输出50长度，防止极端情况下死循环
            feed = {self.enc_state: dec_state,
                    self.zh_seqs: dec_input,
                    self.zh_length: dec_len,
                    self.keep_prob: 1.0}
            dec_state, output_id= self.session.run([self.dec_state, self.output_id], feed_dict=feed)

            char = zt.int_to_word(output_id)
            if char == '</s>':
                break
            output_ids.append(output_id)

            arr = [output_id]+[len(zt.vocab)] * (self.config.seq_length - 1)
            dec_input = np.array([arr, ])
        return output_ids


    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))