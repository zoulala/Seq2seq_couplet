import os
import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import  Model,Config
# from model_att_bi import Model,Config

def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)


    converter = TextConverter(vocab_dir='data/vocabs', max_vocab=Config.vocab_size, seq_length = Config.seq_length)
    print('vocab lens:',converter.vocab_size)


    en_arrs = converter.get_en_arrs('data/train/in.txt')
    de_arrs = converter.get_de_arrs('data/train/out.txt')

    train_g = batch_generator( en_arrs, de_arrs, Config.batch_size)


    # 加载上一次保存的模型
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to training...')
    model.train(train_g, model_path)


if __name__ == '__main__':
    tf.app.run()