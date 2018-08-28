import os
import tensorflow as tf
import numpy as np
from read_utils import TextConverter, batch_generator
from model import Model,Config
# from model_attention import Model,Config

def main(_):

    model_path = os.path.join('models', Config.file_name)

    converter = TextConverter(vocab_dir='data/vocabs', max_vocab=Config.vocab_size, seq_length = Config.seq_length)
    print('vocab lens:',converter.vocab_size)


    # 加载上一次保存的模型
    model = Model(Config)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    while True:

        english_speek = input("上联:")
        english_speek = ' '.join(english_speek)
        english_speek = english_speek.split()
        en_arr, arr_len = converter.text_en_to_arr(english_speek)

        test_g = [np.array([en_arr,]), np.array([arr_len,])]
        output_ids = model.test(test_g, model_path, converter)
        strs = converter.arr_to_text(output_ids)
        print('下联:',strs)


if __name__ == '__main__':
    tf.app.run()