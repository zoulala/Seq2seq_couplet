# Seq2seq_couplet
Tensorflow实现 seq2seq，并训练实现对对联

方法一、 LSTM

方法二、bi-LSTM+ attention（双向双层LSTM+注意力机制）

# Data
70w+ 对的对联数据，数据见下面分享


# Train
> python trian.py

```
.
.
.
step: 19860/20000...  loss: 1.4368412494659424...  0.0420 sec/batch
step: 19880/20000...  loss: 1.473110556602478...  0.0415 sec/batch
step: 19900/20000...  loss: 1.7176944017410278...  0.0429 sec/batch
step: 19920/20000...  loss: 1.5570639371871948...  0.0426 sec/batch
step: 19940/20000...  loss: 1.6691091060638428...  0.0417 sec/batch
step: 19960/20000...  loss: 1.5215612649917603...  0.0420 sec/batch
step: 19980/20000...  loss: 1.4277087450027466...  0.0424 sec/batch
step: 20000/20000...  loss: 1.2709236145019531...  0.0421 sec/batch
```

# Results
> python test.py

```
上联:福如东海长流水
下联: 心有春风有旧人
上联:千秋月色君长看
下联: 一片风光月似来
上联:梦里不知身是客
下联: 心中不觉梦无痕
上联:如此清秋何吝酒
下联: 有情明月不知人
上联:恭喜发财
下联: 和谐富民
上联:你以为的就是你以为啊
下联: 人有有人心不可是非非
上联:深度学习有深度
下联: 长生心事不知人
```
只是一个初步的训练，结果不是很好。


# Models

我的数据+模型：链接：https://pan.baidu.com/s/126sUTtz0wWKlWx8iohugmw 密码：rue9

项目参考：ref:https://github.com/wb14123/seq2seq-couplet

