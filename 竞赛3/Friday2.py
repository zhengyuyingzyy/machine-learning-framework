# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:02:13 2019

@author: DELL
"""

import numpy as np
import os

import win_unicode_console
win_unicode_console.enable()

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate ,BatchNormalization
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
from keras.utils.vis_utils import plot_model
import tensorflow as tf

# 验证码所包含的字符 _表示未知
captcha_word = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# 图片的长度和宽度
width = 150
height = 30

# 每个验证码所包含的字符数
word_len = 5
# 字符总数
word_class = len(captcha_word)


train_dir = 'train'


# 生成字符索引，同时反向操作一次，方面还原
char_indices = dict((c, i) for i, c in enumerate(captcha_word))
indices_char = dict((i, c) for i, c in enumerate(captcha_word))


# 验证码字符串转数组
def captcha_to_vec(captcha):
    # 创建一个长度为 字符个数 * 字符种数 长度的数组
    vector = np.zeros(word_len * word_class)

    # 文字转成成数组
    for i, ch in enumerate(captcha):
        idex = i * word_class + char_indices[ch]
        vector[idex] = 1
    return vector


# 把数组转换回文字
def vec_to_captcha(vec):
    text = []
    # 把概率小于0.5的改为0，标记为错误
    vec[vec < 0.5] = 0

    char_pos = vec.nonzero()[0]

    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)

# 自定义评估函数
def custom_accuracy(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, word_len, word_class])
    max_idx_p = tf.argmax(predict, 2)#这个做法牛逼，不用再做stack和reshape了，2，是在Charset那个维度上
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, word_len,word_class]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e),elems=correct_pred,dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

#获取目录下样本列表
image_list = []

#
for item in os.listdir(train_dir):
    image_list.append(item)
np.random.shuffle(image_list)
#创建数组，储存图片信息。样本个数、宽度和高度。
# 3代表图片的通道数，如果对图片进行了灰度处理，可以改为单通道 1
X = np.zeros((len(image_list), height, width, 3), dtype = np.uint8)
# 创建数组，储存标签信息
y = np.zeros((len(image_list), word_len * word_class), dtype = np.uint8)

for i,img in enumerate(image_list):
    if i % 10000 == 0:
        print(i)
    img_path = train_dir + "/" + img
    #读取图片
    raw_img = image.load_img(img_path, target_size=(height, width))
    #讲图片转为np数组
    X[i] = image.img_to_array(raw_img)
    #讲标签转换为数组进行保存
    y[i] = captcha_to_vec(img.split('.')[0])

#创建输入，结构为 高，宽，通道
input_tensor = Input( shape=(height, width, 3))

x = input_tensor

#构建卷积网络
#两层卷积层，一层池化层，重复3次。因为生成的验证码比较小，padding使用same
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
# x= BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)


x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = Convolution2D(64, 3, padding='same', activation='relu')(x)
# x= BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Convolution2D(128, 3, padding='same', activation='relu')(x)
x = Convolution2D(128, 3, padding='same',activation='relu')(x)
# x= BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
x = Flatten()(x)
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
x = Dropout(0.25)(x)
x= BatchNormalization()(x)
#Dense就是常用的全连接层
#最后连接4个分类器，每个分类器是62个神经元，分别输出62个字符的概率。
x = [Dense(word_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(word_len)]
output = concatenate(x)
#构建模型
model = Model(inputs=input_tensor, outputs=output)
# model.add(BatchNormalization())


#这里优化器选用Adadelta，学习率0.1
opt = Adadelta(lr=0.1)
#编译模型以供训练，损失函数使用 categorical_crossentropy，使用accuracy评估模型在训练和测试时的性能的指标
#这里使用自定义的评估模型
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy',custom_accuracy])
#每次epoch都保存一下权重，用于继续训练
checkpointer = ModelCheckpoint(filepath="model/weights.{epoch:02d}--{val_loss:.2f}.hdf5",
                               verbose=2, save_weights_only=True)
#开始训练，validation_split代表10%的数据不参与训练，用于做验证急
#我之前训练了50个epochs以上，这里根据自己的情况进行选择。如果输出的val_acc已经达到你满意的数值，可以终止训练
model.fit(X, y, epochs= 100,callbacks=[checkpointer], validation_split=0.1)
# plot model
#plot_model(model, to_file='model/model.png', show_shapes=True)

#保存权重和模型
model.save_weights('model/captcha_model_weights100.h5')
model.save('model/captcha__model_100.h5')