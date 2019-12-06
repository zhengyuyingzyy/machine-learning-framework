# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:22:17 2019

@author: DELL
"""

import numpy as np
import os
import pandas as pd
import logging
from keras.models import load_model  # 一系列网络层按顺序构成的栈
from keras.preprocessing import image
import tensorflow as tf

logger = logging.getLogger("forecast by model")
# 每个验证码所包含的字符数
word_len = 5
image_path = 'test/'
# 验证码所包含的字符_表示未知
captcha_word = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


# 字符总数
word_class = len(captcha_word)

#日志初始化
def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

def custom_accuracy(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, word_len, word_class])
    max_idx_p = tf.argmax(predict, 2)#这个做法牛逼，不用再做stack和reshape了，2，是在Charset那个维度上
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, word_len,word_class]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    _result = tf.map_fn(fn=lambda e: tf.reduce_all(e),elems=correct_pred,dtype=tf.bool)
    return tf.reduce_mean(tf.cast(_result, tf.float32))

# load json and create model
def create_model():
    weight_path = 'model/captcha__model_50.h5'
    model = load_model(weight_path,
                        custom_objects={'custom_accuracy': custom_accuracy})
    return model

# 把数组转换回文字
def vec_to_captcha(vec):
    text = []
    # 把概率小于0.5的改为0，标记为错误
    vec[vec < 0.5] = 0

    char_pos = vec.nonzero()[0]

    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)


if __name__ == '__main__':
    dc = dict()
    preee = []
    init_logger()
    model = create_model()
    image_list = []
    for item in os.listdir(image_path):
        image_list.append(item)
    #np.random.shuffle(image_list)
    # 图片总数
    image_count = 0
    # 成功次数
    success_count = 0
    # 图片
    for i, img in enumerate(image_list):
        if i % 1000 == 0:
            print(i)
        img_path = image_path + img
        # 读取图片
        raw_img = image.load_img(img_path, target_size=(30, 150))
        code = img.replace('.jpg', '')
        code = code.split('_')[0]
        logger.debug('正确的验证码为' + code)
        X_test = np.zeros((1, 30, 150, 3), dtype=np.float32)
        X_test[0] = image.img_to_array(raw_img)
        # 预测
        predict = model.predict(X_test)
        n = 62  # 大列表中几个数据组成一个小列表
        arr = []
        arr.append(predict[0][0:62])
        arr.append(predict[0][62:124])
        arr.append(predict[0][124:186])
        arr.append(predict[0][186:248])
        arr.append(predict[0][248:310])
        predictions = []
        predictions.append(np.argmax(arr[0]))
        predictions.append(np.argmax(arr[1]))
        predictions.append(np.argmax(arr[2]))
        predictions.append(np.argmax(arr[3]))
        predictions.append(np.argmax(arr[4]))
        # predictions = np.argmax(predict, axis=1)
        # 标签字典
        keys = range(62)
        label_dict = dict(zip(keys, captcha_word))
        
        result = ''.join([label_dict[pred] for pred in predictions])
        image_count = image_count + 1
        dc[int(code)] = result
        
        #print(te[:,1])

        if result == code:
            success_count  = success_count + 1
        logger.debug("预测的结果为" + result)
        preee.append(result)
        logger.debug("目前正确率" + str(success_count / image_count))
    logger.debug("总次数" + str(image_count))
    logger.debug("成功次数" + str(success_count))
    logger.debug("正确率" + str(success_count/image_count))
    
    
    
def save_csv(labels):
    test_data_0=sorted(labels.items(),key=lambda x:x[0])
    te = np.array(test_data_0)
    #print(te)
    length = len(te[:,1])
    test = pd.DataFrame(data = te[:,1],index = range(0,length),columns=['y'])
    test.index.name = 'id'
    test.to_csv('test2.csv')
    
    
save_csv(dc)
