#coding:utf-8
from __future__ import division
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import  numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.externals import joblib
def get_data(args):
    #从csv读取数据
    data = read_csv_as_pd(args.data_path)
    #将数据全部转换为字符串形式
    data = translate_all_into_string(data)
    #将数据进行结合
    data_text = combin_columns_together(data)
    #数据分词并保存,转化为向量 
    data_vector = get_vector(args,data_text)
    #获取label标签，并转化为one-hot
    data_label = get_label(data)
    #由于只需要查询，，不需要拟合，验证数据和训练数据一样可以完成要求
    train_text = data_vector
    train_label = data_label
    val_text = data_vector
    val_label = data_label
    return train_text,train_label,val_text,val_label


#读取数据
def read_csv_as_pd(data_path):
    data = pd.DataFrame()
    csv_File = open(data_path,"r",encoding='utf-8')
    df_ = pd.read_csv(csv_File)
    data = data.append(df_)
    print(data)
    return data
#全部转化为字符串
def translate_all_into_string(data):
    for columns_name in data.columns:
        data[columns_name] = data[columns_name].astype('str')
    return data
#将数据结合成一句话
def combin_columns_together(data):
    for columns_name in data.columns:
        data[data.columns[0]] = data[data.columns[0]].str.cat([data[columns_name]],sep='_')
    print("数据合并后的样子: ",data[data.columns[0]][0])
    return data[data.columns[0]]
#将数据进行分词并保存在word目录下
def get_vector(args,data_text):
    #获取tokenizer
    tokenizer  = get_tokenizer(args,data_text)
    #保存tokenizer
    joblib.dump(tokenizer,args.word_pkl_save_path)
    #将data_text转化为data_vector
    data_vector = np.array(tokenizer.texts_to_sequences(data_text))
    print("数据转化为向量后的样子：",data_vector[0])
    #将data_vector转化为向量形式，并且设置最大长度
    data_vector = pad_sequences(data_vector,maxlen=args.MAX_SEQUENCE_LENGTH)
    print("训练数据的shape为:",data_vector.shape)
    return data_vector

#获取tokenizer
def get_tokenizer(args,data_text):
    #训练词向量返回，并保存文本
    tokenizer = Tokenizer(num_words = args.MAX_NB_WORDS,filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower = True)
    tokenizer.fit_on_texts(data_text)
    print('分词的总数：',len(tokenizer.word_index))
    #print(tokenizer.word_index)
    return tokenizer

#获取label one-hot
def get_label(data):
    return np.array(pd.get_dummies(data['ItemName'].values).values) #可再修改
    

#计算类别个数
def get_label_counts(data,label_name):
    return data[label_name].value_counts().count()

#将单条数据处理成向量
def single_sample_to_vector(args,tokenizer,text):
    list = []
    list.append(text)
    predict_text = tokenizer.texts_to_sequences(list)
    predict_text = pad_sequences(predict_text,maxlen = args.MAX_SEQUENCE_LENGTH)
    return predict_text
#存下one-hot之后的标签，供给predit函数和验证用
def which_label(args):
    data = read_csv_as_pd(args.data_path)
    data_label = np.array(data['ItemName'].values)
    label = pd.get_dummies(data_label)
    create_csv_file(args.label_path,label.columns)

    return label.columns
#向一个文件写入一些数据
def create_csv_file(path,data):
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)


#画混淆矩阵热点图
def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    #save
    plt.savefig(savname)
    #show
    plt.show()

