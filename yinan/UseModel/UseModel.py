#coding=utf-8
# 模型的加载及使用
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

tokenizer = Tokenizer(num_words=200, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True) #Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
text = input("请输入要预测的数据:")
print("正在加载模型...")
load_model = load_model("/home/sun/文档/地铁项目/LSTM_model.h5")
seq = tokenizer.texts_to_sequences(text)
padded = pad_sequences(seq, maxlen=14)
pred = load_model.predict(padded)
ItemName_id = pred.argmax(axis=1)[0]
ItemName = ItemName_id_df[ItemName_id_df.ItemName_id==ItemName_id]['ItemName'].values[0]

print("\n正在进行预测该数据: ")
print(text)
print("\n预测结果是: ")
print(ItemName)
