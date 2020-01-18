#coding=utf-8
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import jieba as jb
import re

df = pd.read_csv('/home/sun/文档/地铁项目/total0203合并后.csv')
df=df[['ItemName','Feature']]
print("数据总量: %d ." % len(df))
df.sample(69)

d = {'ItemName':df['ItemName'].value_counts().index, 'count': df['ItemName'].value_counts()}
df_ItemName = pd.DataFrame(data=d).reset_index(drop=True)#将会重置索引为0,1,2,3...的这种形式

df['ItemName_id'] = df['ItemName'].factorize()[0] #factorize函数可以将Series中的标称型数据映射称为一组数字，相同的标称型映射为相同的数字
ItemName_id_df = df[['ItemName', 'ItemName_id']].drop_duplicates().sort_values('ItemName_id').reset_index(drop=True)
ItemName_to_id = dict(ItemName_id_df.values)
id_to_ItemName = dict(ItemName_id_df[['ItemName_id', 'ItemName']].values)
df.sample(69)

#ItemName_id_df

# 设置最频繁使用的6000个词(在texts_to_matrix是会取前MAX_NB_WORDS,会取前MAX_NB_WORDS列)
MAX_NB_WORDS = 6000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 14
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100
# 转为词向量
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True) #Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类。
tokenizer.fit_on_texts(df['Feature'].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))

X = tokenizer.texts_to_sequences(df['Feature'].values)
#填充X,让X的各个列的长度统一
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
#多类标签的onehot展开
Y = pd.get_dummies(df['ItemName_id']).values

#拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)

#定义模型
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(69, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
#损失
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#精准度
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

# 该函数进行预测
def predict(text):
    #txt = remove_punctuation(text)
    #txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    ItemName_id= pred.argmax(axis=1)[0]
    return ItemName_id_df[ItemName_id_df.ItemName_id==ItemName_id]['ItemName'].values[0]

print(predict('混凝土支撑-1-800X900 矩形梁 结构框架 45.60367454068167 5.08529490039156 -4.593175853020454 48.228346456692194 71.8175783649585 -1.6404199475086252 2.624671916010506 66.73228346456693 2.952755905511829 517.1762312028607 759.889019778043'))
