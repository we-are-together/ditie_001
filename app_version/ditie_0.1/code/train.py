#coding: utf-8
from tool.tool import get_data
import tensorflow as tf
def train(args):
    '''
    data= [train_text,train_label,val_text,val_label]

    '''
    train_text,train_label,val_text,val_label = get_data(args)
    print(train_text.shape)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(args.MAX_NB_WORDS, output_dim = args.EMBEDDING_DIM, input_length = args.MAX_SEQUENCE_LENGTH,),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
    #     tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(69,activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_text,train_label, epochs=20,validation_data=(val_text,val_label),verbose=1)
# Save the model
    model.save(args.model_path)
