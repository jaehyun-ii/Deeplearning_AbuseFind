import pandas as pd
from konlpy.tag import Komoran
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.models import Sequential, load_model
from keras.metrics import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K
import pickle

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

data = pd.read_csv('data_spacing.csv')
X_data = data['text']
y_data = data['isAbuse']
y_data = pd.to_numeric(y_data)
tokened = []


stop_words = []
ft = open('stopword.txt', 'r')
lines = ft.readlines()
for i in lines:
  i = i.rstrip()
  i=i.split(",")
  
  for j in i:
    stop_words.append(j)
ft.close()

komoran = Komoran()

for i in X_data:

  word_tokens = komoran.morphs(i)
  word_tokens = [word for word in word_tokens if not word in stop_words]
  tokened.append(word_tokens)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(tokened)
threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value


print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

X_data = tokenizer.texts_to_sequences(tokened)

drop_data = [index for index, sentence in enumerate(X_data) if len(sentence) < 1]


paddedX = tf.keras.preprocessing.sequence.pad_sequences(X_data, padding='post', maxlen=100)
X_data = np.array(paddedX)
X_data = np.delete(X_data, drop_data, axis=0)

y_data = np.array(y_data)
y_data = np.delete(y_data, drop_data, axis=0)

with open('tokenizer.pickle', 'wb') as handle:
     pickle.dump(tokenizer, handle)

print(X_data[0].shape)
print(X_data[1].shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1, stratify=y_data, shuffle=True)

embedding_dim = 128 # 임베딩 벡터의 차원
dropout_ratio = 0.5 # 드롭아웃 비율
num_filters = 128 # 커널의 수
kernel_size = [15,10] # 커널의 크기
hidden_units = 128 # 뉴런의 수


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(dropout_ratio))

model.add(Conv1D(num_filters, kernel_size[0], padding='valid', activation='swish', strides = 1))
model.add(Conv1D(num_filters, kernel_size[1], padding='valid', activation='swish', strides = 1))
model.add(GlobalMaxPooling1D(keepdims=True))
model.add(Bidirectional(LSTM(hidden_units))) # Bidirectional LSTM을 사용
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc' ,precision_m, recall_m, f1_m])
history = model.fit(X_train, y_train, epochs=30, callbacks=[es, mc], batch_size=64, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))





