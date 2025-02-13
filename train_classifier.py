import pickle
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_df = pd.read_csv(r'C:\Users\Asus\Desktop\coding\PycharmProjects\sign_language_detector\sign_mnist_train.csv')
test_df = pd.read_csv(r'C:\Users\Asus\Desktop\coding\PycharmProjects\sign_language_detector\sign_mnist_test.csv')
train_label = train_df['label']
trainset = train_df.drop(['label'],axis=1)
X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
test_label=test_df['label']
X_test=test_df.drop(['label'],axis=1)

from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)

X_test=X_test.values.reshape(-1,28,28,1)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

X_test=X_test/255

sns.countplot(train_label)
plt.title("Frequency of each label")

model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5),
                 strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
model.add(Conv2D(64, kernel_size=(2, 2),
                 strides=1, activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), 2, padding='same'))
model.add(Conv2D(32, kernel_size=(2, 2),
                 strides=1, activation='relu', padding='same'))
model.add(MaxPool2D((2, 2), 2, padding='same'))

model.add(Flatten())

model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
         epochs = 35,
          validation_data=(X_test,y_test),
          shuffle=1
         )

(ls,acc)=model.evaluate(x=X_test,y=y_test)
print('MODEL ACCURACY = {}%'.format(acc*100))




f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
