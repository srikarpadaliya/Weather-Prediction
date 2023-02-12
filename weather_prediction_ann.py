
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



df=pd.read_csv('seattle-weather.csv')
df.head()

# Model Building 

from sklearn import tree
from sklearn.model_selection import train_test_split

wcpy_db = df.copy(deep = True)
wcpy_db.drop(labels="date", axis=1, inplace=True)

y = wcpy_db.weather
X = wcpy_db.drop(labels='weather', axis=1)

# Splitting the Data into training and testing part to provide an environment to the model for cross validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)

"""Artificial Neural Network"""

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_ann_train = sc.fit_transform(X_train)
X_ann_test = sc.transform(X_test)
# print(X_ann_train)

# label encode the output column
import numpy as np
# print(y_test)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.fit_transform(y_test)
print(y_train_le)

# Label Encoding Scheme
# 0 - drizzle
# 1 - fog
# 2 - rain
# 3 - snow
# 4 - sun

y_train_le = y_train_le.reshape(-1, 1)
y_test_le = y_test_le.reshape(-1, 1)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y_train_le)
enc.fit(y_test_le)
y_ann_train = enc.transform(y_train_le).toarray()
y_ann_test = enc.transform(y_test_le).toarray()
# print(y_ann_train.shape)
# print(y_ann_test)

from tensorflow import keras
from keras import layers

import keras_tuner
from keras_tuner.tuners import RandomSearch

# tuning parameters 
def model_test(hp):
  model = keras.Sequential()
  # adding base layer of our model number of neurons varry from 10 to 512
  model.add(layers.Dense(units = hp.Int('units' , min_value = 10
                                        , max_value = 512 ,
                                        step = 64) ,
                          activation = 'relu' , input_shape = X_ann_train[0].shape))
  
  # adding second layer of our model number of neurons varry from 10 to 512
  model.add(layers.Dense(units = hp.Int('units' , min_value = 10
                                        , max_value = 512 ,
                                        step = 64) ,
                          activation = 'relu'))
  
  # output layer with shape 5 as categorized into 5 category so no need to tune
  model.add(layers.Dense(units=5 , activation = 'softmax'))
  
  # compiling our model with varrying hyperparamters
  model.compile(optimizer = keras.optimizers.Adam(hp.Choice('learning_rate',
                                                            values = [1e-2 , 1e-4 , 1e-3])), 
                loss = 'categorical_crossentropy' , 
                metrics = ['accuracy'])
  return model

tuner = RandomSearch(
    model_test ,
    objective = 'val_accuracy', 
    max_trials=5 , 
    project_name = 'weatherprediction'
)

tuner.search_space_summary()
tuner.search(X_ann_train, y_ann_train, epochs = 100 ,validation_data = (X_ann_test , y_ann_test))

tuner.results_summary()

"""


out of the 5 trial we got max validation accuracy as 84.7%
Best parameters:
Hyperparameters:
units: 10
learning_rate: 0.0
Score: 0.8469945192337036
"""

best_model = tuner.get_best_models()[0]



best_model.fit(X_ann_train, y_ann_train, batch_size=32, epochs = 100)

predicted = best_model.predict(X_ann_test)

testi = [[5 ,5 , 5 , 5]]
testi = np.array(testi)

predicty = best_model.predict(testi)

index = np.argmax(predicty)
from keras.models import load_model
best_model.save('weather.h5')

modeldone = load_model('weather.h5')

predictyy = modeldone.predict(testi)

indexy = np.argmax(predictyy)
print(indexy)





