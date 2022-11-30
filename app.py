import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import streamlit as st

X=pd.read_csv("datasets/traffic_signs_x_train_gr_smpl.csv",encoding='utf-8')
y=pd.read_csv("datasets/traffic_signs_y_train_smpl.csv",encoding='utf-8')

X = X / 255.0

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

# Declaramos la loss function de tipo categorical ya que tenemos 10 diferentes clases
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

# Compilamos 
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Hacemos el fit
model.fit(x_train, y_train, epochs=5)

# Evaluaremos y aplicamos el softmax
model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# Ahora veremos que nuestro programa si funciona
num_registro=3
print("Tenemos que nuestro programa clasificará los siguientes " + str(num_registro)+" registros, los cuales verdaderamente son: "+str(y_test[:num_registro])) 

print(probability_model(x_test[:num_registro]))