import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import streamlit as st

# Descargamos el dataset 
X=pd.read_csv("datasets/traffic_signs_x_train_gr_smpl.csv",encoding='utf-8')
y=pd.read_csv("datasets/traffic_signs_y_train_smpl.csv",encoding='utf-8')


# Normalizamos
X = X / 255.0

# Asigna dos tipos de sub datasets, una para entrenamiento y otra para pruebas
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

print(type(x_train))
# Declara la secuencia de nuestro modelo
# 1. Aplana los datos de entrada
# 2. Crea una capa de neuronas con una función de activación relu
# 3. Asigna valores de forma aleatoria para prevenir el efecto overfitting
# 4. Crea una capa de salida
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Realiza predicciones de los valores de entrenamiento
predictions = model(x_train[:1]).numpy()
predictions


tf.nn.softmax(predictions).numpy()

# Declaramos la loss function de tipo categorical ya que tenemos 10 
# diferentes tipos de clases a identificar
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

# Compilamos el modelo, con un optimizador adam y obtenemos 
# la métrica de precisión
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Entrenamos el modelo
model.fit(x_train, y_train, epochs=5)

# Evaluamos el modelo utilizando el dataset para pruebas
model.evaluate(x_test,  y_test, verbose=2)

# Terminamos de construir el modelo
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

# Obtenemos las probabilidades de las categorias para el primer numero
# escrito a mano 
print(probability_model(x_test[:1]))
