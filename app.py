import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# leer el dataset 
X=pd.read_csv("datasets/traffic_signs_x_train_gr_smpl.csv",encoding='utf-8')
y=pd.read_csv("datasets/traffic_signs_y_train_smpl.csv",encoding='utf-8')

# normalizar
X = X / 255.0

# dividimos el dataset en parte de entrenamiento y parte de teseo
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# lo convertimos a arreglo
x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

# hacemos el modelo, diciendole que la funcion de activación será relu
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# hacemos nuestras predicciones
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
print("Tenemos que nuestro programa clasificará los siguientes " + str(num_registro)+" registros, los cuales verdaderamente son: "+str(y_test[:num_registro])+" donde tenemos las siguientes categorías \n 0. speed limit 60, \n 1. speed limit 80,\n 2. speed limit 80 lifted,\n 3. right of way at crossing,\n 4. right of way in general,\n 5. give way,\n 6. Stop,\n 7. no speed limit general,\n 8. turn right down,\n 9. turn left down.") 

print(probability_model(x_test[:num_registro]))