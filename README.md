# Importar las librerías necesarias
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paso 0: Cargar el archivo CSV y renombrar columnas
file_path = 'D:\Trabajos Universidad\7mo semestre\inteligencia artificial\hito 4\iris.csv'
iris_data = pd.read_csv(file_path, header=None)

# Renombrar las columnas para mayor claridad
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Guardar el archivo con los nombres de columnas corregidos
iris_data.to_csv(file_path, index=False)

# Paso 1: Preprocesamiento de datos
# Convertir las etiquetas de las especies en valores numéricos
encoder = LabelEncoder()
iris_data['species'] = encoder.fit_transform(iris_data['species'])

# Separar las características (X) y las etiquetas (y)
X = iris_data.iloc[:, :-1].values
y = iris_data['species'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 2: Construcción del modelo
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Paso 3: Compilación del modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Paso 4: Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.2, verbose=1)

# Paso 5: Evaluación del modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Pérdida en el conjunto de prueba:", test_loss)
print("Precisión en el conjunto de prueba:", test_accuracy)

# Paso 6: Predicción con nuevos datos
def predecir_tipo_de_flor(modelo, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = modelo.predict(input_data)
    predicted_class = np.argmax(prediction)
    species_name = encoder.inverse_transform([predicted_class])[0]
    return species_name

# Ejemplo de uso de la función de predicción
nuevas_caracteristicas = [5.1, 3.5, 1.4, 0.2]  # Reemplaza con cualquier valor deseado
resultado = predecir_tipo_de_flor(model, nuevas_caracteristicas)
print("La clase de flor predicha es:", resultado)
