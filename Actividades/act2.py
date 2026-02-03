""" Aprendizaje Maquina  D05  2026A
-------------------------------------------
Aurelio Rendon Viciego      - 220584335 
Juan Carlos Reynoso Ortega  - 220563354
Christian Luevano Zaragoza  - 220468661
"""

#--------------IMPORTACIONES NECESARIAS--------------
import numpy as np # Manejo de arreglos numericos
import pandas as pd # Tablas de datos
import matplotlib.pyplot as plt # Visualizaciones de datos
from sklearn.linear_model import LinearRegression # Modelo de regresion lineal
from sklearn.model_selection import train_test_split # Particion de datos



#--------------DEFINICION DE SET DE DATOS--------------
# Definimos el set de datos
data = {'Años de experiencia'   : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        'Salario Real(MIL MXN)' : [12,14,17,19,22,25,27,30,32,35,42,44,46,48,51,55]}
# Formamos un dataframe de dos columnas con pandas
df = pd.DataFrame(data, columns=['Años de experiencia', 'Salario Real(MIL MXN)'])
# Definimos 'X' (variables explicativas/independientes)
X = df.drop(columns='Salario Real(MIL MXN)')
# Definimos 'y' (variables de respuesta/dependientes)
y = df['Salario Real(MIL MXN)']
# Separamos la data de prueba y la de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#--------------ENTRENAMIENTO Y PRUEBA DEL MODELO--------------
# Inicializar y entrenar el modelo con la data de entrenamiento
modelo = LinearRegression()
modelo.fit(X_train, y_train)
# Evaluar el modelo con los datos de prueba
score = modelo.score(X_test, y_test)
print(f"Puntuacion del modelo: {score}")



#--------------GRAFICA----------------
# Graficamos los puntos de ENTRENAMIENTO y PRUEBAS
plt.scatter(X_train, y_train, color='blue', label='Entrenamiento')
plt.scatter(X_test, y_test, color='red', label='Prueba')
# Graficamos la linea de prediccion del modelo
# Usamos los datos de prueba para ver por donde pasa la linea que aprendio el modelo
y_prediccion = modelo.predict(X_test)
plt.plot(X_test, y_prediccion, color='black', linewidth=2, label='Linea de Regresion')
plt.title('Modelo de regresion lineal')
plt.xlabel('Años de experiencia')
plt.ylabel('Salario Real (MIL MXN)')
plt.legend() 
plt.show()
