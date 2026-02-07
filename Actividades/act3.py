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
from mpl_toolkits.mplot3d import Axes3D # Grafica 3D
from sklearn.metrics import mean_squared_error, r2_score # Metricas del modelo

#--------------DEFINICION DE SET DE DATOS--------------
# Definimos el set de datos
data = {'Horas de estudio'   : [5,20,5,15,10,8,12,18,6,14,9,16,7,11,13,4,17,19,3,10,6,14,8,12,15,7,9,11,13,16],
        'Tareas realizadas'   : [8,2,9,11,10,7,10,3,8,12,9,5,6,11,8,10,4,1,12,7,9,6,8,10,5,11,7,9,12,4],
        'Calificacion real' : [10,30,20,25,15,18,22,28,12,24,17,26,14,23,21,8,27,32,6,19,13,25,16,20,29,15,18,12,24,31]
        }

# Formamos un dataframe de dos columnas con pandas
df = pd.DataFrame(data, columns=['Horas de estudio', 'Tareas realizadas', 'Calificacion real'])
# Definimos 'X' (variables explicativas/independientes)
X = df.drop(columns='Calificacion real')
# Definimos 'y' (variables de respuesta/dependientes)
y = df['Calificacion real']
# Separamos la data de prueba y la de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#--------------ENTRENAMIENTO Y PRUEBA DEL MODELO--------------
# Inicializar y entrenar el modelo con la data de entrenamiento
modelo = LinearRegression()
modelo.fit(X_train, y_train)


#--------------CALCULAR SUPERFICIE DE REGRESION----------------
x1_surf = np.linspace(df['Horas de estudio'].min(), df['Horas de estudio'].max())
x2_surf = np.linspace(df['Tareas realizadas'].min(), df['Tareas realizadas'].max())
X1_surf, X2_surf = np.meshgrid(x1_surf, x2_surf)
Y_surf = modelo.intercept_ + modelo.coef_[0] * X1_surf + modelo.coef_[1] * X2_surf # y = β0 ​+ β1​x1 ​+ β2​x2​

#--------------GRAFICA----------------
# Graficamos los puntos de ENTRENAMIENTO y PRUEBAS
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1_surf, X2_surf, Y_surf, color='red', alpha=0.3) # Superficie del plano de regresion
ax.scatter(X_train['Horas de estudio'], X_train['Tareas realizadas'], y_train, color='blue', label='Entrenamiento')
ax.scatter(X_test['Horas de estudio'], X_test['Tareas realizadas'], y_test, color='red', label='Prueba')
ax.set_xlabel('Horas de estudio')
ax.set_ylabel('Tareas realizadas')
ax.set_zlabel('Calificacion real')
ax.set_title('Modelo de regresion lineal multiple')
plt.legend() 
plt.show()


#--------------RESULTADOS----------------

all_y_pred = modelo.predict(X) # Predecir con el modelo todos los datos
y_pred = modelo.predict(X_test) # Predeci con el modelo los datos de prueba

# Extraemos los coeficientes y el intercepto
b0 = modelo.intercept_
b1 = modelo.coef_[0]
b2 = modelo.coef_[1]
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print(pd.DataFrame({ # Armar el dataframe de resultados
                'Horas de estudio': data['Horas de estudio'],
                'Tareas realizadas': data['Tareas realizadas'],
                'Calificacion real': data['Calificacion real'], 
                'Calificacion predicha' : all_y_pred,
                'Diferencia': data['Calificacion real'] - all_y_pred
                   }))

# Impresion de valores del modelo
print(f'\n -------- VALORES DEL MODELO -------- \n Intercepto: {b0} \n Peso de "Estudio": {b1} \n Peso de "Tareas": {b2} \n Ecuacion: "Calificación = {b0} + ({b1} * Horas) + ({b2} * Tareas)" \n Error cuadratico medio : {mse} \n Raiz de error cuadratico medio: {mse ** (1/2)} \n Coeficiente de determinacion: {score}')

