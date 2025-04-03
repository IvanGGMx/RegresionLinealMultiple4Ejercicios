import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Cargar el dataset
df = pd.read_csv ("data/Salary_dataset2.csv")

# 2. Ver las primeras filas del dataset
print(df.head())

# 3. Definir la variable independiente (X) y la variable dependiente (y)
X = df.iloc[:, 1:2].values  # Selecciona solo la columna "YearsExperience" (matriz 2D)
y = df.iloc[:, -1].values   # Selecciona la columna "Salary" (vector 1D)

# 4. Dividir el dataset en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Verificar las dimensiones para evitar errores
print(f"Shape de X_train: {X_train.shape}, Shape de y_train: {y_train.shape}")

# 6. Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 7. Predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# 8. Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# 9. Visualizar la regresión lineal con los datos de entrenamiento
plt.scatter(X_train.flatten(), y_train, color='blue', label="Datos reales")
plt.plot(X_train.flatten(), modelo.predict(X_train), color='red', label="Línea de regresión")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.title("Regresión Lineal Simple (Entrenamiento)")
plt.legend()
plt.show()

# 10. Visualizar la regresión con los datos de prueba
plt.scatter(X_test.flatten(), y_test, color='green', label="Datos reales (prueba)")
plt.plot(X_train.flatten(), modelo.predict(X_train), color='red', label="Línea de regresión")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.title("Regresión Lineal Simple (Prueba)")
plt.legend()
plt.show()

