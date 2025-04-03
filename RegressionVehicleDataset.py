import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from pandastable import Table
from tkinter import Frame
from sklearn.model_selection import train_test_split  # Asegúrate de importar esto
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar los datasets
df1 = pd.read_csv ("data/EcommerceCustomers.csv")
df2 = pd.read_csv('data/car data.csv')  # Reemplaza con el nombre correcto
df3 = pd.read_csv('data/Car details v3.csv')  # Reemplaza con el nombre correcto
df4 = pd.read_csv('data/car details v4.csv')  # Reemplaza con el nombre correcto

# Seleccionar el dataset correcto
df = df2[['Year', 'Selling_Price', 'Kms_Driven']]  # Ajustado al dataset correcto
df.dropna(inplace=True)  # Eliminar valores nulos

# Definir variables independientes (X) y dependiente (y)
X = df[['Year', 'Kms_Driven']]
y = df['Selling_Price']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualización de resultados
plt.figure(figsize=(12, 5))

# Gráfico de dispersión real vs predicho
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Selling Price')
plt.grid(True)

# Histograma de errores
errors = y_test - y_pred
plt.subplot(1, 2, 2)
plt.hist(errors, bins=20, color='red', edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.grid(True)

plt.tight_layout()
plt.show()

# Función para mostrar los dataframes en una ventana aparte
def show_dataframes():
    def show_table(dataframe, title):
        root = tk.Tk()
        root.title(title)

        # Crear un frame para los dataframes
        frame = Frame(root)
        frame.pack(fill="both", expand=True)

        # Crear la tabla de PandasTable
        table = Table(frame, dataframe=dataframe)

        # Mostrar la tabla
        table.show()

        # Ejecutar la ventana
        root.mainloop()

    # Llamar a la función para mostrar cada DataFrame en su propia ventana
    show_table(df1, "Dataset 1 (df1)")
    show_table(df2, "Dataset 2 (df2)")
    show_table(df3, "Dataset 3 (df3)")
    show_table(df4, "Dataset 4 (df4)")

# Llamar a la función para mostrar los dataframes
show_dataframes()


