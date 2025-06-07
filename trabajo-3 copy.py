import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import  confusion_matrix, accuracy_score, f1_score
#################################   PUNTO 1   #################################  
# === 1. CARGA Y LIMPIEZA DE DATOS ===
def cargar_y_limpiar_datos(url, columnas):
    df = pd.read_csv(url, usecols=columnas, encoding='latin1')
    print(f"[INFO] Datos cargados: {len(df)} filas, columnas: {list(df.columns)}")
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    print(f"[INFO] Datos tras limpieza: {len(df)} filas.")
    return df

# === 2. DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA ===#la buena es la 20
def dividir_entrenamiento_prueba(X, y, prueba_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    n_train = int(len(X) * (1 - prueba_size))
    X_train = X[indices[:n_train]]
    X_test = X[indices[n_train:]]
    y_train = y[indices[:n_train]]
    y_test = y[indices[n_train:]]
    print(f"[INFO] Entrenamiento: {len(X_train)}, Prueba: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# === 3. REGRESIÓN LINEAL MÚLTIPLE ===
def calcular_coeficientes(X, y):
    X_ext = np.c_[np.ones(X.shape[0]), X]
    beta = np.linalg.pinv(X_ext.T @ X_ext) @ X_ext.T @ y
    return beta

def predecir(X, beta):
    X_ext = np.c_[np.ones(X.shape[0]), X]
    return X_ext @ beta

def calcular_r2(y_real, y_pred):
    ss_total = np.sum((y_real - np.mean(y_real)) ** 2)
    ss_residual = np.sum((y_real - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# === 4. MAIN ===
URL = 'https://drive.google.com/uc?export=download&id=1toeE02va0D5F3EqxkMhvz5mp6TADzspg'
COLUMNAS = ['Horas de estudio por semana', 'Calificaciones previas', 'Porcentaje de asistencia', 'Calificación']

df = cargar_y_limpiar_datos(URL, COLUMNAS)

# Entradas (X) y salida (y)
X = df[['Horas de estudio por semana', 'Calificaciones previas', 'Porcentaje de asistencia']].values
y = df['Calificación'].values

# División en entrenamiento/prueba
X_train, X_test, y_train, y_test = dividir_entrenamiento_prueba(X, y)

# Modelo
beta = calcular_coeficientes(X_train, y_train)
print(f"\n[RESULTADO] Coeficientes del modelo (β): {beta}")

# Predicción y R²
y_pred = predecir(X_test, beta)
r2 = calcular_r2(y_test, y_pred)
print(f"[RESULTADO] R² sobre conjunto de prueba: {r2:.4f}")

# Predicción para nuevo estudiante
nuevo_estudiante = np.array([[25, 68, 58]])  # horas, examen previo, asistencia
prediccion = predecir(nuevo_estudiante, beta)
print(f"[RESULTADO] Predicción para nuevo estudiante: {prediccion[0]:.2f}\n")
 
from sklearn.metrics import confusion_matrix

# Convertimos a clasificación binaria
y_test_clasificado = (y_test >= 60).astype(int)
y_pred_clasificado = (y_pred >= 60).astype(int)
print("y_test",y_test)
print("y_test_clasificado",y_test_clasificado)


# Matriz de confusión
matriz = confusion_matrix(y_test_clasificado, y_pred_clasificado)
print("\n[RESULTADO] Matriz de confusión:")
print(matriz)

accuracy = accuracy_score(y_test_clasificado, y_pred_clasificado)
print(f"[RESULTADO] Accuracy: {accuracy:.4f}")

# F1-Score
f1 = f1_score(y_test_clasificado, y_pred_clasificado)
print(f"[RESULTADO] F1 Score: {f1:.4f}")


#################################   PUNTO 2   #################################  





################## DE ACA ABAJO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC

#################################   EJERCICIO 2 - SVM   #################################



# Convertir a clasificación binaria
y_train_clasificado = (y_train >= 60).astype(int) 
#y_test_clasificado = (y_test >= 60).astype(int)

# Probar diferentes kernels y valores de C
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
C_values = [0.1, 1, 10]

for kernel in kernels:
    for C in C_values:
        print(f"\n[MODELO SVM] Kernel: {kernel}, C: {C}")
        modelo_svm = SVC(kernel=kernel, C=C)
        modelo_svm.fit(X_train, y_train_clasificado)
        y_pred_svm = modelo_svm.predict(X_test)

        matriz = confusion_matrix(y_test_clasificado, y_pred_svm)
        print("[RESULTADO] Matriz de confusión:")
        print(matriz)

        accuracy = accuracy_score(y_test_clasificado, y_pred_svm)
        f1 = f1_score(y_test_clasificado, y_pred_svm)
        print(f"[RESULTADO] Accuracy: {accuracy:.4f}")
        print(f"[RESULTADO] F1 Score: {f1:.4f}")

# Predicción para nuevo estudiante usando un modelo elegido (ejemplo: RBF con C=1)
modelo_final = SVC(kernel='rbf', C=1)
modelo_final.fit(X_train, y_train_clasificado)

nuevo_estudiante = np.array([[25, 68, 58]])
prediccion = modelo_final.predict(nuevo_estudiante)
estado = "Aprobado" if prediccion[0] == 1 else "Desaprobado"
print(f"\n[RESULTADO] Condición de aprobación del nuevo estudiante: {estado}")