import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, f1_score

# === 1. CARGA Y LIMPIEZA DE DATOS ===
def cargar_y_limpiar_datos(url, columnas):
    df = pd.read_csv(url, usecols=columnas, encoding='latin1')
    print(f"[INFO] Datos cargados: {len(df)} filas, columnas: {list(df.columns)}")
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    print(f"[INFO] Datos tras limpieza: {len(df)} filas.")
    return df

# === 2. DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA ===
def dividir_entrenamiento_prueba(X, y, prueba_size=0.2, random_state=20):
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
print(f"[RESULTADO] Predicción para nuevo estudiante: {prediccion[0]:.2f}")

# Matriz de confusión
matriz = confusion_matrix(y_test_clf, y_pred_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print("\nMatriz de confusión:")
print(matriz)
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")

# 4. Predicción para nuevo estudiante
nuevo_estudiante = np.array([[25, 58, 68]])  # 25 horas, 58% asistencia, 68 nota previa

# prediccion_lineal = modelo_lineal.predict(nuevo_estudiante)[0]
# prediccion_logistica = modelo_logistico.predict(nuevo_estudiante)[0]
# proba_aprobado = modelo_logistico.predict_proba(nuevo_estudiante)[0][1]

print("\nPredicciones para nuevo estudiante:")
print(f"Calificación estimada: {prediccion_lineal:.2f}")
print(f"¿Aprobado? {'Sí' if prediccion_logistica == 1 else 'No'} (probabilidad: {proba_aprobado:.2f})")
