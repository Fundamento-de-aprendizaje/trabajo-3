import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from funciones import imprimirMatriz,visualizarAcyF1,graficoDeBarrasF1yAc
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

#################################   PUNTO 1   #################################  
# === 1. CARGA Y LIMPIEZA DE DATOS ===
def cargar_y_limpiar_datos(url, columnas):
    df = pd.read_csv(url, usecols=columnas, encoding='latin1')
    print(f"Datos cargados: {len(df)} filas, columnas: {list(df.columns)}")
    df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    print(f"Datos tras limpieza: {len(df)} filas.")
    return df

# === 2. DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA ===#la buena es la 20
def dividir_entrenamiento_prueba(X, y, prueba_size=0.2, random_state=20):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    n_train = int(len(X) * (1 - prueba_size))
    X_train = X[indices[:n_train]]
    X_test = X[indices[n_train:]]
    y_train = y[indices[:n_train]]
    y_test = y[indices[n_train:]]
    print(f"Entrenamiento: {len(X_train)}, Prueba: {len(X_test)}")
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
print("\nEJERCICIO n°1 ")
# Modelo
beta = calcular_coeficientes(X_train, y_train)
print(f"\n Coeficientes del modelo (β): {beta}")

# Mostrar la ecuación del modelo de regresión lineal múltiple
intercepto = beta[0]
coeficientes = beta[1:]
variables = ['Horas de estudio por semana', 'Calificaciones previas', 'Porcentaje de asistencia']

ecuacion = f"y = {intercepto:.4f}"
for i, coef in enumerate(coeficientes):
    signo = '+' if coef >= 0 else '-'
    ecuacion += f" {signo} {abs(coef):.4f} * {variables[i]}"
print(f"\nEcuación del modelo de regresión lineal múltiple:")
print(ecuacion)


# Predicción y R²
y_pred = predecir(X_test, beta)
r2 = calcular_r2(y_test, y_pred)
print(f"R² sobre conjunto de prueba: {r2:.4f}")

# Predicción para nuevo estudiante
nuevo_estudiante = np.array([[25, 68, 58]])  # horas, examen previo, asistencia
prediccion = predecir(nuevo_estudiante, beta)
print(f"Calificacion esperada: {prediccion[0]:.2f}\n")
aprobado = int(prediccion[0] >= 60)
estado = "Aprobado" if aprobado == 1 else "No aprobado"
print(f"Condicion de aprobacion: {estado}")

# === REGRESIÓN LOGÍSTICA ===
print("EJERCICIO REGRESIÓN LOGÍSTICA")

# Convertimos a clasificación binaria (umbral 60 para aprobado)
y_train_clasificado = (y_train >= 60).astype(int)
y_test_clasificado = (y_test >= 60).astype(int)

# Crear y entrenar modelo de regresión logística
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_train, y_train_clasificado)

# Imprimir ecuación del modelo (logit)
intercepto = modelo_logistico.intercept_[0]
coeficientes = modelo_logistico.coef_[0]
print(f"Ecuación del modelo de regresión logística:")
print(f"logit(p) = {intercepto:.4f} + " +
      " + ".join([f"({coef:.4f} * x{i+1})" for i, coef in enumerate(coeficientes)]))

# Predicciones y evaluación
y_pred_logistico = modelo_logistico.predict(X_test)

# Matriz de confusión
print("\nMatriz de Confusión - Regresión Logística")
imprimirMatriz(y_test_clasificado, y_pred_logistico, 'Matriz de Confusión - Regresión Logística')



# Puedes agregar las métricas al arreglo para graficar si quieres
array_de_metricas = []
array_de_metricas=array_de_metricas+[visualizarAcyF1(y_test_clasificado,  y_pred_logistico,'Reg Log')]


prediccion_logistica = modelo_logistico.predict(nuevo_estudiante)
print(f"Condicion de aprobacion: {'Aprobado' if (prediccion_logistica[0] == 1) else 'No Aprobado'}")
probabilidad_logistica = modelo_logistico.predict_proba(nuevo_estudiante)[:, 1]
print(f"Probabilidad de aprobar: {probabilidad_logistica[0]:.4f}\n")



#################################   EJERCICIO 2 - SVM   #################################
print("EJERCICIO n°2 ")
# Convertir a clasificación binaria
y_train_clasificado = (y_train >= 60).astype(int) 

######################////////////////////////   **************************/////////////////
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train) # Aplica PCA al set de entrenamiento (ajuste + transformación)
X_test_2D = pca.transform(X_test)   # Transforma el set de prueba usando el mismo PCA

# Probar diferentes kernels y valores de C
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
C_values = [0.1, 1, 10]

for kernel in kernels:
    for C in C_values:
        
        print(f"\n[MODELO SVM] \nKernel: {kernel}, C: {C}")
        modelo_svm = SVC(kernel=kernel, C=C) # Crear modelo con kernel y C actual
        modelo_svm.fit(X_train, y_train_clasificado) # Entrenar modelo con los datos de entenemiento
        y_pred_svm = modelo_svm.predict(X_test)  # Realiza predicciones sobre los datos de prueba 

        # Calcula la matriz de confusión comparando las predicciones con las verdaderas etiquetas
          # === GRAFICAR MATRIZ DE CONFUSIÓN ===
        imprimirMatriz(y_test_clasificado, y_pred_svm,'Matriz de Confusión\nKernel:','C:',kernel,C)#nuevo

        # Calcula la precisión (accuracy) del modelo
        array_de_metricas=array_de_metricas+[visualizarAcyF1(y_test_clasificado, y_pred_svm,'Metrica de Modelo Kernel:','C:',kernel,C)]
       # visualizarAcyF1(y_test_clasificado, y_pred_svm,'Metrica de Modelo Kernel:','C:',kernel,C)
        
        

        # ===graficar frontera de decision SVM ===
        #Se vuelve a entrenar el modelo usando los datos reducidos 
        # a 2 dimensiones con PCA, para graficar las fronteras de decisión.

        modelo = SVC(kernel=kernel, C=C)
        modelo.fit(X_train_2D, y_train_clasificado)
        y_pred = modelo.predict(X_test_2D)

        # Crear malla para graficar fronteras
        x_min, x_max = X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1
        y_min, y_max = X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                            np.linspace(y_min, y_max, 500))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = modelo.predict(grid).reshape(xx.shape)

        # Graficar
        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, cmap=plt.cm.Greens, alpha=0.4)# Rellena las regiones según clase predicha
        plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=y_test_clasificado, cmap=plt.cm.bwr, edgecolors='k')# Dibuja puntos reales
        plt.title(f"Frontera de decisión SVM (kernel='{kernel}')(C='{C})")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.tight_layout()
        plt.show()

gammas = [0.01, 0.1, 1]
C_values = [0.1, 1, 10]

# PCA para reducción de dimensionalidad a 2D
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
X_test_2D = pca.transform(X_test)

for C in C_values:
    for gamma in gammas:
        print(f"\n[MODELO SVM] Kernel=rbf | C={C} | gamma={gamma}")
        
        # Entrenar modelo
        modelo = SVC(kernel='rbf', C=C, gamma=gamma)
        modelo.fit(X_train_2D, y_train_clasificado)
        
        # Predicciones
        y_pred = modelo.predict(X_test_2D)
        
        # === Graficar frontera de decisión ===
        x_min, x_max = X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1
        y_min, y_max = X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = modelo.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.4)
        plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=y_test_clasificado, cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
        plt.title(f"SVM (kernel='rbf') | C={C} | gamma={gamma}")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.tight_layout()
        plt.show()       

graficoDeBarrasF1yAc(array_de_metricas)
# Predicción para nuevo estudiante usando un modelo elegido (ejemplo: RBF con C=1)
modelo_final = SVC(kernel='rbf', C=10)
modelo_final.fit(X_train, y_train_clasificado)

nuevo_estudiante = np.array([[25, 68, 58]])
prediccion = modelo_final.predict(nuevo_estudiante)
estado = "Aprobado" if prediccion[0] == 1 else "Desaprobado"
print(f"\nCondición de aprobación del nuevo estudiante: {estado}")

