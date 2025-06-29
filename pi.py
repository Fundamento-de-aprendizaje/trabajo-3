import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    auc
)

################################# PUNTO 1: CARGA Y PREPROCESAMIENTO #################################
URL = 'https://drive.google.com/uc?export=download&id=1toeE02va0D5F3EqxkMhvz5mp6TADzspg'
COLUMNAS = [
    'Horas de estudio por semana',
    'Calificaciones previas',
    'Porcentaje de asistencia',
    'Calificación'
]

# Carga y limpieza
df = (
    pd.read_csv(URL, usecols=COLUMNAS, encoding='latin1')
      .dropna()
      .apply(pd.to_numeric, errors='coerce')
      .dropna()
)

# Variables de entrada y salida
X = df[['Horas de estudio por semana',
        'Calificaciones previas',
        'Porcentaje de asistencia']].values
y_reg = df['Calificación'].values
y_clas = (df['Calificación'] >= 60).astype(int).values  # 1 = aprobado, 0 = desaprobado

################################# PUNTO 1.2: DIVISIÓN #################################
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg,
    test_size=0.2,
    random_state=42
)
_, _, y_train_clas, y_test_clas = train_test_split(
    X, y_clas,
    test_size=0.2,
    random_state=42,
    stratify=y_clas
)
print(f"Entrenamiento: {len(X_train)}, Prueba: {len(X_test)}")

################################# PUNTO 1.3: REGRESIÓN LINEAL MÚLTIPLE #################################
# Construcción de X con columna de 1s
X_ext = np.c_[np.ones(X_train.shape[0]), X_train]
beta = np.linalg.pinv(X_ext.T @ X_ext) @ X_ext.T @ y_reg_train

# Mostrar ecuación
intercepto, *coef = beta
vars_ = ['Horas', 'Previas', 'Asistencia']
ecuacion = f"y = {intercepto:.4f}"
for v, c in zip(vars_, coef):
    signo = '+' if c >= 0 else '-'
    ecuacion += f" {signo} {abs(c):.4f}*{v}"
print("\nEcuación regresión lineal múltiple:")
print(ecuacion)

# Predicción y R²
X_test_ext = np.c_[np.ones(X_test.shape[0]), X_test]
y_pred_reg = X_test_ext @ beta
r2 = 1 - (np.sum((y_reg_test - y_pred_reg)**2) /
          np.sum((y_reg_test - y_reg_test.mean())**2))
print(f"R² en prueba: {r2:.4f}")

# Predicción para nuevo estudiante
nuevo = np.array([[25, 68, 58]])
pred_nuevo = np.r_[1, nuevo.flatten()] @ beta
estado = "Aprobado" if pred_nuevo >= 60 else "No aprobado"
print(f"Predicción (lineal) para [25h,68,58%]: {pred_nuevo:.2f} → {estado}")

################################# PUNTO 1.4: REGRESIÓN LOGÍSTICA #################################
modelo_log = LogisticRegression()
modelo_log.fit(X_train, y_train_clas)

# Ecuación logit
i0 = modelo_log.intercept_[0]
cfs = modelo_log.coef_[0]
terms = " + ".join([f"{c:.4f}·x{i+1}" for i, c in enumerate(cfs)])
print(f"\nLogit(p) = {i0:.4f} + {terms}")

# Evaluación
y_pred_log = modelo_log.predict(X_test)
print("\nMatriz de confusión (Logística):")
print(confusion_matrix(y_test_clas, y_pred_log))
print("Accuracy:", accuracy_score(y_test_clas, y_pred_log))
print("F1‑Score:", f1_score(y_test_clas, y_pred_log))
print("\nReporte de clasificación:")
print(classification_report(y_test_clas, y_pred_log))

# Probabilidad y predicción nuevo estudiante
prob_nuevo = modelo_log.predict_proba(nuevo)[:,1][0]
pred_clas_nuevo = modelo_log.predict(nuevo)[0]
print(f"Nuevo estudiante (logístico): prob aprueba={prob_nuevo:.3f}, predicción={'Aprobado' if pred_clas_nuevo else 'Desaprobado'}")

################################# EJERCICIO 2: SVM – ESTUDIO DE HIPERPARÁMETROS #################################
# Pipeline con escalado y SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# Definición de malla de hiperparámetros
param_grid = {
    'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svm__C':      [0.01, 0.1, 1, 10, 100],
    'svm__gamma':  ['scale', 'auto', 0.001, 0.01, 0.1],
    'svm__degree': [2, 3, 4]   # sólo para kernel='poly'
}

# GridSearchCV con 5‑fold CV, optimizando F1
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=['accuracy', 'f1'],
    refit='f1',
    cv=5,
    n_jobs=-1,
    return_train_score=True
)
print("\nIniciando GridSearchCV para SVM...")
grid.fit(X_train, y_train_clas)

# Resultados de la búsqueda
print("\n-- Mejores parámetros SVM --")
for k, v in grid.best_params_.items():
    print(f" {k}: {v}")
print(f"Mejor F1 (CV): {grid.best_score_:.4f}")

# Evaluación final en test
mejor_svm = grid.best_estimator_
y_pred_svm = mejor_svm.predict(X_test)
y_prob_svm = mejor_svm.predict_proba(X_test)[:,1]

print("\n-- Resultados SVM en test --")
print("Accuracy:", accuracy_score(y_test_clas, y_pred_svm))
print("F1‑Score:", f1_score(y_test_clas, y_pred_svm))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test_clas, y_pred_svm))
print("\nReporte de clasificación:")
print(classification_report(y_test_clas, y_pred_svm))

# Curva ROC y AUC
fpr, tpr, _ = roc_curve(y_test_clas, y_prob_svm)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'k--',alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC – Mejor SVM')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
