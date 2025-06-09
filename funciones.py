import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import seaborn as sns


def imprimirMatriz(y_test_clasificado, y_pred_clasificado,titulo1,titulo2="",kernel="",C=""):
    matriz = confusion_matrix(y_test_clasificado, y_pred_clasificado)
    print(" Matriz de confusión:")
    print(matriz)
   
    titulo= f"{titulo1}{kernel}{titulo2}{C}"
    # === GRAFICAR MATRIZ DE CONFUSIÓN ===

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz,
        annot=True,               # Mostrar los números
        fmt='d',                  # Formato decimal
        cmap='Blues',             # Paleta de colores
        cbar=False,               # Quitar barra de color
        annot_kws={"size": 60}    # Cambiar el tamaño de fuente de los números
    )
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Real', fontsize=14)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()



def visualizarAcyF1(y_true, y_pred,titulo1,titulo2="",kernel="",C=""):
    # Métricas
    # Calcula la precisión (accuracy) del modelo
    accuracy = accuracy_score(y_true, y_pred)
   
    # Calcula el F1 Score, una métrica que combina precisión y exhaustividad
    f1 = f1_score(y_true, y_pred)

    # Calcula la precisión (accuracy) del modelo
   
    print(f" Accuracy: {accuracy:.4f}")
    print(f" F1 Score: {f1:.4f}")
    titulo= f"{titulo1}{kernel}{titulo2}{C}"
    metrics = {'Accuracy': accuracy, 'F1 Score': f1}

    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
    plt.title(titulo)
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    plt.show()
