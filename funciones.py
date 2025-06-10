import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
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
    if(kernel==""):
        barraTitulo=f"Reg.Log."
    else:    
        barraTitulo=f"{kernel}{titulo2}{C}"
    metrics = {'Accuracy': accuracy, 'F1 Score': f1}

    sns.set_style("whitegrid")
    plt.figure(figsize=(5, 4))
    sns.barplot(
    x=list(metrics.keys()),
    y=list(metrics.values()),
    hue=list(metrics.keys()),  # assign x variable to hue
    palette="viridis",
    legend=False               # avoid duplicate legend
)

   # sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
    plt.title(titulo)
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    plt.show()
    return((f1,accuracy,barraTitulo))


def graficoDeBarrasF1yAc(array_de_metricas):
 

    

    df = pd.DataFrame(array_de_metricas, columns=['F1 Score', 'Accuracy', 'Modelo'])
    
    # Convertir a formato largo para Seaborn
    df_largo = pd.melt(df, id_vars='Modelo', var_name='Métrica', value_name='Valor')

    # Configuración de estilo
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Crear gráfico de barras
    ax = sns.barplot(data=df_largo, x='Modelo', y='Valor', hue='Métrica', palette='viridis')

    # Agregar etiquetas numéricas a cada barra
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}', 
                    (p.get_x() + p.get_width() / 2., height + 0.01), 
                    ha='center', va='center', fontsize=9)

    # Mejoras visuales
    plt.title('Comparación de Accuracy y F1 Score por Modelo')
    plt.ylim(0, 1.1)
    plt.ylabel('Valor')
    plt.xticks(rotation=15)
    plt.legend(title='Métrica')
    plt.tight_layout()
    plt.show()