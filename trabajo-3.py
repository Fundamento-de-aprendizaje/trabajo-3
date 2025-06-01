import pandas as pd
import numpy as np

def cargar_datos(url, columnas, codificacion='latin1'):
    """
    Carga un CSV desde una URL y devuelve un DataFrame con las columnas especificadas.
    """  # Carga el archivo CSV con las columnas y codificación especificadas
    df = pd.read_csv(url, usecols=columnas, encoding=codificacion) 
         # Muestra información sobre los datos cargados
    print(f"[cargar_datos] Datos cargados con {len(df)} filas y {len(df.columns)} columnas.")  
    return df  # Devuelve el DataFrame cargado


# EVALUAR SI VA

def preprocesar_categoricas(df, columna_target):
    """
    Selecciona solo columnas categóricas y elimina filas sin valor en la variable target.
    """
    # Selecciona columnas categóricas y elimina filas con valores nulos en la columna objetivo
    categ = df.select_dtypes(include='object').dropna(subset=[columna_target]) 
    # Muestra las columnas categóricas y el número de filas restantes 
    print(f"[preprocesar_categoricas] Columnas categóricas: {list(categ.columns)}. Filas tras dropna: {len(categ)}.")  
    # Devuelve el DataFrame procesado
    return categ  


def dividir_entrenamiento_prueba(df, prueba_size=0.2, random_state=42):
    """
    Mezcla aleatoriamente el DataFrame y lo divide en entrenamiento y prueba según prueba_size.
    """
    # Mezcla aleatoriamente las filas del DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  
    # Calcula el índice para dividir el DataFrame
    idx = int(len(df_shuffled)*(1-prueba_size))  
    # Selecciona las filas para el conjunto de entrenamiento
    entrenamiento = df_shuffled.iloc[:idx]  
    # Selecciona las filas para el conjunto de prueba
    prueba  = df_shuffled.iloc[idx:]  
    # Muestra el tamaño de los conjuntos
    print(f"Entrenamiento:{len(entrenamiento)} filas, Prueba:{len(prueba)} filas.") 
    # Devuelve los conjuntos de entrenamiento y prueba
    return entrenamiento, prueba  

     
URL = 'https://drive.google.com/uc?export=download&id=1toeE02va0D5F3EqxkMhvz5mp6TADzspg'
COLS = [1,2,3,8]
TARGET = 'Condición'
df = cargar_datos(URL, COLS)

# # df = preprocesar_categoricas(df, TARGET)

print(f"Datos preprocesados:\n{df}")
# entrenamiento, prueba = dividir_entrenamiento_prueba(df)

# print(f"Conjunto de entrenamiento:\n{entrenamiento}")
# print(f"Conjunto de prueba:\n{prueba}")

#import pandas as pd


# === 1. CARGA DE DATOS ===
def cargar_datos(url, columnas, codificacion='latin1'):
    df = pd.read_csv(url, usecols=columnas, encoding=codificacion)
    print(f"[cargar_datos] Datos cargados con {len(df)} filas y {len(df.columns)} columnas.")
    return df

# === 2. LIMPIEZA ===
def limpiar_datos(df):
    df = df.dropna()  # Elimina filas con valores nulos
    df = df.apply(pd.to_numeric, errors='coerce')  # Convierte todo a numérico
    df = df.dropna()  # Elimina filas con errores de conversión
    return df

# === 3. DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA ===
def dividir_entrenamiento_prueba(df, prueba_size=0.2, random_state=42):
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    idx = int(len(df_shuffled)*(1 - prueba_size))
    entrenamiento = df_shuffled.iloc[:idx]
    prueba = df_shuffled.iloc[idx:]
    print(f"[dividir_entrenamiento_prueba] Entrenamiento: {len(entrenamiento)}, Prueba: {len(prueba)}")
    return entrenamiento, prueba

# === 4. REGRESIÓN LINEAL MÚLTIPLE ===
def calcular_coeficientes(X, y):
    # Agregamos columna de 1s para el término independiente
    X = np.c_[np.ones(X.shape[0]), X]
    # Fórmula de mínimos cuadrados: (XᵗX)^(-1) Xᵗy
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

def predecir(X, beta):
    X = np.c_[np.ones(X.shape[0]), X]# Añade una columna de unos (el término de sesgo/intercepto)
    return X @ beta  # Producto matricial: se calcula la predicción

def calcular_r2(y_real, y_pred):
    ss_total = np.sum((y_real - np.mean(y_real)) ** 2)
    ss_residual = np.sum((y_real - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# === 5. MAIN ===
URL = 'https://drive.google.com/uc?export=download&id=1toeE02va0D5F3EqxkMhvz5mp6TADzspg'
COLS = [1,2,3,8]
# URL = 'https://drive.google.com/uc?export=download&id=1toeE02va0D5F3EqxkMhvz5mp6TADzspg'
# COLUMNAS = ['Study hours per week', 'Previous exam score', 'Attendance', 'Performance score']

# Cargar y preparar los datos
df = cargar_datos(URL, COLS)
df = limpiar_datos(df)

# Separar variables predictoras (X) y variable objetivo (y)
X = df[['Horas de estudio por semana', 'Porcentaje de asistencia', 'Calificaciones previas']].values
y = df['Condición'].values

# Dividir en entrenamiento y prueba
X_train, X_test = dividir_entrenamiento_prueba(pd.DataFrame(X))
y_train, y_test = dividir_entrenamiento_prueba(pd.DataFrame(y))

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.flatten()
y_test = y_test.values.flatten()

# Calcular coeficientes del modelo
beta = calcular_coeficientes(X_train, y_train)
print(f"\nCoeficientes del modelo (β): {beta}")

# Predecir sobre el conjunto de prueba
y_pred = predecir(X_test, beta)

# Calcular R²
r2 = calcular_r2(y_test, y_pred)
print(f"Coeficiente de determinación R²: {r2:.4f}")

# === 6. PREDICCIÓN PARA UN NUEVO ESTUDIANTE ===
nuevo_estudiante = np.array([[25, 68, 58]])  # [Horas estudio, nota previa, asistencia]
prediccion = predecir(nuevo_estudiante, beta)
print(f"\nPredicción de rendimiento para el nuevo estudiante: {prediccion[0]:.2f}")
