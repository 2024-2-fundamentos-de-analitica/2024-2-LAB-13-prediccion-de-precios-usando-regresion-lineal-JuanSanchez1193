#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


import pandas as pd
import gzip
import json
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,median_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

ruta_test = "./files/input/test_data.csv.zip"
ruta_train = "./files/input/train_data.csv.zip"

df_test = pd.read_csv(
    ruta_test,
    index_col=False,
    compression='zip'
)

df_train = pd.read_csv(
    ruta_train,
    index_col=False,
    compression='zip'
)

anio_actual = 2021

df_train['Age'] = anio_actual - df_train['Year']
df_test['Age'] = anio_actual - df_test['Year']

columnas_eliminar = ['Year', 'Car_Name']
df_train = df_train.drop(columns=columnas_eliminar)
df_test = df_test.drop(columns=columnas_eliminar)

entradas_train = df_train.drop(columns="Present_Price")
salidas_train = df_train["Present_Price"]

entradas_test = df_test.drop(columns="Present_Price")
salidas_test = df_test["Present_Price"]

caracteristicas_categoricas = ['Fuel_Type', 'Selling_type', 'Transmission']
caracteristicas_numericas = list(set(entradas_train.columns) - set(caracteristicas_categoricas))

procesador = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), caracteristicas_numericas),
        ("cat", OneHotEncoder(), caracteristicas_categoricas)
    ],
    remainder="passthrough"
)

k_mejor = SelectKBest(f_regression, k='all')

modelo = LinearRegression()

tuberia = Pipeline(
    steps=[
        ("preprocessor", procesador),
        ("k_best", k_mejor),
        ("model", modelo)
    ]
)

rejilla_parametros = {
    "k_best__k": list(range(1, 16)),
    "model__fit_intercept": [True, False]
}

busqueda_grid = GridSearchCV(
    tuberia,
    param_grid=rejilla_parametros,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
    verbose=1
)

busqueda_grid.fit(entradas_train, salidas_train)

with gzip.open("./files/models/model.pkl.gz", 'wb') as f:
    pickle.dump(busqueda_grid, f)
    
resultados_metricas = {}

pred_train = busqueda_grid.predict(entradas_train)
pred_test = busqueda_grid.predict(entradas_test)

resultados_metricas['train'] = {
    'type': 'metrics',
    'dataset': 'train',
    'r2': r2_score(salidas_train, pred_train),
    'mse': mean_squared_error(salidas_train, pred_train),
    'mad': median_absolute_error(salidas_train, pred_train),
}

resultados_metricas['test'] = {
    'type': 'metrics',
    'dataset': 'test',
    'r2': r2_score(salidas_test, pred_test),
    'mse': mean_squared_error(salidas_test, pred_test),
    'mad': median_absolute_error(salidas_test, pred_test),
}

with open("./files/output/metrics.json", 'w') as f:
    f.write(json.dumps(resultados_metricas['train'])+'\n')
    f.write(json.dumps(resultados_metricas['test'])+'\n')