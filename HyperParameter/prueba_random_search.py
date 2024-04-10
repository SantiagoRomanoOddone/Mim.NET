import time_sensitive_random_search as rs
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

# Cargar el conjunto de datos de muestra (en este caso, Iris)
iris = load_iris(as_frame=True)

# Acceder a los datos y las etiquetas
X = iris.data  # Datos
y = iris.target  # Etiquetas

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos las boundaries para Random Forest
boundaries_rf = dict()
boundaries_rf['n_estimators'] = [random.randint, (50, 80)]
boundaries_rf['criterion'] = [random.choice, (['squared_error', 'friedman_mse', 'poisson'], )] # Otras opciones ya descartadas: friedman_mse y poisson
boundaries_rf['max_depth'] = [random.randint, (5, 15)]
boundaries_rf['min_samples_split'] = [random.randint, (60, 95)]
boundaries_rf['max_features'] = [random.uniform, (0.2, 0.8)]
boundaries_rf['bootstrap'] = [random.choice, ([True], )]
boundaries_rf['max_samples'] = [random.uniform, (0.2, 0.8)]
boundaries_rf['random_state'] = [random.choice, ([42], )]
boundaries_rf['n_jobs'] = [random.choice, ([-1], )]

# Con las boundaries, hacemos sampleo de un set de 1000 hiperparametros
hparams = rs.n_samples(500, boundaries_rf)

# Hacemos random search para el algorimo elegido, con los set de X e y, durante 30 minutos o las 500 iteraciones
# El resultado se guarda en un archivo ejemplo.csv
resultados, top_models = rs.hyper_search(RandomForestRegressor
                                         , X_train, y_train, X_test, y_test
                                         , minutos=30
                                         , hparam_n_samples=hparams
                                         , file_path='ejemplo.csv'
                                         )

# Se grafica el random search para ayudar a interpretarlo y seguir iterando mejoras
rs.graficar_rand_search(resultados)
