import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
from datetime import datetime, timedelta
import time
from joblib import dump

import xgboost as xgb


def rmspe(y_true, y_pred):
    """Calcula el RMSPE entre las etiquetas reales y las predichas.

    RMSPE = Root of the Mean Square Percentage Error

    Se filtran los casos de etiquetas reales iguales a cero ya que no se puede
    calcular el error porcentual (division por 0). Se puede utilizar como
    funcion para optimizar un modelo de Regresión.

    Note:
        RMSPE = RAIZ( PROMEDIO( ( (y_real - y_pred) / y_real)^2 ) )

    Args:
        y_true (Series): Serie de Pandas con etiquetas reales.
        y_pred (Series): Serie de Pandas con etiquetas predichas.

    Returns:
        rmspe (float): Retorna el resultado del cálculo del rmspe
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    # Filtrado de los casos donde y_real es 0
    relevantes = np.where(y_true == 0, False, True)
    rmspe = (np.sqrt(np.mean(np.square((y_true[relevantes] - y_pred[relevantes]) / y_true[relevantes]))))
    return rmspe


def n_samples(n, boundaries):
    """Samplea N sets de hiperparámetros con las reglas recibidas en el diccionario boundaries

    Se encarga de crear la random search de hiperparámetros con la cual se va a
    entrenar un modelo para obtener los hiperparámetros óptimos.

    Note:
        El diccionario boundaries debe tener en las keys los nombres de los
        hiperparámetros a samplear y sus values deben ser listas de dos valores:
        la funcion de sampleo a utilizar y los parámetros que debe recibir dicha
        función para hacerlo.

    Args:
        n (int): Cantidad de sampleos, tamaño del random search.
        boundaries (dict): diccionario con los hiperparámetros como claves y la
            funcion de sampleo y los argumentos que debe recibir en los valores.

    Returns:
        samp (dicc): Retorna un diccionario que tiene como claves los mismos
        hiperparámetros recibidos en boundaries, pero los valores son listas de
        n hiperparámetros para entrenar al modelo en cuestion. Es decir que el
        diccionario retornado es una forma de representar el random search a entrenar.

    Example:
        >>> boundaries_xg = dict()
        >>> boundaries_xg['n_estimators'] = [random.randint, (600, 1600)]
        >>> boundaries_xg['learning_rate'] = [random.uniform, (0.01, 0.3)]
        >>> boundaries_xg['max_depth'] = [random.randint, (6, 12)]
        >>> boundaries_xg['colsample_bytree'] = [random.uniform, (0.8, 0.9)]
        >>> boundaries_xg['subsample'] = [random.uniform, (0.7, 0.9)]
        >>> boundaries_xg['gamma'] = [random.uniform, (0.1, 0.4)]
        >>> boundaries_xg['n_jobs'] = [random.choice, ([-1],)]
        >>> boundaries_xg['random_state'] = [random.choice, ([42],)]

        >>> n_samples(2, boundaries_xg)
                {'n_estimators': [980, 1461],
                'learning_rate': [0.22913148222774557, 0.0574317900361972],
                'max_depth': [6, 10],
                'colsample_bytree': [0.8710652035201274, 0.8375843798791736],
                'subsample': [0.7797753379676025, 0.7906648475211833],
                'gamma': [0.10862156063014074, 0.28454450904898587],
                'n_jobs': [-1, -1],
                'random_state': [42, 42]}
    """

    # Se genera una copia del diccionario boundaries ya que es mutable. Se hace solo para que el sampleo tenga las mismas keys que boundaries
    samp = copy.deepcopy(boundaries)
    # Para cada key, se borran los values para dejar listas vacias en las que cargar los valores sampleados
    for key in samp:
        samp[key].clear()
    # Utilizando la funcion y largumentos de boundaries, se producen n sampleos y se guardan en el nuevo diccionario
    for i in range(n):
        for key in samp:
            func = boundaries[key][0]
            args = boundaries[key][1]
            samp[key].append(func(*args))
    return samp


def entrena_evalua(algoritmo, x_train, y_train, x_valid, y_valid, hparam_sample):
    """Entrena y evalua en train y valid un algoritmo con los hiperparámetros recibidos

    Toma un algoritmo de ML con un conjunto de hiperparámetros dados y se encarga
    de entrenar el modelo con el conjunto de entrenamiento. Una vez entrenado, lo
    evalúa en el conjunto de validación. La funcion está preparada para contar el
    tiempo que se tarda en entrenar.

    Devuelve un registro con los hiperparámetros utilizados y el resultado del
    entrenamiento en términos del tiempo que ha demorado y la performance del
    modelo entrenado.

    Args:
        algoritmo (object): Objeto que representa un algoritmo de Machine Learning.
            Por ejemplo, XGBRegressor, RandomForestRegressor u otro algoritmo de
            cualquier otra librería de Machine Learning. El objeto debe tener como
            métodos fit y predict.
        x_train (DataFrame): DataFrame de Pandas con las variables independientes
            del conjunto de entrenamiento.
        y_train (array): array con las etiquietas reales del conjunto de
            entrenamiento.
        x_valid (DataFrame): DataFrame de Pandas con las variables independientes
            del conjunto de validación.
        y_valid (array): array con las etiquietas reales del conjunto de
            validación.
        hparam_sample (dict): diccionario con los hiperparámetros con los que se
            entrenará el algoritmo recibido. Las claves son los nombres del
            hiperparámetro y los valores son el valor del hiperparámetro a usar al
            instanciar el modelo a entrenar.

    Returns:
        Retorna una tupla de dos valores. Primero, el DataFrame de registro del
        modelo entrenado y segundo un objeto que contiene el algoritmo instanciado
        y entrenado.
    """

    # Registro del momento de comienzo
    start_time = time.time()

    # Creacion y entrenamiento del algoritmo
    model = algoritmo(**hparam_sample)
    # Este paso en el caso de XGBoost agrega un parámetro en el entrenamiento que permite hacer early stopping y no overfittear
    if algoritmo == xgb.XGBRegressor:
        model.fit(x_train, y_train, eval_set=[(x_valid, np.array(y_valid))],
                  verbose=False)  # , eval_metric=mean_squared_error)
    else:
        model.fit(x_train, y_train)

    # Prediccion de etiquetas de entrenamiento y calculo de errores
    y_pred_train = pd.Series(model.predict(x_train))
    rmspe_train = rmspe(y_train, y_pred_train)

    # Prediccion de etiquetas de validacion y calculo de errores
    y_pred = pd.Series(model.predict(x_valid))
    rmspe_val = rmspe(y_valid, y_pred)

    # Registro del momento de finalizacion
    stop_time = time.time()

    # Se crea la lista con las columnas que tendra el dataframe que retornara la funcion
    columns = list(hparam_sample.keys())
    columns.extend(['rmspe_train', 'rmspe_val', 'algoritmo', 'tiempo_min'])

    # Si el algoritmo es XGBoost, se agrega el best_ntree_limit que registra la cantidad de arboles en que corto la ejecucion para evitar overfitting
    if algoritmo == xgb.XGBRegressor:
        columns.extend(['best_ntree_limit'])

    # Se registran la info que tendra el dataframe que retornara la funcion. Primero, los hiperparámetros.
    data = []
    for key in hparam_sample:
        data.append(hparam_sample[key])

    # Se agrega a la lista de resultados los errores, el algoritmo y el tiempo
    data.extend([rmspe_train, rmspe_val, algoritmo.__name__, round((stop_time - start_time) / 60, 1)])

    # Si el algoritmo es XGBoost, se agrega la info del best_ntree_limit
    if algoritmo == xgb.XGBRegressor:
        data.extend([model.best_ntree_limit])

    return pd.DataFrame(data=[data], columns=columns), model


def hyper_search(algoritmo, x_train, y_train, x_valid, y_valid, minutos=1, hparam_n_samples=dict(),
                 file_path='archivo.csv'):
    """Búsqueda de Hiperparámetros Random Search por iteraciones y/o tiempo

    Realiza la búsqueda de hiperparámetros por random search durante un lapso de
    tiempo límite o una cantidad de iteraciones. El usuario pasa un algoritmo
    de ML supervisado, un conjunto de N hiperparámetros (que puede obtenerse
    usando la funcion n_samples) y una cantidad de tiempo en minutos máximo que
    tiene disponibles para realizar la búsqueda aleatoria de hiperparámetros.

    Note:
        La funcion irá guardando los resultados intermedios de cada modelo que
        entrene y evalúe con la informacion de los hiperparámetros que llevaron
        a la obtencion del modelo. Se guardará la informacion sobre todos los
        modelos hechos, pero solo se guardará el modelo que mejor performe en
        el set de validación.

    Args:
        algoritmo (object): Objeto que representa un algoritmo de Machine Learning.
            Por ejemplo, XGBRegressor, RandomForestRegressor u otro algoritmo de
            cualquier otra librería de Machine Learning. El objeto debe tener como
            métodos fit y predict.
        x_train (DataFrame): DataFrame de Pandas con las variables independientes
            del conjunto de entrenamiento.
        y_train (array): array con las etiquietas reales del conjunto de
            entrenamiento.
        x_valid (DataFrame): DataFrame de Pandas con las variables independientes
            del conjunto de validación.
        y_valid (array): array con las etiquietas reales del conjunto de
            validación.
        minutos (int): cantidad de tiempo disponible en minutos.
        hparam_n_samples (dict): diccionario con los N conjuntos de hiperparámetros
            con los que se entrenará el algoritmo recibido. Cada hiperparámetro
            tendrá una lista de N valores que la función irá consumiendo en cada
            instancia de entrenamiento. Puede ser resultado de la funcion n_samples.
        file_path (string): nombre del archivo .csv que se va a guardar con los
            resultados de la búsqueda aleatoria de hiperparámetros.

    Returns:
        Retorna una tupla de dos valores. Primero, el DataFrame de registro de todos
        los modelos entrenados en el ciclo completo. Segundo, una lista con los 5
        mejores modelos indicando los indices para buscar los hiperparametros de esos
        modelos en el dataframe de resultaros, los 5 modelos entrenados y el error
        en validacion de cada no de estos modelos
    """

    # with open(f'hparam_samples_{algoritmo.__name__}.txt', 'w') as archivo:
    #         for key in hparam_n_samples:
    #             archivo.write(f'{key}: {hparam_n_samples[key]}\n\n')

    # Inicializamos variables relevantes para el while: cantidad de iteraciones y tiempo
    n = len(hparam_n_samples[list(hparam_n_samples.keys())[0]])
    i = 0
    hasta = datetime.today() + timedelta(minutes=minutos)
    print(f"Comienza el proceso de random search hasta llegar a {n} modelos o {minutos} minutos disponibles")
    print(f"Ahora: {datetime.today()}\nFinalización aproximada: {hasta}")

    # Se inicializa la lista de modelos con el indice que tendrá en el dataframe, el modelo en si y el rmspe_valid
    indices = []
    modelos = []
    errores = []

    # El loop se ejecuta hasta que se cumpla el tiempo o que se recorra el sampleo con exito, lo que suceda antes
    while datetime.today() < hasta and i < n:
        # Populamos el diccionario hparams con los hiperparámetros del loop actual
        hparams = dict()
        for key in hparam_n_samples:
            # Para cada key, buscamos en la lista de hiperparams sampleados el que corresponde para este loop
            hparams[key] = hparam_n_samples[key][i]

        # Se encierra el entrenamiento y evaluacion del modelo en un try para gestionar el error en caso que falle en algun momento
        try:
            # Entrenamiento y evaluacion del modelo
            resultado, modelo = entrena_evalua(algoritmo, x_train, y_train, x_valid, y_valid, hparam_sample=hparams)

            # Se suman los modelos y su info a las listas de los modelos que quedan como los 5 mejores
            modelos.append(modelo)
            errores.append(resultado.loc[0, 'rmspe_val'])
            indices.append(i)

            # Se guarda el resultado del ultimo modelo en el dataframe que va acumulando la info de los modelos.
            if i == 0:
                dataframe = resultado.copy()
            else:
                dataframe = pd.concat([dataframe, resultado])

            # Guardamos el dataset con todos los modelos corridos hasta el momento para que si crashea tenga registro de lo hecho
            dataframe.to_csv(file_path)
            i += 1

        except xgb.core.XGBoostError as err:
            # Esta parte del código se ejecutará si se lanza la excepción
            print(f"Ocurrió un error con XGBoost: {err}")
            print(f"Los hiperparámetros con los que falló el código son:\n{hparams}")
            for key in hparam_n_samples:
                hparam_n_samples[key].pop(i)

        except MemoryError as err:
            print(f"Se produjo un MemoryError: {err}")
            print(f"Los hiperparámetros con los que falló el código son:\n{hparams}")

        # Nos quedamos siempre con los mejores 5 modelos. Si hay mas de 5, se remueve el peor.
        if len(modelos) > 5:
            # Posicion en las listas del peor modelo segun rmspe_valid
            indice_remove = errores.index(max(errores))
            # Se elimina el peor modelo
            modelos.pop(indice_remove)
            errores.pop(indice_remove)
            indices.pop(indice_remove)

        # Guardamos el mejor modelo entrenado en memoria para tenerlo en caso de que crashee
        dump(modelos[errores.index(min(errores))], f'{file_path.split(".")[0]}_top_model')

        # La función va hablando cada 5 modelos entrenados
        if (i) % 5 == 0:
            print(
                f"Modelo {i} terminado. Quedan {n - i} modelos por correr o {int((hasta - datetime.today()).total_seconds() / 60)} minutos. ({datetime.today()})")

    # La funcion informa que ha terminado de entrenar y evaluar y explica si se quedó sin tiempo o sin hiperparámetros para probar
    else:
        # Chequeamos por qué se salió del while y lo informamos al usuario
        if datetime.today() >= hasta:
            print(f'Se acabo el tiempo el tiempo, se procesaron {i} modelos')
        elif i >= n:
            print(f'Termino el proceso por haber recorrido los {n} modelos')

        mejores_modelos = [indices, modelos, errores]
        # En caso de que no crashee y se llegue hasta el final de la ejecucion, se tienen los mejores 5 modelos entrenados y todos los resultados probados
    return dataframe, mejores_modelos


def graficar_rand_search(random_search):
    """Grafica el DataFrame que resulta del Random Search

    Hace tantos gráficos como sean necesarios para mostrar como varia el error
    en validacion respecto al variar cada uno de los hiperparámetros que se
    probaron.

    Note:
        La función ignorará algunos nombres de columna en particular como
        random_state ya que no tiene valor graficar esos hiperparámetros.

    Args:
        random_search (DataFrame): DataFrame de Pandas con el resultado del
            random search, lo que se obtiene de la funcion hyper_search.
    """
    columnas = list(random_search.columns)

    # Se remueve la info que no tiene valor graficar
    remover = ['random_state', 'n_jobs', 'bootstrap', 'algoritmo', 'rmspe_val']
    for item in remover:
        if item in columnas:
            columnas.remove(item)

    # Calculo del mosaico para graficar
    graf_cell = int(len(columnas) ** 0.5 if len(columnas) ** 0.5 % 1 == 0 else len(columnas) ** 0.5 // 1 + 1)

    # Se crea un conjunto de graficos segun la cantidad necesaria de columnas a graficar
    fig, ax = plt.subplots(graf_cell, graf_cell, layout='tight', figsize=(10, 10))

    # Slice del dataframe a graficar
    graficar = random_search[columnas + ['rmspe_val']]

    for col in columnas:
        pos = (columnas.index(col) // graf_cell, columnas.index(col) % graf_cell)

        if graficar.dtypes[col] in ['int64', 'float64']:
            ax[pos].scatter(graficar[col], graficar['rmspe_val'], s=10, alpha=0.7)

            # Configurar los ejes
            ax[pos].set_xlabel(col)
            ax[pos].set_ylabel('VALID RMSPE')

        elif graficar.dtypes[col] in ['object']:
            # Agrupar los valores de "rmspe_val" por categoría las columnas de texto
            grouped_data = graficar.groupby(col)['rmspe_val'].apply(list)

            # Crear una lista de etiquetas y una lista de valores para el boxplot
            labels = grouped_data.index.tolist()
            values = grouped_data.tolist()
            ax[pos].boxplot(values, labels=labels)

            # Configurar los ejes
            ax[pos].set_xlabel(col)
            ax[pos].set_ylabel('VALID RMSPE')
            ax[pos].set_xticklabels(ax[pos].get_xticklabels(), rotation=90)

        ax[pos].set_title(col)
    plt.show()