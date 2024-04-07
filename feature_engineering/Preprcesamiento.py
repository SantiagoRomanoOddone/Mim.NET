import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
from sklearn.base import BaseEstimator, TransformerMixin


def apply_log(data, features):
    """Transformación logarítmica sobre una lista de columnas"""
    for feature in features:
        # Aplicar log(1 + x) para evitar problemas con valores cero (base e)
        data[feature] = np.log1p(data[feature])
    return data


class OutlierCorrector(BaseEstimator, TransformerMixin):
    """Objeto que tiene como finalidad aprender la media y desvio de un conjunto de columnas individualmente para luego realizar la detección y corrección de outliers en esa columna
    Si la columna no es entero o float, no es una columna aceptable y se avisa al usuario"""
    def __init__(self, desv_threshold=3):
        self.desv_threshold = desv_threshold
        self.column_stats = {}

    def params(self):
        print(f'{"":->100}')
        print(f'desv_threshold: {self.desv_threshold}')
        print(f'column_stats: {self.column_stats}')
        print(f'{"":->100}')

    def fit(self, data_slice, y=None):
        # Cada vez que se fitea, se debe limpiar las stats de las columnas
        self.column_stats = {}
        # Calcular la media y el desvío estándar para cada columna
        for col in data_slice.columns:
            if data_slice[col].dtype in ['int64', 'float64']:
                mean = data_slice[col].mean()
                std = data_slice[col].std()
                self.column_stats[col] = {'mean': mean, 'std': std}
            else:
                print(f'No se puede identificar outliers en la columna "{col}" debido a que no es una columna numerica')
        return self

    def transform(self, data_slice):
        # Verificar y corregir los outliers para cada columna
        for col in data_slice.columns:
            if col in self.column_stats:
                mean = self.column_stats[col]['mean']
                std = self.column_stats[col]['std']
                lower_bound = mean - self.desv_threshold * std
                upper_bound = mean + self.desv_threshold * std

                # Corregir los outliers
                data_slice[col] = np.where(data_slice[col] < lower_bound, mean - self.desv_threshold * std, data_slice[col])
                data_slice[col] = np.where(data_slice[col] > upper_bound, mean + self.desv_threshold * std, data_slice[col])
            else:
                print(f'La columna "{col}" no fue parte del fit de este transformer')

        return data_slice


class OneHot(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_last_column=True):
        self.column_categories = {}
        self.exclude_last_column = exclude_last_column

    def params(self):
        print(f'{"":->100}')
        print(f'exclude_last_column: {self.exclude_last_column}')
        print(f'column_categories: {self.column_categories}')
        print(f'{"":->100}')

    def fit(self, data, cols, y=None):
        # Cada vez que se fitea, se debe limpiar las categorias de las columnas
        self.column_categories = {}
        data_slice = data[cols]
        # Para cada columna se guarda los valores unicos, que luego al transformar el dataset serán las nuevas columnas
        for col in data_slice.columns:
            if data_slice[col].dtype in ['int64', 'float64']:
                print(
                    f'No se puede ejecutar One Hot Encoding sobre la columna "{col}" debido a que es una columna numerica')
            else:
                categs = list(data_slice[col].unique())
                categs.sort()
                self.column_categories[col] = categs
        return self

    def transform(self, data, cols):
        # Para cada columna, revisar las categorias y hacer OHE
        for col in data.columns:
            # Revisar si la columna existio en la fase de fit y evita crashear
            if col in self.column_categories:
                # Recorrer cada categoria dentro de la columna actual
                for cat in self.column_categories[col]:
                    # Revisar si hay que excluir la ultima categoria y si se esta recorriendo esa ultima categoria
                    if self.exclude_last_column and self.column_categories[col].index(cat) == len(
                            self.column_categories[col]) - 1:
                        pass
                    else:
                        # Operacion de agregado de nuevas columnas con nuevo nombre donde hay solamente 1s y 0s
                        data[f'{col}_{cat.replace(" ", "_")}'.lower()] = np.where(data[col] == cat, 1, 0)
                # Dropear la columna original al terminar el One Hot Encoding
                data.drop(col, inplace=True, axis=1)
            else:
                # Si se recibieron más columnas en transform que en fit, avisar y seguir. esa columna no se eliminara del slice
                print(f'La columna "{col}" no fue parte del fit de este transformer')

        return data


class ZScaler(BaseEstimator, TransformerMixin):
    '''Objeto que tiene como finalidad aprender la media y desvio de un conjunto de columnas individualmente para luego realizar la detección y corrección de outliers en esa columna
    Si la columna no es entero o float, no es una columna aceptable y se avisa al usuario'''

    def __init__(self, check_unique_qty=True):
        self.check_unique_qty = check_unique_qty
        self.column_stats = {}

    def params(self):
        print(f'{"":->100}')
        print(f'check_unique_qty: {self.check_unique_qty}')
        print(f'column_stats: {self.column_stats}')
        print(f'{"":->100}')

    def fit(self, data_slice, y=None):
        # Cada vez que se fitea, se debe limpiar las stats de las columnas
        self.column_stats = {}
        # Calcular la media y el desvío estándar para cada columna
        for col in data_slice.columns:
            if data_slice[col].dtype in ['int64', 'float64'] and data_slice[
                col].nunique() <= 2 and self.check_unique_qty:
                print(
                    f'No se escala la columna "{col}" debido a que, a pesar de ser numerica, solo tiene 2 o menos valores unicos')
            elif data_slice[col].dtype in ['int64', 'float64']:
                print(col)
                mean = data_slice[col].mean()
                std = data_slice[col].std()
                self.column_stats[col] = {'mean': mean, 'std': std}
            else:
                print(f'No se puede escalar la columna "{col}" debido a que no es una columna numerica')
        return self

    def transform(self, data_slice):
        # Verificar las columnas y escalar los valores restando la media y dividiendo por el desvio
        for col in data_slice.columns:
            if col in self.column_stats:
                mean = self.column_stats[col]['mean']
                std = self.column_stats[col]['std']

                # Escalamiento de la columna
                data_slice[col] = (data_slice[col] - mean) / std
            else:
                print(f'La columna "{col}" no fue parte del fit de este transformer')

        return data_slice


class ZeroOneScaler(BaseEstimator, TransformerMixin):
    '''Objeto que tiene como finalidad aprender la media y desvio de un conjunto de columnas individualmente para luego realizar la detección y corrección de outliers en esa columna
    Si la columna no es entero o float, no es una columna aceptable y se avisa al usuario'''

    def __init__(self, check_unique_qty=True):
        self.check_unique_qty = check_unique_qty
        self.column_stats = {}

    def params(self):
        print(f'{"":->100}')
        print(f'check_unique_qty: {self.check_unique_qty}')
        print(f'column_stats: {self.column_stats}')
        print(f'{"":->100}')

    def fit(self, data_slice, y=None):
        # Cada vez que se fitea, se debe limpiar las stats de las columnas
        self.column_stats = {}
        # Calcular la media y el desvío estándar para cada columna
        for col in data_slice.columns:
            if data_slice[col].dtype in ['int64', 'float64'] and data_slice[
                col].nunique() <= 2 and self.check_unique_qty:
                print(
                    f'No se escala la columna "{col}" debido a que, a pesar de ser numerica, solo tiene 2 o menos valores unicos')
            elif data_slice[col].dtype in ['int64', 'float64']:
                print(col)
                min = data_slice[col].min()
                max = data_slice[col].max()
                gap = max - min
                self.column_stats[col] = {'min': min, 'max': max, 'gap': gap}
            else:
                print(f'No se puede escalar la columna "{col}" debido a que no es una columna numerica')
        return self

    def transform(self, data_slice):
        # Verificar las columnas y escalar los valores restando la media y dividiendo por el desvio
        for col in data_slice.columns:
            if col in self.column_stats:
                min = self.column_stats[col]['min']
                gap = self.column_stats[col]['gap']

                # Escalamiento de la columna
                data_slice[col] = (data_slice[col] - min) / gap
            else:
                print(f'La columna "{col}" no fue parte del fit de este transformer')

        return data_slice


def day_counter(df, bool_col, date_col):
    '''
    El propósito de esta función es contar el número de días que pasaron desde que se activó una columna booleana.

    Parámetros:
    df: DataFrame
    bool_col: str, nombre de la columna booleana en el DataFrame.
    date_col: str, nombre de la columna de fecha en el DataFrame.

    Retorna:
    df: DataFrame con una nueva columna que cuenta el número de días que pasaron desde que se activó la columna booleana.
    '''

    df[date_col] = pd.to_datetime(df[date_col])
    df[bool_col] = df[bool_col].astype(int)
    df.sort_values(by=[date_col], inplace=True)

    # Obtener el primer dia con un 1
    last_date = df.loc[df[bool_col] == 1, date_col].iloc[0]

    # Obtener la primer posicion con un 1
    first_pos = df[bool_col].idxmax()

    for i in range(len(df)):
        if i < first_pos:
            df.loc[i, f'days_since_{bool_col}'] = 0
        else:
            if df.loc[i, bool_col] == 1:
                df.loc[i, f'days_since_{bool_col}'] = 0
                last_date = df.loc[i, date_col]
            else:
                df.loc[i, f'days_since_{bool_col}'] = (df.loc[i, date_col] - last_date).days

    return df

def days_with_1_streak(df, bool_col, date_col):
    '''
    El propósito de esta función es contar el número de días consecutivos con una columna booleana activa.

    Parámetros:
    df: DataFrame
    bool_col: str, nombre de la columna booleana en el DataFrame.
    date_col: str, nombre de la columna de fecha en el DataFrame.

    Retorna:
    df: DataFrame con una nueva columna que expresa los dias con la columna booleana activa.
    '''

    df[date_col] = pd.to_datetime(df[date_col])
    df[bool_col] = df[bool_col].astype(int)
    df.sort_values(by=[date_col], inplace=True)

    i_streak = 0
    for i in range(len(df)):
        if df.loc[i, bool_col] == 1:
            i_streak += 1
            df.loc[i, f'{bool_col}_streak'] = i_streak
        else:
            df.loc[i, f'{bool_col}_streak'] = 0
            i_streak = 0

    return df

def days_with_0_streak(df, bool_col, date_col):
    '''
    El propósito de esta función es contar el número de días consecutivos con una columna booleana inactiva.

    Parámetros:
    df: DataFrame
    bool_col: str, nombre de la columna booleana en el DataFrame.
    date_col: str, nombre de la columna de fecha en el DataFrame.

    Retorna:
    df: DataFrame con una nueva columna que expresa los dias con la columna booleana inactiva.
    '''

    df[date_col] = pd.to_datetime(df[date_col])
    df[bool_col] = df[bool_col].astype(int)
    df.sort_values(by=[date_col], inplace=True)
    
    i_streak = 0
    for i in range(len(df)):
        if df.loc[i, bool_col] == 0:
            i_streak += 1
            df.loc[i, f'{bool_col}_streak'] = i_streak
        else:
            df.loc[i, f'{bool_col}_streak'] = 0
            i_streak = 0

    return df

def fit_linear_regression(df, X, y, group = False, groupping_variable='', intercept=True, plot=False, pred= False, X_test= None):
    '''
    El propósito de esta función es ajustar un modelo de regresión lineal a un conjunto de datos.
    Puede ser útil si buscamos encontrar una relación lineal entre dos o más variables.
    Por ejemplo, encontrar la tendencia de los datos en el tiempo para diferentes subconjuntos de datos.
    En caso de no agrupar, se ajustará un modelo de regresión lineal a todos los datos.

    Parámetros:
    df: DataFrame
    X: str, nombre de la columna que se utilizará como variable independiente. (Puede ser una lista de columnas)
    y: str, nombre de la columna que se utilizará como variable dependiente.
    group: bool, si se quiere ajustar un modelo de regresión lineal para diferentes subconjuntos de datos.
    groupping_variable: str, nombre de la columna que se utilizará para agrupar los datos. 
        Por ejemplo, si elegimos por año, se ajustará un modelo de regresión lineal para cada año.
    intercept: bool, si se quiere incluir el intercepto en el modelo.
    plot: bool, si se quiere graficar la relación entre las variables.
    pred: bool, si se quiere predecir los valores de y para un conjunto de datos de test.
    X_test: DataFrame, conjunto de datos de prueba para evaluar el modelo. 
        (El modelo va a predecir los valores para X_test usando los coeficientes ajustados en el conjunto de entrenamiento)


    Retorna:
    df: DataFrame con una nueva columna que contiene las predicciones del modelo de regresión lineal.
    X_test: DataFrame con una nueva columna que contiene las predicciones del modelo de regresión lineal.
    '''

    if group == False:
        print('Fitteando una regresión lineal en todo el conjunto de datos (sin agrupaciones)')
        X_cols = df[X]
        y_cols = df[y]

        if intercept:
            model = LinearRegression().fit(X_cols, y_cols)
        else:
            model = LinearRegression(fit_intercept=False).fit(X_cols, y_cols)

        df['linear_regression_prediction'] = model.predict(X_cols)
        

        if plot == True:
            if len(X_cols.columns) == 1:
                sns.scatterplot(x=X_cols.iloc[:, 0], y=y_cols, data=df)
                sns.lineplot(x=X_cols.iloc[:, 0], y=df['linear_regression_prediction'], data=df)
                plt.title('Regresión lineal')
                plt.show()
            else:
                print('No se puede graficar la relación entre más de dos variables')
        print(f'R2 score: {r2_score(y_cols, df["linear_regression_prediction"])}')

        if pred == True:
            print('Prediciendo valores para X_test')
            X_test = X_test[X]
            X_test['linear_regression_prediction'] = model.predict(X_test)
            
            return df, X_test
        else:

            return df
        
    else:
        temp_df = df.copy()
        print('Fitteando una regresión lineal agrupada por: ', groupping_variable)

        for group in temp_df[groupping_variable].unique():
            print(f'Regresión del grupo {group}')
            df_groupped = temp_df[temp_df[groupping_variable] == group]
            X_cols = df_groupped[X]
            y_cols = df_groupped[y]

            if intercept:
                model = LinearRegression().fit(X_cols, y_cols)
            else:
                model = LinearRegression(fit_intercept=False).fit(X_cols, y_cols)
                
            df.loc[df[groupping_variable] == group, f'linear_regression_prediction_{groupping_variable}'] = model.predict(X_cols)
            df_groupped['linear_regression_prediction'] = model.predict(X_cols)

            if plot == True:
                if len(X_cols.columns) == 1:
                    sns.scatterplot(x=X_cols.iloc[:, 0], y=y_cols, data=df_groupped)
                    sns.lineplot(x=X_cols.iloc[:, 0], y='linear_regression_prediction', data=df_groupped)
                    plt.title(f'Regresión lineal para el grupo {group}')
                    plt.show()
                else:
                    print('No se puede graficar la relación entre más de dos variables')
            print(f'R2 score: {r2_score(y_cols, df_groupped["linear_regression_prediction"])}')

            print('---------------------------------------------')

            if pred:
                print('Prediciendo valores para X_test')
                X_test_groupped = X_test.copy()
                X_test_groupped = X_test_groupped.loc[X_test_groupped[groupping_variable] == group, X]
                X_test.loc[X_test[groupping_variable] == group, 'linear_regression_prediction'] = model.predict(X_test_groupped)

        if pred:
            return df, X_test
        else:
            return df

