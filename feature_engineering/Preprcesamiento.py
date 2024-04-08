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

def add_fourier_terms(df, date_col, n_terms, pred = False ,pred_df=None):
    '''
    El propósito de esta función es agregar términos de Fourier a un DataFrame.
    Son funciones de seno y coseno que se utilizan para modelar la estacionalidad en series de tiempo.
    Importante notar que esta función solo agrega terminos de Fourier para la estacionalidad anual. 
    En caso de querer agregar estacionalidad de otro periodo, se podría ajustar la funcion.
        Por ejemplo: para estacionalidad mensual, la formula sería sin/cos(2 * np.pi * i * df[date_col].dt.month / 12)

    Los terminos de Fourier nos podrían ser útiles para un xgboost como variables independientes, 
    aunque las variables de fecha ya podrían ser suficientes para capturar la estacionalidad.
    
    Parámetros:
    df: DataFrame
    date_col: str, nombre de la columna de fecha en el DataFrame.
    n_terms: int, número de términos de Fourier que se agregarán al DataFrame.

    Retorna:
    df: DataFrame con los términos de Fourier agregados.
    '''

    df[date_col] = pd.to_datetime(df[date_col])
    for i in range(1, n_terms+1):
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * df[date_col].dt.dayofyear / 365)
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * df[date_col].dt.dayofyear / 365)

    if pred:
        pred_df[date_col] = pd.to_datetime(pred_df[date_col])
        for i in range(1, n_terms+1):
            pred_df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * pred_df[date_col].dt.dayofyear / 365)
            pred_df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * pred_df[date_col].dt.dayofyear / 365)
        return df, pred_df
    else:
        return df
    
def aggregation_by_col(df, group_col, agg_col, agg_type='mean'):
    '''
    El propósito de la función es agregar datos por una columna especifica de un Dataframe (por agrupaciones) y agregar la columna con la estadística agregada.

    Parámetros:
    df: DataFrame
    group_col: str, nombre de la columna por la cual se quiere agrupar. Puede ser una lista de columnas.
    agg_col: str, nombre de la columna que se quiere agregar. Puede ser una lista de columnas.
    agg_type: str, tipo de agregación que se quiere realizar. Puede ser 'mean', 'sum', 'median', 'max', 'min', 'std', 'var', 'count'.

    Retorna:
    df: DataFrame con una nueva columna que contiene la estadística agregada.
    '''

    if type(agg_col) == str:
        agg_col = [agg_col]
    
    if len(agg_col) > 1:
        if len(agg_type) > 1:
            for i in agg_type:
                for j in agg_col:
                    df[f'{j}_{i}'] = df.groupby(group_col)[j].transform(i)
        else:
            for j in agg_col:
                df[f'{j}_{agg_type}'] = df.groupby(group_col)[j].transform(i)
    else:
        if len(agg_type) > 1:
            for i in agg_type:
                df[f'{agg_col[0]}_{i}'] = df.groupby(group_col)[agg_col[0]].transform(i)
        else:
            df[f'{agg_col[0]}_{agg_type}'] = df.groupby(group_col)[agg_col[0]].transform(agg_type)
    
    return df

def bining(df, cols, mode='quantile', q=None, limits=None):
    '''
    El propósito de esta función es discretizar variables continuas en intervalos.
    Puede ser útil para convertir variables continuas en categóricas, y utilizar la variable categorica en un modelo de machine learning.

    Parámetros:
    df: DataFrame
    cols: Lista de str, nombre de la columna que se quiere discretizar.
    mode: str, modo de discretización. Puede ser 'quantile' o 'limits'.
    q: int, número de cuantiles en los que se quiere dividir la variable.
    limits: Lista de int, límites de los intervalos.

    Retorna:
    df: DataFrame con las columnas discretizadas.
    '''
    if mode == 'quantile':
        for col in cols:
            df[f'{col}_quantile'] = pd.qcut(df[col], q=q, labels=False)
    elif mode == 'limits':
        for col in cols:
            df[f'{col}_bin'] = pd.cut(df[col], bins=limits, labels=False)
    else:
        print('Mode {} not supported, try "limits" or "quantile"'.format(mode))

    return df

def create_time_feat(df):
    # Copy of df
    df = df.copy()
    # df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    # Creating time feats
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    df['quarter'] = df['Date'].dt.quarter
    df['is_month_start'] = df['Date'].dt.is_month_start
    df['is_month_end'] = df['Date'].dt.is_month_end
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end
    df['is_year_start'] = df['Date'].dt.is_year_start
    df['is_year_end'] = df['Date'].dt.is_year_end
    df['dayofyear'] = df['Date'].dt.dayofyear
    
    # Season of year
    df['Winter'] = df['month'].isin([12, 1, 2])
    df['Spring'] = df['month'].isin([3, 4, 5])
    df['Summer'] = df['month'].isin([6, 7, 8])
    df['Fall'] = df['month'].isin([9, 10, 11])
    
    # Trend
    # df['Trend'] = range(1, len(df) + 1)

    # Some special dates
    # df['Black_Friday'] = (df['Date'] == '2023-11-24') | (df['Date'] == '2024-11-29')
    # df['Christmas'] = (df['Date'] == '2023-12-25') | (df['Date'] == '2024-12-25')

    # Turning variables into dummies
    df = pd.get_dummies(df, columns=['year', 'month', 'day', 'weekofyear', 'quarter', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end'],drop_first=False)

    # Obtener las columnas que tienen valores False
    columns_to_drop = [col for col in df.columns if '_False' in col]

    # Eliminar las columnas que tienen valores False
    df.drop(columns=columns_to_drop, inplace=True)
    
    # After turning variables into dummies, some of them should be kept as numerical as well
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    df['quarter'] = df['Date'].dt.quarter
    
    # 'dayofyear',
    return df

def calculate_mape(y_pred,y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def add_lags(df, feat, lags):
    # Copy df
    df = df.copy()

    # Sorting dates
    df.sort_values('Date', inplace=True)
    
    # Adding lags
    for lag in lags:
        df[feat + f'lag{lag}'] = df[feat].shift(lag)

    return df