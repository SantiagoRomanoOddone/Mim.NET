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


