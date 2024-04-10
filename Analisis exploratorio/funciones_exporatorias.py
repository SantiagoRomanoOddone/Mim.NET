import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_distributions(data):
    """Funcion que grafica histogramas y gráficos de barras para un dataframe que se desea explorar\n
    input: dataframe de pandas\n\n
    consideraciones: no pasarle a la funcion ninguna variable que sea un objeto o categorica de demasiados valores únicos"""
    cols = list(data.columns)

    graf_cell = int(len(cols)**0.5//1 + 1)

    fig, ax = plt.subplots(graf_cell, graf_cell, layout='tight', figsize=(10,10))

    for col in cols:
        pos = (cols.index(col) // graf_cell, cols.index(col) % graf_cell)
        if data[col].dtype == 'object':
            graf = (data
                    .groupby(col)
                    .agg({col: 'count'})
                    .rename(columns={col: 'cant'})
                    .sort_values('cant', ascending=False)
                    )
            ax[pos].bar(graf.index, graf['cant'])
            ax[pos].set_xticklabels(ax[pos].get_xticklabels(), rotation=90)
        elif data[col].dtype in ['int64', 'float64']:
            ax[pos].hist(data[col], bins=min(50, round(len(data) ** 0.5),0), edgecolor='white', linewidth=0.3)
        else:
            print(f'{col} no se puede graficar')
        ax[pos].set_title(col)

    return fig

def plot_missing_values(df, cols):
    '''
    Plottea los valores NaNs de un DataFrame.

    Parámetros:
    df: DataFrame
    cols: Lista de str, nombre de la columna que se quiere graficar.
    '''

    df.isna().sum()[cols].plot(kind='bar', color='black')
    plt.title('Missing values')
    plt.ylabel('Number of missing values')
    plt.show()

