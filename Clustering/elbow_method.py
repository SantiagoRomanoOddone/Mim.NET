from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def elbow_method(data, columns, rango=(2, 30)):
    # Lista para almacenar las inercias
    inertia_values = []
    iteration_values = []

    # Se itera por todos los k en los
    for n_clusters in range(*rango):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # Se fitea el modelo
        kmeans.fit(data[columns])
        iteration_values.append(kmeans.n_iter_)
        inertia_values.append(kmeans.inertia_)
        print(f'{n_clusters} clusters: inertia value of {kmeans.inertia_} and {kmeans.n_iter_} iterations')

    # Graficar la curva de la suma de las inercias en función del número de clusters
    fig, ax = plt.subplots(1, 2, layout='tight', figsize=(15, 7))

    ax[0].plot(range(*rango), inertia_values, marker='o', linestyle='--')
    ax[0].set_xlabel('Número de clusters')
    ax[0].set_ylabel('Inercia')
    ax[0].set_title('Método del codo')
    ax[0].set_xticks(range(*rango))

    ax[1].plot(range(*rango), iteration_values, marker='o', linestyle='--')
    ax[1].set_xlabel('Número de clusters')
    ax[1].set_ylabel('Iteraciones')
    ax[1].set_title('Iteraciones hasta cortar')
    ax[1].set_xticks(range(*rango))

    return [inertia_values, iteration_values], fig