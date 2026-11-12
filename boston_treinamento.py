import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from pickle import dump

dados = pd.read_csv('boston.csv', sep=';')

colunas_categoricas = ['CHAS', 'RAD']
colunas_numericas = [c for c in dados.columns if c not in colunas_categoricas]

dados_numericos = dados[colunas_numericas]
dados_categoricos = dados[colunas_categoricas].astype(str)

normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)
dump(modelo_normalizador, open('normalizador_boston.model', 'wb'))

dados_numericos_normalizados = pd.DataFrame(
    modelo_normalizador.transform(dados_numericos),
    columns=colunas_numericas
)

dados_categoricos_normalizados = pd.get_dummies(dados_categoricos, prefix_sep='_', dtype=int)
colunas_dummies = list(dados_categoricos_normalizados.columns)

dados_normalizados = dados_numericos_normalizados.join(dados_categoricos_normalizados)
colunas_modelo = list(dados_normalizados.columns)

dump({
    'colunas_numericas': colunas_numericas,
    'colunas_categoricas': colunas_categoricas,
    'colunas_dummies': colunas_dummies,
    'colunas_modelo': colunas_modelo
}, open('colunas_boston.model', 'wb'))

distorcoes = []
K = range(1, 16)
for i in K:
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42).fit(dados_normalizados)
    distorcoes.append(kmeans.inertia_)

x0, y0 = K[0], distorcoes[0]
x1, y1 = K[-1], distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x, y = K[i], distorcoes[i]
    numerador = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    denominador = math.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    distancias.append(numerador / denominador)

numero_otimo_clusters = K[distancias.index(max(distancias))]
print('Número ótimo de clusters:', numero_otimo_clusters)

fig, ax = plt.subplots()
ax.plot(K, distorcoes, marker='o')
ax.plot([x0, x1], [y0, y1], linestyle='--', color='gray')
ax.axvline(numero_otimo_clusters, color='red', linestyle=':')
ax.set(xlabel='Número de clusters', ylabel='Distorção (inércia)', title='Método do cotovelo')
ax.grid()
fig.savefig('cotovelo_boston.png')

boston_kmeans = KMeans(n_clusters=numero_otimo_clusters, n_init=10, random_state=42).fit(dados_normalizados)
dump(boston_kmeans, open('boston_kmeans.model', 'wb'))

dados['grupo'] = boston_kmeans.labels_
print(dados['grupo'].value_counts().sort_index())
