import pandas as pd
from pickle import load

nova_residencia = {
    'CRIM': 0.25,
    'ZN': 0.0,
    'INDUS': 8.14,
    'CHAS': 0,
    'NOX': 0.538,
    'RM': 6.1,
    'AGE': 85.0,
    'DIS': 4.0,
    'RAD': 4,
    'TAX': 307,
    'PTRATIO': 21.0,
    'B': 390.0,
    'LSTAT': 12.5,
    'MEDV': 20.0
}

normalizador = load(open('normalizador_boston.model', 'rb'))
colunas = load(open('colunas_boston.model', 'rb'))
boston_kmeans = load(open('boston_kmeans.model', 'rb'))

colunas_numericas = colunas['colunas_numericas']
colunas_categoricas = colunas['colunas_categoricas']
colunas_dummies = colunas['colunas_dummies']
colunas_modelo = colunas['colunas_modelo']

nova_residencia = pd.DataFrame([nova_residencia])

nova_numericos = pd.DataFrame(
    normalizador.transform(nova_residencia[colunas_numericas]),
    columns=colunas_numericas
)
nova_categoricos = pd.get_dummies(nova_residencia[colunas_categoricas].astype(str), prefix_sep='_', dtype=int)
nova_categoricos = nova_categoricos.reindex(columns=colunas_dummies, fill_value=0)

nova_normalizada = nova_numericos.join(nova_categoricos)[colunas_modelo]

grupo = boston_kmeans.predict(nova_normalizada)[0]
print('Grupo da nova residência:', grupo)

centroide = pd.DataFrame([boston_kmeans.cluster_centers_[grupo]], columns=colunas_modelo)

centroide_numerico = pd.DataFrame(
    normalizador.inverse_transform(centroide[colunas_numericas]),
    columns=colunas_numericas
)

centroide_desnormalizado = centroide_numerico.iloc[0].to_dict()
for coluna in colunas_categoricas:
    dummies_coluna = [c for c in colunas_dummies if c.startswith(coluna + '_')]
    categoria = centroide[dummies_coluna].iloc[0].idxmax()
    centroide_desnormalizado[coluna] = categoria[len(coluna) + 1:]

centroide_desnormalizado = pd.Series(centroide_desnormalizado)[list(nova_residencia.columns)]

print('Centroide desnormalizado:')
print(centroide_desnormalizado.to_string())
