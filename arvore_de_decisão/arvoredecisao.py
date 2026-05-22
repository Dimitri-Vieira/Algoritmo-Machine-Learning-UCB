import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

try:
    base_treinamento = pd.read_csv('treino.csv')
    base_teste = pd.read_csv('teste.csv')
    print("Bases de Treino e Teste carregadas com sucesso.")
except FileNotFoundError:
    print("ERRO: Os arquivos 'treino.csv' e 'teste.csv' não foram encontrados.")
    print("Por favor, rode primeiro o script do R para gerar esses arquivos.")
    exit()

X_treino = base_treinamento.drop('NObeyesdad', axis=1)
y_treino = base_treinamento['NObeyesdad']

X_teste = base_teste.drop('NObeyesdad', axis=1)
y_teste = base_teste['NObeyesdad']

X_treino_codificado = pd.get_dummies(X_treino)
X_teste_codificado = pd.get_dummies(X_teste)

colunas_comuns = X_treino_codificado.columns.intersection(X_teste_codificado.columns)
X_treino_codificado = X_treino_codificado[colunas_comuns]
X_teste_codificado = X_teste_codificado[colunas_comuns]

modelo_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
modelo_dt.fit(X_treino_codificado, y_treino)

previsoes = modelo_dt.predict(X_teste_codificado)

mapa_abreviacoes = {
    'Insufficient_Weight': 'I.F',
    'Normal_Weight': 'N.W',
    'Overweight_Level_I': 'OW.L1',
    'Overweight_Level_II': 'OW.L2',
    'Obesity_Type_I': 'O.T1',
    'Obesity_Type_II': 'O.T2',
    'Obesity_Type_III': 'O.T3'
}

mapa_legendas = {
    'I.F': 'peso insuficiente',
    'N.W': 'peso normal',
    'OW.L1': 'sobrepeso nível 1',
    'OW.L2': 'sobrepeso nível 2',
    'O.T1': 'obesidade tipo 1',
    'O.T2': 'obesidade tipo 2',
    'O.T3': 'obesidade tipo 3'
}

classes = sorted(y_teste.unique())
abreviacoes = [mapa_abreviacoes[classe] for classe in classes]

matriz_confusao = confusion_matrix(y_teste, previsoes, labels=classes)

print("\n--- Matriz de Confusão ---\n")

cabecalho = "     " + "".join(f"{abrev:>6}" for abrev in abreviacoes) + "    LEGENDA:"
print(cabecalho)

for i, abrev in enumerate(abreviacoes):
    linha = f"{abrev:>4}" + "".join(f"{valor:>6}" for valor in matriz_confusao[i])
    legenda = f"    {abrev:>5} = {mapa_legendas[abrev]}"
    print(f"{linha}{legenda}")

acuracia = accuracy_score(y_teste, previsoes)
print(f"\n Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")

print("\n--- Relatório por Classe (Precision / Recall) ---")
print(classification_report(y_teste, previsoes, target_names=abreviacoes))
