import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("--- 1. CARREGAMENTO DOS DADOS PADRONIZADOS ---")

try:
    base_treinamento = pd.read_csv('treino.csv')
    base_teste = pd.read_csv('teste.csv')
    print("Bases de Treino e Teste carregadas com sucesso.")
except FileNotFoundError:
    print("ERRO: Os arquivos 'treino.csv' e 'teste.csv' não foram encontrados.")
    print("Por favor, rode primeiro o script do R para gerar esses arquivos.")
    exit()

print("\n--- 2. TRATAMENTO (LABEL ENCODING PARA PYTHON) ---")


colunas_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in colunas_categoricas:
    le = LabelEncoder()
    le.fit(pd.concat([base_treinamento[col], base_teste[col]]))
    
    base_treinamento[col] = le.transform(base_treinamento[col])
    base_teste[col] = le.transform(base_teste[col])

X_treinamento = base_treinamento.drop('NObeyesdad', axis=1)
y_treinamento = base_treinamento['NObeyesdad']

X_teste = base_teste.drop('NObeyesdad', axis=1)
y_teste = base_teste['NObeyesdad']


print("\n--- 3. TREINAMENTO DO CLASSIFICADOR (NAIVE BAYES) ---")
# Instanciando o modelo Gaussiano (ideal para nossos dados numéricos)
classificador = GaussianNB()
classificador.fit(X_treinamento, y_treinamento)


print("\n--- 4. PREVISÕES E AVALIAÇÃO ---")
# Fazendo as previsões com a base de teste
previsoes = classificador.predict(X_teste)

# Acurácia
acuracia = accuracy_score(y_teste, previsoes)
print(f"Acurácia Global: {acuracia:.4f} ({acuracia*100:.2f}%)\n")

# Matriz de Confusão
print("--- Matriz de Confusão ---")
# pd.crosstab cria uma matriz visualmente mais bonita e parecida com a do R
matriz_confusao = pd.crosstab(y_teste, previsoes, rownames=['Real'], colnames=['Previsto'], margins=True)
print(matriz_confusao)

# Relatório Completo (Acurácia, Precisão, Revocação e F1-Score)
print("\n--- Relatório de Métricas (Exigência do Edital) ---")
relatorio = classification_report(y_teste, previsoes)
print(relatorio)