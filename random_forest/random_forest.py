import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Preparação dos Dados
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Separando as 16 características (X) da classe que queremos prever (y)
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Tratamento obrigatório no Python: Convertendo colunas de texto em números binários (One-Hot Encoding)
X_codificado = pd.get_dummies(X)

# Dividindo o banco: 75% para a floresta aprender, 25% para testarmos a matriz de confusão
X_train, X_test, y_train, y_test = train_test_split(X_codificado, y, test_size=0.25, random_state=42)

# 2. O Treinamento da Floresta
# n_estimators = cria 100 árvores.
# n_jobs = -1 usa todos os núcleos do processador da sua máquina em paralelo (ótimo para acelerar o processo localmente)
modelo_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42, n_jobs=-1)

# A força bruta acontece aqui:
modelo_rf.fit(X_train, y_train)

# 3. Previsão e Avaliação
previsoes = modelo_rf.predict(X_test)

print("--- Matriz de Confusão ---")
print(confusion_matrix(y_test, previsoes))

print("\n--- Relatório por Classe (Precision / Recall) ---")
print(classification_report(y_test, previsoes))