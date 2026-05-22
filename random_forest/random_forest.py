import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carregando os Dados (Já separados na sua estratégia de 75/25)
df_train = pd.read_csv('treino.csv')
df_test = pd.read_csv('teste.csv')

# 2. Separando as características (X) do alvo (y)
X_train = df_train.drop('NObeyesdad', axis=1)
y_train = df_train['NObeyesdad']

X_test = df_test.drop('NObeyesdad', axis=1)
y_test = df_test['NObeyesdad']

# 3. Tratamento de Variáveis Categóricas (One-Hot Encoding)
X_train_codificado = pd.get_dummies(X_train)
X_test_codificado = pd.get_dummies(X_test)

# 4. O TRUQUE DE ALINHAMENTO: Forçando o teste a ter as mesmas colunas do treino
X_test_codificado = X_test_codificado.reindex(columns=X_train_codificado.columns, fill_value=0)

# 5. Treinamento da Floresta (Força Bruta)
modelo_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=1, n_jobs=-1)
modelo_rf.fit(X_train_codificado, y_train)

# 6. Previsões e Avaliação
previsoes = modelo_rf.predict(X_test_codificado)

print("--- Matriz de Confusão ---")
print(confusion_matrix(y_test, previsoes))

print("\n--- Relatório por Classe ---")
print(classification_report(y_test, previsoes))

# ==========================================
# 7. EXTRAÇÃO DE INTELIGÊNCIA: Importância de Atributos
# ==========================================
# Extraindo as notas matemáticas calculadas pelas 100 árvores
importancias = modelo_rf.feature_importances_

# Criando a tabela relacionando o nome da coluna com a sua nota
tabela_importancia = pd.DataFrame({
    'Atributo': X_train_codificado.columns,
    'Importancia': importancias
})

# Ordenando do atributo mais forte para o mais fraco
tabela_importancia = tabela_importancia.sort_values(by='Importancia', ascending=False)

print("\n--- Top 10 Atributos Mais Importantes ---")
print(tabela_importancia.head(10)) 

# Gerando o Gráfico Visual
plt.figure(figsize=(12, 8))
# Desenhamos apenas os 15 mais fortes para o gráfico não ficar esmagado na tela
top_15_atributos = tabela_importancia.head(15)
plt.barh(top_15_atributos['Atributo'], top_15_atributos['Importancia'], color='skyblue')
plt.gca().invert_yaxis() # Inverte o eixo Y para o campeão ficar no topo da tela
plt.xlabel('Importância (Redução da Impureza de Gini)')
plt.title('Top 15 Características Mais Importantes - Random Forest')
plt.tight_layout()
plt.show()