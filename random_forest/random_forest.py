import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

df_train = pd.read_csv('treino.csv')
df_test = pd.read_csv('teste.csv')

X_train = df_train.drop('NObeyesdad', axis=1)
y_train = df_train['NObeyesdad']

X_test = df_test.drop('NObeyesdad', axis=1)
y_test = df_test['NObeyesdad']

X_train_codificado = pd.get_dummies(X_train)
X_test_codificado = pd.get_dummies(X_test)

X_test_codificado = X_test_codificado.reindex(columns=X_train_codificado.columns, fill_value=0)

modelo_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=1, n_jobs=-1)
modelo_rf.fit(X_train_codificado, y_train)

previsoes = modelo_rf.predict(X_test_codificado)

print("--- Matriz de Confusão ---")
print(confusion_matrix(y_test, previsoes))

print("\n--- Relatório por Classe ---")
print(classification_report(y_test, previsoes))


importancias = modelo_rf.feature_importances_

tabela_importancia = pd.DataFrame({
    'Atributo': X_train_codificado.columns,
    'Importancia': importancias
})

tabela_importancia = tabela_importancia.sort_values(by='Importancia', ascending=False)

print("\n--- Top 10 Atributos Mais Importantes ---")
print(tabela_importancia.head(10)) 

plt.figure(figsize=(12, 8))
top_15_atributos = tabela_importancia.head(15)
plt.barh(top_15_atributos['Atributo'], top_15_atributos['Importancia'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Importância (Redução da Impureza de Gini)')
plt.title('Top 15 Características Mais Importantes - Random Forest')
plt.tight_layout()
plt.show()
