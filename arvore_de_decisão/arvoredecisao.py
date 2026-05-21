import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

X_codificado = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_codificado, y, test_size=0.25, random_state=42)

modelo_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)

modelo_dt.fit(X_train, y_train)

previsoes = modelo_dt.predict(X_test)

print("--- Matriz de Confusão ---")
print(confusion_matrix(y_test, previsoes))

print("\n--- Relatório por Classe (Precision / Recall) ---")
print(classification_report(y_test, previsoes))