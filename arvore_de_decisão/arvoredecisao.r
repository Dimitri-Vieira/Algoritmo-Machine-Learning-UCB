library(tidyverse)
library(rpart)
library(caret)

df <- read.csv('ObesityDataSet_raw_and_data_sinthetic.csv')

X <- df %>% select(-NObeyesdad)
y <- df$NObeyesdad

X_codificado <- model.matrix(~ . - 1, data = X)
X_codificado <- as.data.frame(X_codificado)

dados_completos <- cbind(X_codificado, NObeyesdad = y)

set.seed(42)
indice_treino <- createDataPartition(dados_completos$NObeyesdad, p = 0.75, list = FALSE)
X_train <- dados_completos[indice_treino, ]
X_test <- dados_completos[-indice_treino, ]

modelo_dt <- rpart(NObeyesdad ~ ., 
                   data = X_train, 
                   method = "class",
                   control = rpart.control(cp = 0.01))


previsoes <- predict(modelo_dt, X_test, type = "class")

print("--- Matriz de Confusão ---")
matriz_confusao <- confusionMatrix(previsoes, as.factor(X_test$NObeyesdad))
print(matriz_confusao)

print("\n--- Relatório por Classe (Precision / Recall) ---")
print(matriz_confusao$byClass)