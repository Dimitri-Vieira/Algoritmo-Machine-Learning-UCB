
rm(list = ls()) # Limpa o ambiente
library(dplyr)
library(caTools)
library(e1071)
library(caret)

# Lembre-se de ter o arquivo original na mesma pasta do script
base_dados <- read.csv("ObesityDataSet_raw.csv", stringsAsFactors = FALSE)


colunas_categoricas <- c("Gender", "family_history_with_overweight", "FAVC", "CAEC", 
                         "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad")

base_dados <- base_dados %>%
  mutate(
    Age = round(Age),
    Height = round(Height, 2),
    Weight = round(Weight, 2),
    FCVC = round(FCVC),
    NCP = round(NCP),
    CH2O = round(CH2O),
    FAF = round(FAF),
    TUE = round(TUE)
  ) %>%
  mutate(across(all_of(colunas_categoricas), as.factor))

set.seed(1)
divisao <- sample.split(base_dados$NObeyesdad, SplitRatio = 0.75)

base_treinamento <- subset(base_dados, divisao == TRUE)
base_teste <- subset(base_dados, divisao == FALSE)

write.csv(base_treinamento, "treino_oficial.csv", row.names = FALSE)
write.csv(base_teste, "teste_oficial.csv", row.names = FALSE)


classificador <- naiveBayes(NObeyesdad ~ ., data = base_treinamento)

previsoes <- predict(classificador, newdata = base_teste[-17])

matriz_confusao <- table(base_teste$NObeyesdad, previsoes)
print("--- Matriz de Confusão ---")
print(matriz_confusao)

print("--- Relatório Completo (Acurácia, Precisão, Revocação e F1-Score) ---")

confusionMatrix(matriz_confusao, mode = "prec_recall")
