library(randomForest)
library(caret)

treino <- read.csv("treino.csv")
teste  <- read.csv("teste.csv")

colunas_categoricas <- c("Gender", "family_history_with_overweight", "FAVC", "CAEC", 
                         "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad")

for(coluna in colunas_categoricas) {

  treino[[coluna]] <- as.factor(treino[[coluna]])

  teste[[coluna]]  <- factor(teste[[coluna]], levels = levels(treino[[coluna]]))
}

set.seed(1) 

modelo_rf <- randomForest(NObeyesdad ~ ., data = treino, ntree = 100, importance = TRUE)

previsoes <- predict(modelo_rf, teste)

matriz_completa <- confusionMatrix(previsoes, teste$NObeyesdad)
print(matriz_completa)

# 4. Bônus: Gráfico de Importância
varImpPlot(modelo_rf)
