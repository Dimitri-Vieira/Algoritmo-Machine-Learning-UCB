library(randomForest)
library(caret)

# 1. Carregando os Dados (Já separados na proporção 75/25)
treino <- read.csv("treino.csv")
teste  <- read.csv("teste.csv")

# Lista das colunas que são texto/categorias e precisam virar Fatores
colunas_categoricas <- c("Gender", "family_history_with_overweight", "FAVC", "CAEC", 
                         "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad")

# TRUQUE DE ALINHAMENTO NO R: Convertendo para fator e garantindo níveis iguais
for(coluna in colunas_categoricas) {
  # Transforma a coluna do treino em Fator
  treino[[coluna]] <- as.factor(treino[[coluna]])
  
  # Transforma a coluna do teste forçando ela a ter as exatas categorias do treino
  teste[[coluna]]  <- factor(teste[[coluna]], levels = levels(treino[[coluna]]))
}

# A etapa de createDataPartition foi removida pois as bases já estão divididas!

# 2. O Treinamento da Floresta
# Travando a semente com o número 1, afetando a aleatoriedade da floresta
set.seed(1) 

# A sintaxe "NObeyesdad ~ ." manda usar todas as colunas para prever o alvo
modelo_rf <- randomForest(NObeyesdad ~ ., data = treino, ntree = 100, importance = TRUE)

# 3. Previsão e Avaliação
previsoes <- predict(modelo_rf, teste)

# Gerando a matriz de confusão e as métricas
matriz_completa <- confusionMatrix(previsoes, teste$NObeyesdad)
print(matriz_completa)

# 4. Bônus: Gráfico de Importância
varImpPlot(modelo_rf)