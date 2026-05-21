# Instale os pacotes se não os tiver no seu ambiente: install.packages(c("randomForest", "caret", "e1071"))
library(randomForest)
library(caret)

# 1. Carregando os Dados
df <- read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Tipagem estrita: Garantimos que o alvo (7 classes) e as colunas categóricas sejam Fatores
df$Gender <- as.factor(df$Gender)
df$family_history_with_overweight <- as.factor(df$family_history_with_overweight)
df$FAVC <- as.factor(df$FAVC)
df$CAEC <- as.factor(df$CAEC)
df$SMOKE <- as.factor(df$SMOKE)
df$SCC <- as.factor(df$SCC)
df$CALC <- as.factor(df$CALC)
df$MTRANS <- as.factor(df$MTRANS)
df$NObeyesdad <- as.factor(df$NObeyesdad)
# df$Sistemas <- as.factor(df$Sistemas) # Repita para suas outras colunas de texto

# Dividindo o banco (75/25)
set.seed(42)
indice_treino <- createDataPartition(df$NObeyesdad, p = 0.75, list = FALSE)
treino <- df[indice_treino, ]
teste  <- df[-indice_treino, ]

# 2. O Treinamento da Floresta
# A sintaxe "Classe ~ ." manda o algoritmo usar as 16 colunas restantes. ntree = 100 árvores.
set.seed(42)
modelo_rf <- randomForest(NObeyesdad ~ ., data = treino, ntree = 100, importance = TRUE)

# 3. Previsão e Avaliação
previsoes <- predict(modelo_rf, teste)

# A função do caret já gera a matriz e todo o relatório de recall/precision por classe de uma vez
matriz_completa <- confusionMatrix(previsoes, teste$NObeyesdad)
print(matriz_completa)

# 4. Bônus: Gráfico de Importância
# Isso vai abrir uma janela mostrando quais das suas 16 variáveis definem mais as regras de negócio
varImpPlot(modelo_rf)