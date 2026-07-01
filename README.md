# 📊 Predição de Níveis de Obesidade: Comparativo de Machine Learning entre R e Python

Este repositório contém a implementação e a análise comparativa de modelos de classificação supervisionada desenvolvidos como trabalho prático da disciplina de Inteligência Artificial do curso de Engenharia de Software da Universidade Católica de Brasília (UCB).

O objetivo central do projeto é prever o nível de obesidade de indivíduos com base em seus hábitos alimentares e condições físicas, contrastando o desempenho, a estabilidade e a implementação de três algoritmos nas linguagens **R** e **Python**.

## 👥 Equipe e Orientação
* **Autores:** Dimitri Kael, Enzo Gomide Martins, Guilherme Ayala, Gustavo Campos e Nicole Flaminio.
* **Orientador:** Prof. William Roberto Malvezzi.

## 🗂️ Sobre o Dataset
Os dados utilizados são provenientes do repositório público da UCI Machine Learning Repository: *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*.
* **Volume:** 2.111 registros.
* **Características:** 17 atributos (variáveis numéricas e categóricas).
* **Variável-alvo (`NObeyesdad`):** 7 categorias (Peso Insuficiente, Peso Normal, Sobrepeso Nível I, Sobrepeso Nível II, Obesidade Tipo I, Obesidade Tipo II e Obesidade Tipo III).
* O conjunto de dados original e a versão com os dados sintéticos gerados via SMOTE estão incluídos no pré-processamento.

## 🛠️ Tecnologias e Bibliotecas Utilizadas
A metodologia do projeto exigiu rigorosa equivalência nos testes. As partições de Treino (75%) e Teste (25%) e a semente randômica (seed) foram padronizadas para garantir que ambas as linguagens avaliassem os mesmos cenários.

* **Python:** `pandas`, `numpy`, `scikit-learn` (`GaussianNB`, `DecisionTreeClassifier`, `RandomForestClassifier`), `matplotlib`, `seaborn`.
* **R:** `tidyverse`, `dplyr`, `caret` (particionamento e matrizes), `e1071` (Naive Bayes), `rpart` (Árvore de Decisão), `randomForest`.

## 🤖 Modelos Avaliados
A abordagem metodológica avaliou três algoritmos distintos, buscando entender o comportamento estatístico de cada um frente à alta correlação das variáveis do dataset:

1. **Naive Bayes:** Testado para avaliar a probabilidade condicional. 
2. **Árvore de Decisão:** Utilizada pela sua alta explicabilidade (Visualização de nós) e recursividade na separação dos dados.
3. **Random Forest (Ensemble):** Implementado para contornar o *overfitting* da Árvore de Decisão por meio da criação de múltiplas árvores e votação majoritária (Bagging).

## 📈 Principais Resultados e Discussão Crítica

A pesquisa revelou que a natureza dos dados (fortemente correlacionada) dita o sucesso do modelo escolhido:

* 🏆 **Random Forest (Melhor Desempenho):** Alcançou a maior acurácia (acima de **92%** em Python e **94%** em R) e extrema estabilidade. O uso de múltiplas árvores distribuiu o processamento e evitou o ajuste excessivo aos dados de treino, contornando a complexidade das zonas de transição (ex: limites entre Sobrepeso Nível I e II). Variáveis numéricas como Peso e Altura mostraram-se as mais importantes para a floresta.
* 🥈 **Árvore de Decisão:** Apresentou ótimo desempenho (acurácia na casa dos **86% - 90%**), compreendendo bem a correlação entre atributos por meio de sucessivos cortes de dados. Contudo, exigiu controle para não sofrer com crescimento ou ajuste excessivo ao treino.
* ⚠️ **Naive Bayes (Menor Desempenho):** Obteve o pior resultado global (acurácia na casa dos **62% - 64%**). A queda de eficácia justifica-se pela sua premissa de independência condicional: o algoritmo ignora as fortes relações intrínsecas da base (como o impacto da dieta na altura/peso) e é penalizado pela sobreposição de classes do meio da tabela.

## 🚀 Como Executar o Projeto

Para reproduzir o ambiente e testar os modelos localmente:

**1. Clone o repositório**
```bash
git clone [https://github.com/SeuUsuario/NomeDoRepositorio.git](https://github.com/SeuUsuario/NomeDoRepositorio.git)
cd NomeDoRepositorio
2. Para rodar a versão em R

Abra a pasta no RStudio.

Instale os pacotes necessários: install.packages(c("tidyverse", "rpart", "caret", "e1071", "randomForest"))

Execute os scripts desejados para gerar os modelos e as exportações dos arquivos .csv.

3. Para rodar a versão em Python

É recomendado o uso de um ambiente virtual (venv).

Instale as dependências:

Bash
pip install pandas scikit-learn matplotlib seaborn
Certifique-se de que os arquivos treino.csv e teste.csv gerados pelo script base estejam no mesmo diretório antes de rodar os scripts (python naive_bayes.py, etc).
