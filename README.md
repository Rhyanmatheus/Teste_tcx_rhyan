# Classificação de Imagens com Redes Neurais

Este projeto implementa um modelo de rede neural para classificar imagens de frutas, utilizando o **Fruits 360 Dataset**. O modelo é construído com **Transfer Learning** usando a MobileNetV2 e atinge a acurácia mínima esperada de 70%.

---

## 📝 Ideia do Projeto

O objetivo é criar um modelo capaz de identificar e classificar imagens de frutas entre três categorias: **Maçã**, **Banana** e **Morango**. Para isso:
1. O conjunto de dados é dividido em treinamento e teste.
2. Técnicas de **Data Augmentation** são aplicadas para melhorar a robustez do modelo.
3. A arquitetura MobileNetV2, pré-treinada no ImageNet, é utilizada para extrair características, e camadas adicionais são adicionadas para a classificação específica.

---

## 📂 Passo a Passo do Código

### 1. **Importação de Bibliotecas**
As bibliotecas necessárias são importadas, incluindo o TensorFlow, MobileNetV2 e ferramentas de pré-processamento de imagens.

### 2. **Carregamento dos Dados**
As imagens são carregadas de diretórios organizados em pastas de treinamento e teste:
- **Data Augmentation** é aplicado no conjunto de treinamento para aumentar a diversidade das imagens.
- As imagens são normalizadas (valores dos pixels entre 0 e 1).

### 3. **Transfer Learning**
- O modelo **MobileNetV2**, pré-treinado no ImageNet, é utilizado para extrair características.
- As camadas superiores da MobileNetV2 são congeladas para preservar os pesos pré-treinados.

### 4. **Construção do Modelo**
O modelo é construído adicionando:
- Camada de pooling global para redução de dimensionalidade.
- Camada totalmente conectada com 128 neurônios e ativação ReLU.
- Camada de saída com 3 neurônios (uma para cada classe) e ativação softmax.

### 5. **Treinamento do Modelo**
O modelo é compilado com o otimizador Adam e a perda categórica cruzada. É treinado por 10 épocas com validação em dados de teste.

### 6. **Salvamento e Avaliação**
- O modelo treinado é salvo em um arquivo (`modelo_frutas_treinado.h5`).
- O desempenho é avaliado nos dados de teste.

### 7. **Predição de Novas Imagens**
O modelo carrega imagens de uma pasta separada (`test_images/`), processa cada imagem e exibe a classe predita e a acurácia associada.

---


