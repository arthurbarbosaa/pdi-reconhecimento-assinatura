# Reconhecimento de Assinatura Manuscrita

Sistema de verificação de assinaturas manuscritas utilizando técnicas de Processamento Digital de Imagens (PDI) e Machine Learning. O projeto classifica assinaturas como genuínas ou forjadas através da extração de características (Hu Moments, HOG, espessura do traço e razão de área) e classificação com SVM.

## 📋 Requisitos

- Python 3.7 ou superior
- pip (gerenciador de pacotes Python)

## 🚀 Instalação

1. **Clone o repositório ou navegue até a pasta do projeto:**
   ```bash
   cd pdi-reconhecimento-assinatura
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Estrutura do Projeto

```
pdi-reconhecimento-assinatura/
├── src/
│   ├── experiment.py      # Arquivo principal - orquestra o pipeline
│   ├── dataset.py         # Construção do dataset
│   ├── preprocessing.py   # Pré-processamento de imagens
│   ├── features.py        # Extração de características
│   └── train_model.py     # Treinamento e avaliação do modelo
├── signatures/
│   ├── full_org/          # Assinaturas genuínas (original_*.png)
│   └── full_forg/         # Assinaturas forjadas (forgeries_*.png)
├── requirements.txt       # Dependências do projeto
└── README.md             # Este arquivo
```

## 🎯 Como Executar

### Execução Básica

Execute o arquivo principal do experimento:

```bash
python -m src.experiment
```

### Configuração do Experimento

Para personalizar o experimento, edite o arquivo `src/experiment.py`:

```python
# Escolha quais pessoas incluir no treinamento
person_ids = [1, 2, 3, 4, 5]  # Altere aqui

# Quantas amostras usar de cada pessoa (padrão: 24 = todas)
samples_per_person = 24  # Altere aqui
```

**Importante:** Certifique-se de que as imagens existem nos diretórios:
- `signatures/full_org/original_{person_id}_{i}.png`
- `signatures/full_forg/forgeries_{person_id}_{i}.png`

## 📊 Saída do Programa

O programa exibe:

1. **Número de amostras processadas:**
   ```
   Dataset montado: 240 amostra(s) processada(s) com sucesso
   ```

2. **Acurácia do modelo:**
   ```
   Acurácia: 95.83%
   ```

3. **Matriz de confusão:**
   ```
   Matriz de confusão:
                   Previsto
                 Forjada  Genuina
   Real Forjada     35       1
        Genuína      2      36
   ```

4. **Métricas de segurança:**
   ```
   FAR: 2.78%
   FRR: 5.56%
   ```

### Explicação das Métricas

- **Acurácia**: Porcentagem de classificações corretas
- **FAR (False Acceptance Rate)**: Taxa de falsa aceitação - assinaturas forjadas aceitas como genuínas
- **FRR (False Rejection Rate)**: Taxa de falsa rejeição - assinaturas genuínas rejeitadas como forjadas

## 🔧 Modos de Extração de Features

O projeto suporta diferentes modos de extração de características. Para alterar, modifique o parâmetro `mode` em `src/experiment.py`:

```python
X, y = build_dataset(samples, mode="full")  # Altere aqui
```

**Modos disponíveis:**
- `"full"`: Todas as features (Hu Moments + HOG + espessura + área)
- `"hu"`: Apenas Hu Moments (7 valores)
- `"hog"`: Apenas HOG (centenas de valores)
- `"hu_extra"`: Hu Moments + espessura + área (9 valores)

## 📝 Dependências

- `opencv-python`: Processamento de imagens
- `numpy`: Operações numéricas
- `matplotlib`: Visualização (se necessário)
- `scikit-learn`: Machine Learning (SVM)

## 🔍 Pipeline do Sistema

1. **Pré-processamento**: Carrega, inverte, normaliza, recorta e centraliza as imagens
2. **Extração de Features**: Extrai características (Hu Moments, HOG, espessura, área)
3. **Construção do Dataset**: Monta matriz X (features) e array y (labels)
4. **Treinamento**: Treina classificador SVM com divisão treino/teste (70%/30%)
5. **Avaliação**: Calcula acurácia, matriz de confusão, FAR e FRR

## ⚠️ Troubleshooting

### Erro: "É necessário pelo menos 2 amostras para treinar o modelo"
- Verifique se os caminhos das imagens estão corretos
- Confirme que as imagens existem nos diretórios `signatures/full_org/` e `signatures/full_forg/`
- Verifique se os `person_ids` configurados correspondem a imagens existentes

### Erro: "Unable to load image from: ..."
- Verifique se o caminho da imagem está correto
- Confirme que o arquivo existe e está no formato PNG
- Verifique permissões de leitura do arquivo

### Erro de importação de módulos
- Certifique-se de executar como módulo: `python -m src.experiment`
- Verifique se está na raiz do projeto
- Confirme que todas as dependências foram instaladas
