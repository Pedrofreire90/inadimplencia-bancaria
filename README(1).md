# 🏦 Previsão de Inadimplência Bancária

Projeto de Machine Learning para prever a probabilidade de inadimplência de clientes bancários, com análise exploratória de dados e pipeline de modelagem.

## 📋 Sobre o Projeto

Este projeto utiliza dados sintéticos realistas de clientes bancários para construir um modelo preditivo de inadimplência. O objetivo é identificar clientes com maior risco de não honrar seus compromissos financeiros.

**Problema:** Classificação binária — o cliente irá inadimplir? (1 = Sim, 0 = Não)

---

## 🚀 Tecnologias Utilizadas

- **Python 3.10+**
- **Pandas & NumPy** — manipulação e análise de dados
- **Matplotlib & Seaborn** — visualizações
- **Scikit-learn** — pré-processamento e modelos de ML
- **Imbalanced-learn** — tratamento de desbalanceamento (SMOTE)

---

## 📁 Estrutura do Projeto

```
inadimplencia_bancaria/
│
├── data/
│   └── clientes.csv          # Dataset gerado
│
├── src/
│   ├── gerar_dados.py        # Geração do dataset sintético
│   ├── preprocessamento.py   # Limpeza e feature engineering
│   └── modelo.py             # Treinamento e avaliação
│
├── reports/
│   └── figures/              # Gráficos gerados
│
├── main.py                   # Pipeline principal
├── requirements.txt
└── README.md
```

---

## ⚙️ Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/Pedrofreire90/inadimplencia-bancaria.git
cd inadimplencia-bancaria

# 2. Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute o pipeline completo
python main.py
```

---

## 📊 Resultados

O pipeline executa as seguintes etapas:

1. **Geração de dados** — cria dataset com 5.000 clientes e 12 variáveis
2. **EDA** — análise exploratória com gráficos salvos em `reports/figures/`
3. **Pré-processamento** — encoding, escalonamento e balanceamento com SMOTE
4. **Treinamento** — compara Regressão Logística, Random Forest e Gradient Boosting
5. **Avaliação** — métricas, matriz de confusão e curva ROC

### Variáveis do Modelo

| Variável | Descrição |
|---|---|
| `idade` | Idade do cliente |
| `renda_mensal` | Renda mensal em R$ |
| `score_credito` | Score de crédito (0–1000) |
| `num_emprestimos` | Número de empréstimos ativos |
| `valor_divida` | Valor total da dívida em R$ |
| `meses_emprego` | Tempo de emprego em meses |
| `razao_divida_renda` | Dívida / Renda (DTI ratio) |
| `historico_atraso` | Já atrasou pagamentos? |
| `num_dependentes` | Número de dependentes |
| `tipo_conta` | Tipo de conta bancária |
| `estado_civil` | Estado civil |
| `inadimplente` | **Target** — inadimpliu? |

---

## 📈 Métricas Avaliadas

- **Accuracy** — acurácia geral
- **Precision / Recall / F1-Score** — foco na classe positiva (inadimplente)
- **ROC-AUC** — capacidade discriminativa do modelo
- **Matriz de Confusão** — erros tipo I e II

---

## 👨‍💻 Autor

**Pedro Freire**  
[GitHub](https://github.com/Pedrofreire90) · [LinkedIn](https://linkedin.com/in/pedro-freire)
