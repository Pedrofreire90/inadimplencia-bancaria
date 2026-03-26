"""
gerar_dados.py
--------------
Gera um dataset sintético realista de clientes bancários
para o projeto de previsão de inadimplência.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Semente para reprodutibilidade
SEED = 42


def gerar_dataset(n_clientes: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """
    Gera um DataFrame com dados sintéticos de clientes bancários.

    Args:
        n_clientes: Número de registros a gerar.
        seed: Semente aleatória.

    Returns:
        DataFrame com as features e o target (inadimplente).
    """
    rng = np.random.default_rng(seed)

    # --- Dados demográficos ---
    idade = rng.integers(18, 75, size=n_clientes)
    estado_civil = rng.choice(
        ["Solteiro", "Casado", "Divorciado", "Viúvo"],
        size=n_clientes,
        p=[0.35, 0.45, 0.15, 0.05],
    )
    num_dependentes = rng.choice([0, 1, 2, 3, 4], size=n_clientes, p=[0.30, 0.25, 0.25, 0.15, 0.05])

    # --- Situação financeira ---
    renda_mensal = np.clip(
        rng.normal(loc=4500, scale=2500, size=n_clientes), 1200, 30000
    ).round(2)
    score_credito = np.clip(
        rng.normal(loc=600, scale=150, size=n_clientes), 0, 1000
    ).astype(int)
    num_emprestimos = rng.choice([0, 1, 2, 3, 4, 5], size=n_clientes, p=[0.20, 0.30, 0.25, 0.15, 0.07, 0.03])
    valor_divida = np.clip(
        rng.exponential(scale=8000, size=n_clientes), 0, 120000
    ).round(2)
    historico_atraso = rng.choice([0, 1], size=n_clientes, p=[0.70, 0.30])

    # --- Situação profissional ---
    meses_emprego = np.clip(
        rng.exponential(scale=36, size=n_clientes), 0, 480
    ).astype(int)

    # --- Tipo de conta ---
    tipo_conta = rng.choice(
        ["Corrente", "Poupança", "Salário", "Premium"],
        size=n_clientes,
        p=[0.40, 0.30, 0.20, 0.10],
    )

    # --- Feature derivada: razão dívida/renda (DTI) ---
    razao_divida_renda = np.clip(valor_divida / (renda_mensal * 12), 0, 5).round(4)

    # --- Target: inadimplente ---
    # Calcula probabilidade de inadimplência com base nas features
    # Taxa base ajustada para ~25% de inadimplência (mais realista para modelos)
    score_norm = score_credito / 1000.0
    renda_norm = renda_mensal / 30000.0

    prob_base = (
        0.30
        + 0.35 * historico_atraso
        - 0.40 * score_norm
        + 0.20 * razao_divida_renda
        + 0.04 * num_emprestimos
        - 0.20 * renda_norm
        + 0.02 * num_dependentes
        - 0.0002 * meses_emprego
    )
    prob_inadimplencia = np.clip(prob_base, 0.03, 0.95)
    inadimplente = rng.binomial(1, prob_inadimplencia)

    # --- Monta o DataFrame ---
    df = pd.DataFrame(
        {
            "idade": idade,
            "estado_civil": estado_civil,
            "num_dependentes": num_dependentes,
            "renda_mensal": renda_mensal,
            "score_credito": score_credito,
            "num_emprestimos": num_emprestimos,
            "valor_divida": valor_divida,
            "historico_atraso": historico_atraso,
            "meses_emprego": meses_emprego,
            "tipo_conta": tipo_conta,
            "razao_divida_renda": razao_divida_renda,
            "inadimplente": inadimplente,
        }
    )

    return df


def salvar_dataset(df: pd.DataFrame, caminho: str = "data/clientes.csv") -> None:
    """Salva o DataFrame em CSV."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    print(f"[✔] Dataset salvo em '{caminho}' — {len(df):,} registros.")


if __name__ == "__main__":
    df = gerar_dataset()
    salvar_dataset(df)
    print(df.head())
    print(f"\nDistribuição do target:\n{df['inadimplente'].value_counts(normalize=True).round(3)}")
