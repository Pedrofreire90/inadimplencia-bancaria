"""
preprocessamento.py
--------------------
Limpeza, análise exploratória (EDA) e preparação dos dados
para o pipeline de Machine Learning.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Configuração visual global
PALETTE = {"0": "#2196F3", "1": "#E53935"}
CORES = ["#2196F3", "#E53935"]
FIGDIR = Path("reports/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def plotar_distribuicao_target(df: pd.DataFrame) -> None:
    """Gráfico de barras da distribuição da variável target."""
    counts = df["inadimplente"].value_counts()
    labels = ["Adimplente", "Inadimplente"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=CORES, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")

    for i, (bar, pct) in enumerate(zip(bars, counts / counts.sum() * 100)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="center", va="center",
            color="white", fontsize=13, fontweight="bold",
        )

    ax.set_title("Distribuição do Target — Inadimplência", fontweight="bold", pad=14)
    ax.set_ylabel("Número de Clientes")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(FIGDIR / "01_distribuicao_target.png", dpi=150)
    plt.close()
    print("[✔] Figura salva: 01_distribuicao_target.png")


def plotar_variaveis_numericas(df: pd.DataFrame) -> None:
    """Histogramas das variáveis numéricas por classe."""
    cols = ["idade", "renda_mensal", "score_credito", "valor_divida",
            "meses_emprego", "razao_divida_renda"]
    titulos = ["Idade", "Renda Mensal (R$)", "Score de Crédito",
               "Valor da Dívida (R$)", "Meses de Emprego", "Razão Dívida/Renda"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, col, titulo in zip(axes, cols, titulos):
        for classe, cor, label in zip([0, 1], CORES, ["Adimplente", "Inadimplente"]):
            subset = df[df["inadimplente"] == classe][col]
            ax.hist(subset, bins=30, alpha=0.6, color=cor, label=label, edgecolor="none")
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=9)

    fig.suptitle("Distribuição das Variáveis Numéricas por Classe", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGDIR / "02_variaveis_numericas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✔] Figura salva: 02_variaveis_numericas.png")


def plotar_correlacao(df: pd.DataFrame) -> None:
    """Heatmap de correlação entre variáveis numéricas."""
    numericas = df.select_dtypes(include=np.number)
    corr = numericas.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("Matriz de Correlação", fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(FIGDIR / "03_correlacao.png", dpi=150)
    plt.close()
    print("[✔] Figura salva: 03_correlacao.png")


def plotar_score_por_inadimplencia(df: pd.DataFrame) -> None:
    """Boxplot do score de crédito por classe."""
    fig, ax = plt.subplots(figsize=(7, 5))
    df_plot = df.copy()
    df_plot["Classe"] = df_plot["inadimplente"].map({0: "Adimplente", 1: "Inadimplente"})

    sns.boxplot(
        data=df_plot, x="Classe", y="score_credito",
        palette={"Adimplente": CORES[0], "Inadimplente": CORES[1]},
        width=0.45, linewidth=1.2, ax=ax,
    )
    ax.set_title("Score de Crédito vs. Inadimplência", fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score de Crédito")
    plt.tight_layout()
    plt.savefig(FIGDIR / "04_score_por_classe.png", dpi=150)
    plt.close()
    print("[✔] Figura salva: 04_score_por_classe.png")


def executar_eda(df: pd.DataFrame) -> None:
    """Executa toda a análise exploratória."""
    print("\n--- ANÁLISE EXPLORATÓRIA DE DADOS ---")
    print(f"Shape: {df.shape}")
    print(f"\nValores nulos:\n{df.isnull().sum()}")
    print(f"\nEstatísticas descritivas:\n{df.describe().round(2)}")
    print(f"\nDistribuição do target:\n{df['inadimplente'].value_counts()}")

    plotar_distribuicao_target(df)
    plotar_variaveis_numericas(df)
    plotar_correlacao(df)
    plotar_score_por_inadimplencia(df)
    print("\n[✔] EDA concluída. Figuras em 'reports/figures/'.\n")


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

def preprocessar(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Executa o pipeline completo de pré-processamento:
    - Encoding de variáveis categóricas
    - Divisão treino/teste
    - Escalonamento (StandardScaler)
    - Balanceamento com SMOTE

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    # Encoding das categóricas
    le = LabelEncoder()
    for col in ["estado_civil", "tipo_conta"]:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["inadimplente"])
    y = df["inadimplente"]
    feature_names = X.columns.tolist()

    # Divisão treino / teste (estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Escalonamento
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Balanceamento com SMOTE (apenas no treino)
    smote = SMOTE(random_state=seed)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"[✔] Pré-processamento concluído.")
    print(f"    Treino: {X_train.shape[0]:,} amostras | Teste: {X_test.shape[0]:,} amostras")
    print(f"    Distribuição após SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, feature_names
