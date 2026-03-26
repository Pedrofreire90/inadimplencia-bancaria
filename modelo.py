"""
modelo.py
---------
Treinamento, comparação e avaliação de modelos de Machine Learning
para previsão de inadimplência bancária.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

FIGDIR = Path("reports/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

CORES_MODELOS = {
    "Regressão Logística": "#5C6BC0",
    "Random Forest": "#26A69A",
    "Gradient Boosting": "#EF5350",
}


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def obter_modelos() -> dict:
    """Retorna dicionário com os modelos a comparar."""
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ),
    }


def treinar_modelos(modelos: dict, X_train, y_train) -> dict:
    """Treina todos os modelos e retorna os treinados."""
    treinados = {}
    for nome, modelo in modelos.items():
        print(f"  → Treinando: {nome}...")
        modelo.fit(X_train, y_train)
        treinados[nome] = modelo
    print("[✔] Todos os modelos treinados.\n")
    return treinados


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def avaliar_modelos(modelos: dict, X_test, y_test) -> pd.DataFrame:
    """
    Avalia todos os modelos e retorna DataFrame com métricas comparativas.
    """
    resultados = []
    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)

        resultados.append({
            "Modelo": nome,
            "Accuracy": report["accuracy"],
            "Precision (inadimp.)": report["1"]["precision"],
            "Recall (inadimp.)": report["1"]["recall"],
            "F1-Score (inadimp.)": report["1"]["f1-score"],
            "ROC-AUC": auc,
        })

    df_resultados = pd.DataFrame(resultados).set_index("Modelo")
    print("--- MÉTRICAS COMPARATIVAS ---")
    print(df_resultados.round(4).to_string())
    return df_resultados


def plotar_comparacao_metricas(df_resultados: pd.DataFrame) -> None:
    """Gráfico de barras comparando as métricas dos modelos."""
    metricas = ["Accuracy", "Precision (inadimp.)", "Recall (inadimp.)", "F1-Score (inadimp.)", "ROC-AUC"]
    cores = [CORES_MODELOS[m] for m in df_resultados.index]

    fig, axes = plt.subplots(1, len(metricas), figsize=(16, 5), sharey=False)

    for ax, metrica in zip(axes, metricas):
        valores = df_resultados[metrica]
        bars = ax.bar(range(len(valores)), valores.values, color=cores, edgecolor="white", width=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(valores)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in valores.index], fontsize=8)
        ax.set_title(metrica, fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("")

    fig.suptitle("Comparação de Modelos — Métricas de Avaliação", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGDIR / "05_comparacao_modelos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✔] Figura salva: 05_comparacao_modelos.png")


def plotar_matrizes_confusao(modelos: dict, X_test, y_test) -> None:
    """Plota as matrizes de confusão dos modelos lado a lado."""
    fig, axes = plt.subplots(1, len(modelos), figsize=(15, 4))

    for ax, (nome, modelo) in zip(axes, modelos.items()):
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Adimplente", "Inadimplente"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(nome, fontweight="bold", fontsize=10)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")

    fig.suptitle("Matrizes de Confusão", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGDIR / "06_matrizes_confusao.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[✔] Figura salva: 06_matrizes_confusao.png")


def plotar_curvas_roc(modelos: dict, X_test, y_test) -> None:
    """Plota as curvas ROC de todos os modelos sobrepostas."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for nome, modelo in modelos.items():
        y_prob = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.3f})", color=CORES_MODELOS[nome], lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Aleatório (AUC = 0.500)")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax.set_title("Curvas ROC — Comparação dos Modelos", fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGDIR / "07_curvas_roc.png", dpi=150)
    plt.close()
    print("[✔] Figura salva: 07_curvas_roc.png")


def plotar_importancia_features(modelo, feature_names: list, nome_modelo: str = "Random Forest") -> None:
    """Plota importância das features do melhor modelo (Random Forest)."""
    if not hasattr(modelo, "feature_importances_"):
        print(f"  Modelo '{nome_modelo}' não suporta feature_importances_. Pulando.")
        return

    importancias = pd.Series(modelo.feature_importances_, index=feature_names)
    importancias = importancias.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(importancias.index, importancias.values, color="#26A69A", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_title(f"Importância das Features — {nome_modelo}", fontweight="bold", pad=14)
    ax.set_xlabel("Importância")
    ax.set_xlim(0, importancias.max() * 1.18)
    plt.tight_layout()
    plt.savefig(FIGDIR / "08_importancia_features.png", dpi=150)
    plt.close()
    print("[✔] Figura salva: 08_importancia_features.png")


def selecionar_melhor_modelo(df_resultados: pd.DataFrame) -> str:
    """Retorna o nome do modelo com maior ROC-AUC."""
    melhor = df_resultados["ROC-AUC"].idxmax()
    print(f"\n🏆 Melhor modelo: {melhor} (ROC-AUC = {df_resultados.loc[melhor, 'ROC-AUC']:.4f})")
    return melhor
