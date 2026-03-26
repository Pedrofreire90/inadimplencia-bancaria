"""
main.py
-------
Pipeline principal do projeto de Previsão de Inadimplência Bancária.

Executa em sequência:
  1. Geração do dataset sintético
  2. Análise Exploratória de Dados (EDA)
  3. Pré-processamento e balanceamento
  4. Treinamento e comparação de modelos
  5. Avaliação final com gráficos
"""

import sys
from pathlib import Path

# Garante que o diretório src está no path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gerar_dados import gerar_dataset, salvar_dataset
from preprocessamento import executar_eda, preprocessar
from modelo import (
    obter_modelos,
    treinar_modelos,
    avaliar_modelos,
    plotar_comparacao_metricas,
    plotar_matrizes_confusao,
    plotar_curvas_roc,
    plotar_importancia_features,
    selecionar_melhor_modelo,
)

DATA_PATH = "data/clientes.csv"


def main():
    print("=" * 55)
    print("  PREVISÃO DE INADIMPLÊNCIA BANCÁRIA — Pipeline ML")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Geração dos dados
    # ------------------------------------------------------------------
    print("\n[1/5] Gerando dataset sintético...")
    df = gerar_dataset(n_clientes=5000)
    salvar_dataset(df, DATA_PATH)

    # ------------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------------
    print("\n[2/5] Executando Análise Exploratória de Dados...")
    executar_eda(df)

    # ------------------------------------------------------------------
    # 3. Pré-processamento
    # ------------------------------------------------------------------
    print("\n[3/5] Pré-processando os dados...")
    X_train, X_test, y_train, y_test, feature_names = preprocessar(df)

    # ------------------------------------------------------------------
    # 4. Treinamento
    # ------------------------------------------------------------------
    print("\n[4/5] Treinando os modelos...")
    modelos = obter_modelos()
    modelos_treinados = treinar_modelos(modelos, X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Avaliação
    # ------------------------------------------------------------------
    print("\n[5/5] Avaliando os modelos e gerando relatórios...\n")
    df_resultados = avaliar_modelos(modelos_treinados, X_test, y_test)
    plotar_comparacao_metricas(df_resultados)
    plotar_matrizes_confusao(modelos_treinados, X_test, y_test)
    plotar_curvas_roc(modelos_treinados, X_test, y_test)

    melhor = selecionar_melhor_modelo(df_resultados)
    plotar_importancia_features(
        modelos_treinados[melhor], feature_names, nome_modelo=melhor
    )

    # ------------------------------------------------------------------
    # Conclusão
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  ✅ Pipeline concluído com sucesso!")
    print("  📊 Gráficos salvos em: reports/figures/")
    print("  📁 Dataset salvo em:   data/clientes.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()
