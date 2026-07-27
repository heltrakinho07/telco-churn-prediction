"""
eda.py — Análise Exploratória de Dados (EDA).

Antes de treinar qualquer modelo, temos de PERCEBER os dados.
Este script gera gráficos que respondem a: quem é que faz churn e porquê?
As figuras vão para reports/figures/ — prontas a usar nos teus posts.
"""
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR, TARGET
from data_prep import get_clean_data

# Estilo visual limpo e profissional
sns.set_theme(style="whitegrid", palette="Set2")


def plot_churn_balance(df):
    """Mostra quantos clientes ficam vs. saem (o desequilíbrio de classes)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET].map({0: "Ficou", 1: "Saiu (churn)"}).value_counts()
    counts.plot(kind="bar", ax=ax, color=["#4c9f70", "#d1495b"])
    ax.set_title("Distribuição de Churn")
    ax.set_ylabel("Nº de clientes")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_churn_balance.png", dpi=120)
    plt.close(fig)


def plot_churn_by_contract(df):
    """Churn por tipo de contrato — costuma ser o fator mais forte."""
    fig, ax = plt.subplots(figsize=(7, 4))
    rate = df.groupby("Contract")[TARGET].mean().sort_values(ascending=False)
    rate.plot(kind="bar", ax=ax, color="#d1495b")
    ax.set_title("Taxa de churn por tipo de contrato")
    ax.set_ylabel("Taxa de churn")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    for i, v in enumerate(rate):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_churn_by_contract.png", dpi=120)
    plt.close(fig)


def plot_tenure_distribution(df):
    """Distribuição do tempo de casa (tenure) por churn."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name, color in [(0, "Ficou", "#4c9f70"), (1, "Saiu", "#d1495b")]:
        sns.histplot(
            df[df[TARGET] == label]["tenure"],
            bins=30, label=name, color=color, alpha=0.6, ax=ax,
        )
    ax.set_title("Tempo de casa (meses) por churn")
    ax.set_xlabel("Tenure (meses)")
    ax.set_ylabel("Nº de clientes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_tenure_distribution.png", dpi=120)
    plt.close(fig)


def main():
    df = get_clean_data(save=False)
    print("A gerar figuras...")
    plot_churn_balance(df)
    plot_churn_by_contract(df)
    plot_tenure_distribution(df)
    print(f"3 figuras guardadas em: {FIGURES_DIR}")

    # Um resumo textual rápido para a consola
    print("\n--- Descobertas rápidas ---")
    by_contract = df.groupby("Contract")[TARGET].mean()
    print("Churn por contrato:")
    for k, v in by_contract.items():
        print(f"  {k:20s}: {v:.1%}")


if __name__ == "__main__":
    main()
