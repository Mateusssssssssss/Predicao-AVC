"""EDA que registra qualidade, codifica categorias para visualizações e gera figuras."""
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.paths_config import DATA_PATH, FIGURES_DIR as FIGURES, REPORTS_DIR as REPORTS
from config.training_config import TARGET_COLUMN as TARGET, ID_COLUMN
from src.features.engineering import target_correlations, LOW_CORRELATION_THRESHOLD


def run_eda(path=DATA_PATH):
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])
    missing = df.isna().sum().sort_values(ascending=False)
    summary = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in missing.items()},
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).items()},
        "target_rate": float(df[TARGET].mean()),
    }
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    missing.rename("nulos").to_csv(REPORTS / "missing_values.csv", header=True)
    # One-hot preserva nominalidade na correlação exploratória em vez de impor ordem às categorias.
    features_for_corr = df.drop(columns=[ID_COLUMN], errors="ignore").copy()
    categorical = features_for_corr.select_dtypes(include=["object", "str"]).columns.tolist()
    corr_frame = pd.get_dummies(features_for_corr, columns=categorical, dtype=int)
    correlation = corr_frame.corr(numeric_only=True)
    correlation.to_csv(REPORTS / "feature_correlation.csv")
    correlation.to_csv(REPORTS / "encoded_correlation.csv")  # mantém o nome usado por versões anteriores

    # Correlação absoluta com stroke; categóricas representam o maior indicador one-hot.
    target_corr = target_correlations(df.drop(columns=[TARGET, ID_COLUMN], errors="ignore"), df[TARGET])
    target_corr.rename("abs_correlation_with_stroke").to_csv(REPORTS / "stroke_feature_correlations.csv")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET, hue=TARGET, legend=False, palette="Set2")
    plt.title("Distribuição da variável-alvo (stroke)"); plt.xlabel("AVC (0 = não, 1 = sim)"); plt.ylabel("Pacientes")
    plt.tight_layout(); plt.savefig(FIGURES / "target_imbalance.png", dpi=160); plt.close()

    plt.figure(figsize=(10, max(5, 0.42 * len(target_corr))))
    colors = ["#d97941" if value < LOW_CORRELATION_THRESHOLD else "#4c78a8"
              for value in target_corr.values]
    bars = plt.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
    plt.xlim(0, 1)
    plt.xticks([i / 10 for i in range(11)])
    plt.xlabel("Correlação absoluta com stroke (0 a 1)")
    plt.title("Correlação das variáveis com stroke")
    plt.axvline(LOW_CORRELATION_THRESHOLD, color="#b22222", linestyle="--",
                label=f"Referência visual ({LOW_CORRELATION_THRESHOLD:.2f})")
    plt.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "correlation_with_stroke.png", dpi=180); plt.close()

    numeric = [c for c in df.select_dtypes(include="number").columns if c not in [TARGET, ID_COLUMN]]
    melted = df[numeric].melt(var_name="variavel", value_name="valor")
    plt.figure(figsize=(11, 6)); sns.boxplot(data=melted, x="variavel", y="valor", color="#77aadd", showfliers=True)
    plt.title("Boxplots das variáveis numéricas (outliers mantidos para inspeção)"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(FIGURES / "boxplots_numeric.png", dpi=160); plt.close()

    plt.figure(figsize=(10, 5)); missing[missing.gt(0)].sort_values().plot.barh(color="#d97941")
    plt.title("Valores ausentes por coluna"); plt.xlabel("Quantidade")
    plt.tight_layout(); plt.savefig(FIGURES / "missing_values.png", dpi=160); plt.close()
    return summary


def main():
    print(json.dumps(run_eda(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
