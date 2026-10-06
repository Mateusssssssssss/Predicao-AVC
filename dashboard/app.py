"""Painel inicial para acompanhar execuções e métricas de treinamento no MLflow."""
from pathlib import Path
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{(ROOT / 'experiments' / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "stroke-risk-classification"

st.set_page_config(page_title="AVC | Model Monitor", page_icon="🩺", layout="wide")
st.title("Acompanhamento do modelo de AVC")
st.caption("Painel inicial de execuções de treinamento via MLflow. A telemetria de predições de produção ainda precisa ser integrada.")

@st.fragment(run_every=10)
def show_runs():
    try:
        client = MlflowClient(tracking_uri=TRACKING_URI)
        experiments = client.search_experiments(filter_string=f"name = '{EXPERIMENT_NAME}'")
        if not experiments:
            st.info("Nenhuma execução encontrada. Rode `python -m src.models.train` primeiro.")
            return
        runs = client.search_runs([experiments[0].experiment_id], order_by=["attributes.start_time DESC"], max_results=100)
        if not runs:
            st.info("O experimento ainda não possui execuções.")
            return
        rows = []
        for run in runs:
            rows.append({
                "início": pd.to_datetime(run.info.start_time, unit="ms", utc=True),
                "status": run.info.status,
                "modelo": run.data.params.get("model", ""),
                "amostragem": run.data.params.get("sampler", ""),
                "threshold": run.data.params.get("threshold", ""),
                "CV AUPRC": run.data.metrics.get("cv_average_precision_mean"),
                "Holdout AUPRC": run.data.metrics.get("test_average_precision"),
                "Holdout ROC AUC": run.data.metrics.get("test_roc_auc"),
                "Holdout F2": run.data.metrics.get("test_f2"),
                "recall": run.data.metrics.get("test_recall"),
            })
        frame = pd.DataFrame(rows)
        latest = frame.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Última execução", latest["status"])
        b.metric("Modelo", latest["modelo"])
        c.metric("CV AUPRC", f"{latest['CV AUPRC']:.3f}" if pd.notna(latest["CV AUPRC"]) else "—")
        d.metric("Holdout AUPRC", f"{latest['Holdout AUPRC']:.3f}" if pd.notna(latest["Holdout AUPRC"]) else "—")
        st.subheader("Execuções recentes")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption("Atualiza a cada 10 s. Valores holdout servem para comparação exploratória; acompanhe mudança de dados, calibração e desempenho externo antes de qualquer uso real.")
    except Exception as exc:
        st.error(f"Não foi possível ler o MLflow local: {exc}")

show_runs()
