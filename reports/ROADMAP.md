# Roadmap de evolução

- [x] Auditoria exploratória: nulos, duplicados, distribuição do alvo e boxplots.
- [x] Matriz de correlação com indicadores one-hot para categorias nominais.
- [x] Pipeline de pré-processamento e amostragem encapsulado na validação cruzada.
- [x] Comparação inicial de Random Forest, XGBoost e LightGBM com SMOTENC e SMOTENC + undersampling.
- [x] Seleção por F2 média em validação cruzada e threshold F2 em predições out-of-fold para priorizar recall de stroke=1.
- [x] Persistência do pipeline e threshold com joblib e API de inferência.
- [x] Tracking local de parâmetros e métricas por execução com MLflow.
- [ ] Reavaliar calibração, análise de subgrupos e validação temporal/externa com dados representativos.
- [ ] Integrar API a um serviço de tracking de predições e painel (Prometheus/Grafana ou dashboard dedicado).
- [ ] Configurar CI, testes de contrato, autenticação, limites de taxa, monitoramento de drift e processo de retreino.

O conjunto atual é educacional e pequeno; as etapas futuras são necessárias antes de qualquer uso clínico.
