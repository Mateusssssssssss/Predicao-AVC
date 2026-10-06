# Dados

O CSV usado pelo projeto fica em `data/raw/healthcare-dataset-stroke-data.csv` e é ignorado pelo Git conforme pedido. A fonte/contexto e o dicionário de atributos estão em `Contexto.txt`. `data/processed/` é reservado para saídas intermediárias; o pipeline atual faz split e preprocessamento em memória para evitar vazamento entre folds.
