# Código sem imgs
"""
╔══════════════════════════════════════════════════════════════╗
║   PIPELINE COMPLETO DE MACHINE LEARNING                      ║
║   Diagnósticos para Biomedicina — Predição de Risco Clínico  ║
║   Modelos: Regressão Logística | Random Forest | KNN         ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

SEP1 = "═" * 62
SEP2 = "─" * 62

# ─────────────────────────────────────────────────────────────
# 1. LEITURA DO DATASET
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  ETAPA 1 — LEITURA DO DATASET")
print(SEP1)

df = pd.read_csv("pacientes_corrigido.csv")

print(f"\n  Arquivo carregado    : pacientes_corrigido.csv")
print(f"  Registros            : {len(df):,}")
print(f"  Features disponíveis : {list(df.columns)}")
print(f"\n  Distribuição do target (risco):")
dist = df["risco"].value_counts().sort_index()
labels = {0: "Baixo (0)", 1: "Médio (1)", 2: "Alto  (2)"}
for k, v in dist.items():
    bar = "█" * int(v / 20)
    print(f"    {labels[k]}: {v:>5} ({v/len(df)*100:.1f}%)  {bar}")

# ─────────────────────────────────────────────────────────────
# 2. SEPARAÇÃO DE FEATURES (X) E TARGET (y)
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  ETAPA 2 — SEPARAÇÃO FEATURES (X) E TARGET (y)")
print(SEP1)

FEATURES = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
TARGET   = "risco"

X = df[FEATURES]
y = df[TARGET]

print(f"\n  Features (X) selecionadas : {FEATURES}")
print(f"  Target  (y)               : '{TARGET}'  — classes {sorted(y.unique())}")
print(f"  Shape de X                : {X.shape}")

# ─────────────────────────────────────────────────────────────
# 3. DIVISÃO TREINO / TESTE
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  ETAPA 3 — DIVISÃO TREINO / TESTE  (80% / 20%)")
print(SEP1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # mantém proporção das classes em ambos os splits
)

print(f"\n  Treino  : {len(X_train):>5} registros  ({len(X_train)/len(df)*100:.0f}%)")
print(f"  Teste   : {len(X_test):>5} registros  ({len(X_test)/len(df)*100:.0f}%)")
print(f"\n  Distribuição no treino:")
for k, v in y_train.value_counts().sort_index().items():
    print(f"    Classe {k}: {v}")
print(f"\n  Distribuição no teste:")
for k, v in y_test.value_counts().sort_index().items():
    print(f"    Classe {k}: {v}")

# ─────────────────────────────────────────────────────────────
# 4. NORMALIZAÇÃO — StandardScaler
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  ETAPA 4 — NORMALIZAÇÃO (StandardScaler)")
print(SEP1)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit APENAS no treino
X_test_sc  = scaler.transform(X_test)        # aplica o mesmo scaler no teste

print(f"\n  Scaler ajustado no treino (fit) e aplicado no teste (transform).")
print(f"  Médias por feature (treino):")
for feat, mean, std in zip(FEATURES, scaler.mean_, scaler.scale_):
    print(f"    {feat:<20}: μ = {mean:>7.2f}  σ = {std:>6.2f}")

# ─────────────────────────────────────────────────────────────
# 5. DEFINIÇÃO DOS MODELOS
# ─────────────────────────────────────────────────────────────
modelos = {
    "Regressão Logística": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        metric="euclidean",
        weights="distance"
    ),
}

# ─────────────────────────────────────────────────────────────
# 6. TREINAMENTO + AVALIAÇÃO HOLD-OUT
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  ETAPA 5/6 — TREINAMENTO E AVALIAÇÃO DOS MODELOS")
print(SEP1)

resultados = {}

for nome, modelo in modelos.items():
    print(f"\n{'─'*62}")
    print(f"  🔧 {nome.upper()}")
    print(f"{'─'*62}")

    # Treina
    modelo.fit(X_train_sc, y_train)

    # Predição no conjunto de teste
    y_pred = modelo.predict(X_test_sc)

    # Métricas
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Métricas no conjunto de TESTE (hold-out):")
    print(f"    Acurácia  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")

    # Relatório por classe
    print(f"\n  Relatório detalhado por classe:")
    report = classification_report(
        y_test, y_pred,
        target_names=["Baixo (0)", "Médio (1)", "Alto (2)"],
        zero_division=0
    )
    for linha in report.split("\n"):
        print(f"    {linha}")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Matriz de Confusão:")
    print(f"    {'':12} Pred:Baixo  Pred:Médio  Pred:Alto")
    for i, label in enumerate(["Real:Baixo ", "Real:Médio ", "Real:Alto  "]):
        row = "  ".join(f"{cm[i][j]:>10}" for j in range(3))
        print(f"    {label}  {row}")

    resultados[nome] = {
        "modelo":    modelo,
        "acuracia":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "y_pred":    y_pred,
    }

# ─────────────────────────────────────────────────────────────
# 7. VALIDAÇÃO CRUZADA — K-FOLD (k=5, Stratified)
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  ETAPA 7 — VALIDAÇÃO CRUZADA  (StratifiedKFold, k=5)")
print(SEP1)

print(f"\n  Usando X_train normalizado com 5 folds estratificados.")
print(f"  Métrica: F1-Score weighted\n")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_resultados = {}

print(f"  {'Modelo':<24} {'F1 por Fold':<42} {'Média':>7}  {'Std':>6}")
print(f"  {'─'*24} {'─'*42} {'─'*7}  {'─'*6}")

for nome, info in resultados.items():
    scores = cross_val_score(
        info["modelo"], X_train_sc, y_train,
        cv=kfold, scoring="f1_weighted", n_jobs=-1
    )
    cv_resultados[nome] = scores
    folds_str = "  ".join(f"{s:.3f}" for s in scores)
    print(f"  {nome:<24} [{folds_str}]  {scores.mean():.4f}  {scores.std():.4f}")

# ─────────────────────────────────────────────────────────────
# 8. COMPARAÇÃO FINAL DOS MODELOS
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  ETAPA 8 — COMPARAÇÃO FINAL DOS MODELOS")
print(SEP1)

print(f"\n  {'Modelo':<24} {'Acurácia':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV-F1 μ':>9} {'CV-F1 σ':>8}")
print(f"  {'─'*24} {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")

ranking = []
for nome, info in resultados.items():
    cv_scores = cv_resultados[nome]
    print(
        f"  {nome:<24}"
        f"  {info['acuracia']:>8.4f}"
        f"  {info['precision']:>9.4f}"
        f"  {info['recall']:>7.4f}"
        f"  {info['f1']:>7.4f}"
        f"  {cv_scores.mean():>8.4f}"
        f"  {cv_scores.std():>7.4f}"
    )
    ranking.append((nome, info["f1"], cv_scores.mean(), cv_scores.std()))

# Ordena por CV-F1 médio (critério mais robusto)
ranking.sort(key=lambda x: x[2], reverse=True)
melhor = ranking[0]

print(f"\n{SEP1}")
print(f"  🏆 MODELO VENCEDOR: {melhor[0]}")
print(SEP1)
print(f"\n  Critério de seleção: maior F1-Score médio na validação cruzada")
print(f"  (mais robusto que hold-out por avaliar o modelo em 5 partições distintas)")
print(f"\n  Resultado:")
print(f"    CV F1-Score médio : {melhor[2]:.4f}")
print(f"    CV F1-Score std   : {melhor[3]:.4f}  ← menor = mais estável")
print(f"    F1 no hold-out    : {resultados[melhor[0]]['f1']:.4f}")

print(f"\n  Ranking completo (por CV F1-Score):")
for i, (nome, f1_ho, cv_mean, cv_std) in enumerate(ranking):
    medalha = ["🥇", "🥈", "🥉"][i]
    print(f"    {medalha} {i+1}º  {nome:<24}  CV-F1 = {cv_mean:.4f} ± {cv_std:.4f}")

# ─────────────────────────────────────────────────────────────
# IMPORTÂNCIA DAS FEATURES (Random Forest)
# ─────────────────────────────────────────────────────────────
print(f"\n{SEP1}")
print("  BÔNUS — IMPORTÂNCIA DAS FEATURES (Random Forest)")
print(SEP1)

rf_model = resultados["Random Forest"]["modelo"]
importancias = pd.Series(rf_model.feature_importances_, index=FEATURES)
importancias = importancias.sort_values(ascending=False)

print(f"\n  {'Feature':<22} {'Importância':>12}   Peso visual")
print(f"  {'─'*22} {'─'*12}   {'─'*28}")
for feat, imp in importancias.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<22} {imp:>12.4f}   {bar}")

print(f"\n{SEP1}")
print("  PIPELINE CONCLUÍDO COM SUCESSO ✅")
print(SEP1 + "\n")