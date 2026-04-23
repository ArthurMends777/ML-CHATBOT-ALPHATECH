"""
╔══════════════════════════════════════════════════════════════╗
║   VISUALIZAÇÕES — PIPELINE DE MACHINE LEARNING               ║
║   Diagnósticos para Biomedicina                              ║
║   • Comparação de Acurácia e F1 dos modelos                  ║
║   • Matriz de Confusão — Random Forest (melhor modelo)       ║
║   • Curva ROC Multiclasse (OvR) — Random Forest              ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# ─────────────────────────────────────────────────────────────
# PALETA E ESTILO GLOBAL
# ─────────────────────────────────────────────────────────────
PALETTE = {
    "bg":        "#0F1117",
    "panel":     "#1A1D27",
    "border":    "#2A2D3A",
    "text":      "#E8EAF0",
    "subtext":   "#8A8FA8",
    "accent":    "#00D4AA",
    "models": {
        "Regressão Logística": "#4C9BE8",
        "Random Forest":       "#00D4AA",
        "KNN":                 "#F5A623",
    },
    "classes": {
        0: "#4C9BE8",
        1: "#F5A623",
        2: "#E85C6B",
    },
    "roc": ["#4C9BE8", "#F5A623", "#E85C6B"],
}

CLASSES     = [0, 1, 2]
CLASS_NAMES = ["Baixo (0)", "Médio (1)", "Alto (2)"]

def apply_base_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["subtext"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=12,
                     fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=PALETTE["subtext"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["subtext"], fontsize=9)
    ax.tick_params(axis="x", colors=PALETTE["subtext"])
    ax.tick_params(axis="y", colors=PALETTE["subtext"])

# ─────────────────────────────────────────────────────────────
# 1. PIPELINE DE DADOS (reutilizado do script anterior)
# ─────────────────────────────────────────────────────────────
print("  Carregando dataset e treinando modelos...")

df = pd.read_csv("pacientes_corrigido.csv")
FEATURES = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
X = df[FEATURES]
y = df["risco"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"),
    "KNN":                 KNeighborsClassifier(n_neighbors=7, metric="euclidean", weights="distance"),
}

kfold    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train_sc, y_train)
    y_pred = modelo.predict(X_test_sc)
    cv_scores = cross_val_score(modelo, X_train_sc, y_train,
                                cv=kfold, scoring="f1_weighted", n_jobs=-1)
    resultados[nome] = {
        "modelo":    modelo,
        "acuracia":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cv_mean":   cv_scores.mean(),
        "cv_std":    cv_scores.std(),
        "y_pred":    y_pred,
    }

MELHOR = "Random Forest"
print(f"  Modelos treinados. Melhor: {MELHOR}\n")

# ─────────────────────────────────────────────────────────────
# FIGURA 1 — COMPARAÇÃO DE MODELOS
# ─────────────────────────────────────────────────────────────
print("  Gerando Fig 1 — Comparação de modelos...")

nomes      = list(resultados.keys())
cores      = [PALETTE["models"][n] for n in nomes]
metricas   = ["acuracia", "precision", "recall", "f1"]
labels_met = ["Acurácia", "Precision", "Recall", "F1-Score"]

fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
fig1.patch.set_facecolor(PALETTE["bg"])
fig1.suptitle("Comparação de Desempenho dos Modelos",
              color=PALETTE["text"], fontsize=15, fontweight="bold", y=1.01)

# ── Subplot A: barras agrupadas (4 métricas × 3 modelos) ──────
ax = axes[0]
apply_base_style(ax, title="Métricas Hold-out por Modelo",
                 xlabel="", ylabel="Score")

x    = np.arange(len(labels_met))
w    = 0.22
offsets = [-w, 0, w]

for i, (nome, off) in enumerate(zip(nomes, offsets)):
    vals = [resultados[nome][m] for m in metricas]
    bars = ax.bar(x + off, vals, width=w, color=cores[i],
                  label=nome, alpha=0.88, zorder=3,
                  edgecolor=PALETTE["bg"], linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.2f}", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels_met, color=PALETTE["subtext"], fontsize=10)
ax.set_ylim(0.6, 1.02)
ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="lower right", facecolor=PALETTE["panel"],
          edgecolor=PALETTE["border"], labelcolor=PALETTE["text"],
          fontsize=8.5)

# Destaque do melhor modelo
ax.axhline(resultados[MELHOR]["f1"], color=PALETTE["accent"],
           linestyle=":", linewidth=1.2, alpha=0.6)
ax.text(3.55, resultados[MELHOR]["f1"] + 0.005,
        f"RF F1={resultados[MELHOR]['f1']:.3f}",
        color=PALETTE["accent"], fontsize=7.5, ha="right")

# ── Subplot B: CV F1 com barras de erro ───────────────────────
ax2 = axes[1]
apply_base_style(ax2,
                 title="Validação Cruzada — F1-Score Médio (k=5)",
                 xlabel="Modelo", ylabel="CV F1-Score (weighted)")

cv_means = [resultados[n]["cv_mean"] for n in nomes]
cv_stds  = [resultados[n]["cv_std"]  for n in nomes]

bars2 = ax2.bar(nomes, cv_means, color=cores, alpha=0.88,
                edgecolor=PALETTE["bg"], linewidth=0.8, zorder=3)
ax2.errorbar(nomes, cv_means, yerr=cv_stds,
             fmt="none", ecolor=PALETTE["text"], elinewidth=2,
             capsize=8, capthick=2, zorder=4)

for bar, v, s in zip(bars2, cv_means, cv_stds):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             v + s + 0.012,
             f"{v:.4f}\n±{s:.4f}",
             ha="center", va="bottom",
             color=PALETTE["text"], fontsize=8.5, fontweight="bold")

ax2.set_ylim(0.65, 1.02)
ax2.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", labelsize=9)

# Halo no vencedor
idx_best = nomes.index(MELHOR)
bars2[idx_best].set_edgecolor(PALETTE["accent"])
bars2[idx_best].set_linewidth(2.5)
ax2.text(idx_best, 0.67, "🏆", ha="center", fontsize=14)

plt.tight_layout()
fig1.savefig("fig1_comparacao_modelos.png", dpi=150,
             bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(fig1)
print("  ✔ fig1_comparacao_modelos.png")

# ─────────────────────────────────────────────────────────────
# FIGURA 2 — MATRIZ DE CONFUSÃO (Random Forest)
# ─────────────────────────────────────────────────────────────
print("  Gerando Fig 2 — Matriz de Confusão...")

cm = confusion_matrix(y_test, resultados[MELHOR]["y_pred"])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalizada por linha

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor(PALETTE["bg"])
fig2.suptitle(f"Matriz de Confusão — {MELHOR}",
              color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01)

cmap_teal = LinearSegmentedColormap.from_list(
    "teal_dark", [PALETTE["panel"], PALETTE["accent"]], N=256
)

for idx, (mat, titulo, fmt) in enumerate([
    (cm,      "Valores Absolutos",   "d"),
    (cm_norm, "Normalizada (% linha)", ".2%"),
]):
    ax = axes2[idx]
    apply_base_style(ax, title=titulo)

    im = ax.imshow(mat, cmap=cmap_teal,
                   vmin=0, vmax=(1 if idx == 1 else cm.max()))

    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            txt = f"{val:{fmt}}"
            cor = PALETTE["bg"] if (val > (0.5 if idx == 1 else cm.max() * 0.5)) else PALETTE["text"]
            ax.text(j, i, txt, ha="center", va="center",
                    color=cor, fontsize=12, fontweight="bold")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, color=PALETTE["subtext"], fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, color=PALETTE["subtext"], fontsize=9)
    ax.set_xlabel("Predito", color=PALETTE["subtext"], fontsize=10)
    ax.set_ylabel("Real",    color=PALETTE["subtext"], fontsize=10)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=PALETTE["subtext"], labelsize=8)

    # Destaque diagonal (acertos)
    for k in range(3):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                              fill=False, edgecolor=PALETTE["accent"],
                              linewidth=2.2, zorder=5)
        ax.add_patch(rect)

# Métricas resumidas abaixo
acc_rf = resultados[MELHOR]["acuracia"]
f1_rf  = resultados[MELHOR]["f1"]
fig2.text(0.5, -0.04,
          f"Acurácia: {acc_rf:.4f}   |   F1-Score (weighted): {f1_rf:.4f}   |   Registros no teste: {len(y_test)}",
          ha="center", color=PALETTE["subtext"], fontsize=9)

plt.tight_layout()
fig2.savefig("fig2_matriz_confusao.png", dpi=150,
             bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(fig2)
print("  ✔ fig2_matriz_confusao.png")

# ─────────────────────────────────────────────────────────────
# FIGURA 3 — CURVA ROC MULTICLASSE (OvR) — Random Forest
# ─────────────────────────────────────────────────────────────
print("  Gerando Fig 3 — Curva ROC...")

# Binariza y_test para OvR
y_test_bin   = label_binarize(y_test, classes=CLASSES)
y_prob_rf    = resultados[MELHOR]["modelo"].predict_proba(X_test_sc)

fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
for i, cls in enumerate(CLASSES):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_prob_rf[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

# Micro-average
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_prob_rf.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

fig3, ax3 = plt.subplots(figsize=(8, 6))
fig3.patch.set_facecolor(PALETTE["bg"])
apply_base_style(ax3,
                 title=f"Curva ROC Multiclasse (OvR) — {MELHOR}",
                 xlabel="Taxa de Falsos Positivos (FPR)",
                 ylabel="Taxa de Verdadeiros Positivos (TPR)")

# Linha de referência (classificador aleatório)
ax3.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
         color=PALETTE["subtext"], label="Aleatório (AUC = 0.50)", zorder=1)

# Curva por classe
for i, cls in enumerate(CLASSES):
    ax3.plot(fpr_dict[i], tpr_dict[i],
             color=PALETTE["roc"][i], linewidth=2.2, zorder=3,
             label=f"Classe {CLASS_NAMES[i]}  (AUC = {roc_auc_dict[i]:.3f})")
    # Área sombreada
    ax3.fill_between(fpr_dict[i], tpr_dict[i], alpha=0.06,
                     color=PALETTE["roc"][i], zorder=2)

# Micro-average
ax3.plot(fpr_micro, tpr_micro,
         color=PALETTE["accent"], linewidth=2.6, linestyle="-.",
         label=f"Micro-average (AUC = {roc_auc_micro:.3f})", zorder=4)

# Ponto ótimo — Youden J (por classe)
for i, cls in enumerate(CLASSES):
    j_scores = tpr_dict[i] - fpr_dict[i]
    idx_opt  = np.argmax(j_scores)
    ax3.scatter(fpr_dict[i][idx_opt], tpr_dict[i][idx_opt],
                s=70, color=PALETTE["roc"][i], zorder=5,
                edgecolors=PALETTE["text"], linewidths=1.2)

ax3.set_xlim(-0.01, 1.01)
ax3.set_ylim(-0.01, 1.05)
ax3.xaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.4)
ax3.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.4)
ax3.set_axisbelow(True)

legend = ax3.legend(loc="lower right", facecolor=PALETTE["panel"],
                    edgecolor=PALETTE["border"], labelcolor=PALETTE["text"],
                    fontsize=9.5, framealpha=0.9)

# Anotação AUC médio
auc_medio = np.mean(list(roc_auc_dict.values()))
ax3.text(0.98, 0.10,
         f"AUC médio (por classe): {auc_medio:.3f}",
         ha="right", va="bottom", color=PALETTE["accent"],
         fontsize=9, fontweight="bold",
         transform=ax3.transAxes)

# Pontos ótimos — legenda
ax3.scatter([], [], s=70, color=PALETTE["subtext"],
            edgecolors=PALETTE["text"], linewidths=1.2,
            label="Ponto ótimo (Youden J)")

plt.tight_layout()
fig3.savefig("fig3_curva_roc.png", dpi=150,
             bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close(fig3)
print("  ✔ fig3_curva_roc.png")

print("\n  ✅ Todas as visualizações geradas com sucesso!")
print("     fig1_comparacao_modelos.png")
print("     fig2_matriz_confusao.png")
print("     fig3_curva_roc.png\n")