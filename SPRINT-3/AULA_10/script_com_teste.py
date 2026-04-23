"""
╔══════════════════════════════════════════════════════════════════════╗
║   PROJETO COMPLETO DE MACHINE LEARNING — DIAGNÓSTICOS BIOMEDICINA   ║
║   Disciplina: Introdução ao Machine Learning Aplicado à Saúde        ║
║                                                                      ║
║   ETAPAS COBERTAS:                                                   ║
║     1. Geração do dataset sintético (2000 pacientes)                 ║
║     2. Limpeza e validação dos dados                                 ║
║     3. Pipeline de ML (treino, avaliação, validação cruzada)         ║
║     4. Visualizações com Matplotlib                                  ║
║     5. Predição de novo paciente via input interativo                ║
║                                                                      ║
║   Bibliotecas: pandas | numpy | scikit-learn | matplotlib           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────
# IMPORTAÇÕES
# Todas as bibliotecas utilizadas no projeto estão centralizadas aqui.
# Isso facilita a leitura e a identificação de dependências.
# ──────────────────────────────────────────────────────────────────────
import pandas as pd                   # manipulação de dados tabulares
import numpy as np                    # operações numéricas e vetoriais
import warnings                       # supressão de alertas desnecessários
import matplotlib                     # backend gráfico
import matplotlib.pyplot as plt       # plotagem de gráficos
import matplotlib.patches as mpatches # elementos gráficos adicionais
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import (
    train_test_split,   # divisão treino/teste
    cross_val_score,    # validação cruzada
    StratifiedKFold     # k-fold com estratificação de classes
)
from sklearn.preprocessing import (
    StandardScaler,     # normalização: média 0, desvio padrão 1
    label_binarize      # binarização para curva ROC multiclasse
)
from sklearn.linear_model import LogisticRegression   # modelo linear
from sklearn.ensemble import RandomForestClassifier   # ensemble de árvores
from sklearn.neighbors import KNeighborsClassifier    # modelo baseado em distância
from sklearn.metrics import (
    accuracy_score,     # proporção de acertos totais
    precision_score,    # dos que previu positivo, quantos eram de fato?
    recall_score,       # dos positivos reais, quantos o modelo pegou?
    f1_score,           # média harmônica entre precision e recall
    classification_report,  # relatório completo por classe
    confusion_matrix,       # tabela de acertos e erros por classe
    roc_curve,              # pontos da curva ROC
    auc                     # área sob a curva
)

warnings.filterwarnings("ignore")   # evita mensagens de convergência
matplotlib.use("Agg")               # backend sem janela (salva em arquivo)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO VISUAL GLOBAL
# Centraliza cores e estilos para manter consistência em todos os gráficos.
# ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":     "#0F1117",   # fundo escuro principal
    "panel":  "#1A1D27",   # fundo dos painéis internos
    "border": "#2A2D3A",   # bordas e grades
    "text":   "#E8EAF0",   # texto principal
    "sub":    "#8A8FA8",   # texto secundário
    "accent": "#00D4AA",   # verde-ciano de destaque
    "models": {
        "Regressão Logística": "#4C9BE8",  # azul
        "Random Forest":       "#00D4AA",  # verde
        "KNN":                 "#F5A623",  # laranja
    },
    "roc":  ["#4C9BE8", "#F5A623", "#E85C6B"],
    "risk": {
        0: ("#4C9BE8", "💙 BAIXO RISCO"),
        1: ("#F5A623", "🟡 RISCO MÉDIO"),
        2: ("#E85C6B", "🔴 RISCO ALTO"),
    }
}

# ──────────────────────────────────────────────────────────────────────
# CONSTANTES DO PROJETO
# ──────────────────────────────────────────────────────────────────────
N_REGISTROS  = 2000
FEATURES     = ["idade", "glicose", "pressao_arterial", "imc", "colesterol"]
TARGET       = "risco"
CLASSES      = [0, 1, 2]
CLASS_NAMES  = ["Baixo (0)", "Médio (1)", "Alto (2)"]
MELHOR_MODELO_NOME = "Random Forest"

# Separadores visuais para o console
SEP1 = "═" * 68
SEP2 = "─" * 68


# ══════════════════════════════════════════════════════════════════════
# BLOCO 1 — GERAÇÃO DO DATASET SINTÉTICO
# Criamos 2000 pacientes fictícios com variáveis biomédicas realistas.
# A variável-alvo "risco" é calculada por um sistema de pontuação
# baseado em limiares clínicos reais.
# ══════════════════════════════════════════════════════════════════════
def gerar_dataset(n: int = N_REGISTROS, seed: int = 42) -> pd.DataFrame:
    """
    Gera um DataFrame com dados biomédicos sintéticos e variável alvo 'risco'.

    Parâmetros:
        n    : número de registros a gerar
        seed : semente para reprodutibilidade

    Retorna:
        DataFrame com colunas: nome, idade, glicose, pressao_arterial,
                                imc, colesterol, risco
    """
    rng = np.random.default_rng(seed)

    # ── Nomes fictícios (sem sobrenomes) ────────────────────────────
    nomes = [
        "Carlos","João","Pedro","Lucas","Marcos","Rafael","Bruno","Diego",
        "Felipe","Gabriel","Henrique","Igor","Jorge","Leonardo","Mateus",
        "Roberto","Samuel","Thiago","Vinícius","André","Breno","César",
        "Davi","Edson","Fábio","Guilherme","Hugo","Júlio","Leandro","Mário",
        "Ana","Beatriz","Carla","Daniela","Elisa","Fernanda","Gabriela",
        "Helena","Isabela","Juliana","Karen","Larissa","Márcia","Natália",
        "Olivia","Patrícia","Renata","Sandra","Tatiana","Valéria","Alice",
        "Bruna","Cláudia","Débora","Érica","Flávia","Giovana","Hanna",
        "Íris","Jéssica","Luana","Mônica","Nara","Yara","Zilda"
    ]
    nome_col = rng.choice(nomes, size=n)

    # ── Idade: distribuição normal centrada em 50 anos, range 18–99 ─
    idade = np.clip(
        rng.normal(loc=50, scale=17, size=n).astype(int), 18, 99
    )

    # ── Glicose (mg/dL): normal ~105, com leve influência da idade ──
    # Referência: normal < 100, pré-diabético 100–125, diabético ≥ 126
    glicose = np.clip(
        rng.normal(105, 28, n) + (idade - 50) * 0.3, 60, 400
    ).round(1)

    # ── Pressão arterial sistólica (mmHg): normal ~120 ───────────────
    # Referência: normal <120, elevada 120–129, hipertensão ≥130
    pressao = np.clip(
        rng.normal(120, 18, n) + (idade - 50) * 0.35, 80, 220
    ).round(1)

    # ── IMC (kg/m²): normal ~26.5 ────────────────────────────────────
    # Referência: normal 18.5–24.9, sobrepeso 25–29.9, obeso ≥30
    imc = np.clip(rng.normal(26.5, 5.2, n), 14.0, 55.0).round(1)

    # ── Colesterol total (mg/dL): desejável <200, alto ≥240 ──────────
    colesterol = np.clip(
        rng.normal(210, 42, n) + (idade - 50) * 0.4, 100, 400
    ).round(1)

    # ── Variável alvo: sistema de pontuação clínica ───────────────────
    # Cada condição de risco vale pontos; a soma define a classe final.
    pontos = np.zeros(n, dtype=int)

    pontos += (idade >= 45).astype(int)        # +1 se 45+
    pontos += (idade >= 65).astype(int)        # +1 adicional se 65+
    pontos += (glicose >= 100).astype(int)     # pré-diabético
    pontos += (glicose >= 126).astype(int)     # diabético (+1)
    pontos += (pressao >= 130).astype(int)     # hipertensão estágio 1
    pontos += (pressao >= 160).astype(int)     # hipertensão estágio 2
    pontos += (imc >= 25).astype(int)          # sobrepeso
    pontos += (imc >= 30).astype(int)          # obesidade (+1)
    pontos += (colesterol >= 200).astype(int)  # limítrofe
    pontos += (colesterol >= 240).astype(int)  # alto (+1)

    # Classificação: 0–2 pts = baixo | 3–5 = médio | 6+ = alto
    risco = np.where(pontos <= 2, 0, np.where(pontos <= 5, 1, 2))

    return pd.DataFrame({
        "nome":             nome_col,
        "idade":            idade,
        "glicose":          glicose,
        "pressao_arterial": pressao,
        "imc":              imc,
        "colesterol":       colesterol,
        "risco":            risco
    })


# ══════════════════════════════════════════════════════════════════════
# BLOCO 2 — LIMPEZA E VALIDAÇÃO DOS DADOS
# Detecta e corrige valores nulos ou zero em colunas onde isso é
# biologicamente impossível, substituindo pela mediana da coluna.
# ══════════════════════════════════════════════════════════════════════
def limpar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta e corrige valores inválidos (NaN ou zero) em colunas críticas.
    A mediana é calculada ANTES da substituição para preservar a distribuição.
    """
    # Colunas onde zero ou nulo é clinicamente impossível
    colunas_criticas = ["glicose", "pressao_arterial", "imc"]

    df = df.copy()  # nunca modifica o DataFrame original (boas práticas)

    for col in colunas_criticas:
        # Passo 1: converte zeros para NaN (zeros são inválidos nessas colunas)
        df[col] = df[col].replace(0, np.nan)

        # Passo 2: calcula mediana ignorando NaN
        mediana = df[col].median()

        # Passo 3: substitui NaN pela mediana
        n_corrigidos = df[col].isnull().sum()
        df[col] = df[col].fillna(mediana)

        if n_corrigidos > 0:
            print(f"    [{col}] {n_corrigidos} valor(es) corrigido(s) → mediana = {mediana:.2f}")

    return df


# ══════════════════════════════════════════════════════════════════════
# BLOCO 3 — UTILITÁRIO VISUAL
# Função auxiliar para aplicar o estilo escuro em todos os eixos.
# ══════════════════════════════════════════════════════════════════════
def estilizar_eixo(ax, titulo="", xlabel="", ylabel=""):
    """Aplica paleta escura e configurações visuais ao eixo matplotlib."""
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["sub"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    if titulo:
        ax.set_title(titulo, color=PALETTE["text"], fontsize=12,
                     fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=PALETTE["sub"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["sub"], fontsize=9)


# ══════════════════════════════════════════════════════════════════════
# BLOCO 4 — VISUALIZAÇÃO 1: COMPARAÇÃO DE MODELOS
# Dois painéis: métricas hold-out (barras agrupadas) e CV F1 com
# barras de erro mostrando estabilidade de cada modelo.
# ══════════════════════════════════════════════════════════════════════
def plot_comparacao_modelos(resultados: dict):
    nomes    = list(resultados.keys())
    cores    = [PALETTE["models"][n] for n in nomes]
    metricas = ["acuracia", "precision", "recall", "f1"]
    labels   = ["Acurácia", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Comparação de Desempenho dos Modelos",
                 color=PALETTE["text"], fontsize=15, fontweight="bold", y=1.01)

    # ── Painel A: barras agrupadas por métrica ───────────────────────
    ax = axes[0]
    estilizar_eixo(ax, "Métricas Hold-out por Modelo", ylabel="Score")

    x = np.arange(len(labels))
    w = 0.22
    offsets = [-w, 0, w]

    for i, (nome, off) in enumerate(zip(nomes, offsets)):
        vals = [resultados[nome][m] for m in metricas]
        bars = ax.bar(x + off, vals, width=w, color=cores[i], label=nome,
                      alpha=0.88, zorder=3, edgecolor=PALETTE["bg"], linewidth=0.6)
        # Rótulo numérico acima de cada barra
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008, f"{v:.2f}",
                    ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=PALETTE["sub"], fontsize=10)
    ax.set_ylim(0.60, 1.04)
    ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"], fontsize=8.5)

    # Linha de referência no F1 do melhor modelo
    f1_melhor = resultados[MELHOR_MODELO_NOME]["f1"]
    ax.axhline(f1_melhor, color=PALETTE["accent"], linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(3.55, f1_melhor + 0.005, f"RF F1={f1_melhor:.3f}",
            color=PALETTE["accent"], fontsize=7.5, ha="right")

    # ── Painel B: CV F1 com barra de erro ───────────────────────────
    ax2 = axes[1]
    estilizar_eixo(ax2, "Validação Cruzada — F1-Score Médio (k=5)",
                   xlabel="Modelo", ylabel="CV F1-Score (weighted)")

    cv_means = [resultados[n]["cv_mean"] for n in nomes]
    cv_stds  = [resultados[n]["cv_std"]  for n in nomes]

    bars2 = ax2.bar(nomes, cv_means, color=cores, alpha=0.88,
                    edgecolor=PALETTE["bg"], linewidth=0.8, zorder=3)
    # Barras de erro: desvio padrão entre os 5 folds
    ax2.errorbar(nomes, cv_means, yerr=cv_stds,
                 fmt="none", ecolor=PALETTE["text"], elinewidth=2,
                 capsize=8, capthick=2, zorder=4)

    for bar, v, s in zip(bars2, cv_means, cv_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 v + s + 0.012, f"{v:.4f}\n±{s:.4f}",
                 ha="center", va="bottom",
                 color=PALETTE["text"], fontsize=8.5, fontweight="bold")

    ax2.set_ylim(0.65, 1.04)
    ax2.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="x", labelsize=9)

    # Destaca vencedor com borda especial
    idx_best = nomes.index(MELHOR_MODELO_NOME)
    bars2[idx_best].set_edgecolor(PALETTE["accent"])
    bars2[idx_best].set_linewidth(2.5)
    ax2.text(idx_best, 0.67, "🏆", ha="center", fontsize=14)

    plt.tight_layout()
    fig.savefig("fig1_comparacao_modelos.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("    ✔ fig1_comparacao_modelos.png salvo")


# ══════════════════════════════════════════════════════════════════════
# BLOCO 5 — VISUALIZAÇÃO 2: MATRIZ DE CONFUSÃO
# Exibe lado a lado: valores absolutos e versão normalizada por linha.
# A diagonal principal representa os acertos do modelo.
# ══════════════════════════════════════════════════════════════════════
def plot_matriz_confusao(y_test, y_pred, nome_modelo: str):
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(f"Matriz de Confusão — {nome_modelo}",
                 color=PALETTE["text"], fontsize=14, fontweight="bold", y=1.01)

    # Colormap personalizado: do fundo escuro ao verde-ciano
    cmap = LinearSegmentedColormap.from_list(
        "teal", [PALETTE["panel"], PALETTE["accent"]], N=256
    )

    for idx, (mat, titulo, fmt) in enumerate([
        (cm,      "Valores Absolutos",    "d"),
        (cm_norm, "Normalizada (% linha)", ".2%"),
    ]):
        ax = axes[idx]
        estilizar_eixo(ax, titulo)

        im = ax.imshow(mat, cmap=cmap,
                       vmin=0, vmax=(1 if idx == 1 else cm.max()))

        # Rótulo em cada célula
        for i in range(3):
            for j in range(3):
                v   = mat[i, j]
                txt = f"{v:{fmt}}"
                # Texto preto em células claras, branco em escuras
                cor = PALETTE["bg"] if v > (0.5 if idx == 1 else cm.max() * 0.5) else PALETTE["text"]
                ax.text(j, i, txt, ha="center", va="center",
                        color=cor, fontsize=12, fontweight="bold")

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, color=PALETTE["sub"], fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, color=PALETTE["sub"], fontsize=9)
        ax.set_xlabel("Predito", color=PALETTE["sub"], fontsize=10)
        ax.set_ylabel("Real",    color=PALETTE["sub"], fontsize=10)

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=PALETTE["sub"], labelsize=8)

        # Borda verde na diagonal (acertos)
        for k in range(3):
            rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                  fill=False, edgecolor=PALETTE["accent"],
                                  linewidth=2.2, zorder=5)
            ax.add_patch(rect)

    fig.text(0.5, -0.04,
             f"Acurácia: {accuracy_score(y_test, y_pred):.4f}  |  "
             f"F1 (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}  |  "
             f"Registros no teste: {len(y_test)}",
             ha="center", color=PALETTE["sub"], fontsize=9)

    plt.tight_layout()
    fig.savefig("fig2_matriz_confusao.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("    ✔ fig2_matriz_confusao.png salvo")


# ══════════════════════════════════════════════════════════════════════
# BLOCO 6 — VISUALIZAÇÃO 3: CURVA ROC MULTICLASSE (OvR)
# Estratégia One-vs-Rest: cada classe é avaliada como binária.
# AUC próximo de 1.0 indica excelente separação das classes.
# ══════════════════════════════════════════════════════════════════════
def plot_curva_roc(y_test, y_prob, nome_modelo: str):
    # Binariza y_test: cada coluna representa uma classe (1 = pertence, 0 = não)
    y_bin = label_binarize(y_test, classes=CLASSES)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    estilizar_eixo(ax,
                   f"Curva ROC Multiclasse (OvR) — {nome_modelo}",
                   "Taxa de Falsos Positivos (FPR)",
                   "Taxa de Verdadeiros Positivos (TPR)")

    # Linha de referência: classificador aleatório (AUC = 0.50)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
            color=PALETTE["sub"], label="Aleatório (AUC = 0.50)", zorder=1)

    # Curva ROC para cada classe
    auc_por_classe = []
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_por_classe.append(roc_auc)

        ax.plot(fpr, tpr, color=PALETTE["roc"][i], linewidth=2.2, zorder=3,
                label=f"Classe {CLASS_NAMES[i]}  (AUC = {roc_auc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.06, color=PALETTE["roc"][i], zorder=2)

        # Ponto ótimo de Youden: maximiza TPR − FPR
        j_idx = np.argmax(tpr - fpr)
        ax.scatter(fpr[j_idx], tpr[j_idx], s=70,
                   color=PALETTE["roc"][i], zorder=5,
                   edgecolors=PALETTE["text"], linewidths=1.2)

    # Curva micro-average: agrega todas as classes de uma vez
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro,
            color=PALETTE["accent"], linewidth=2.6, linestyle="-.",
            label=f"Micro-average (AUC = {auc_micro:.3f})", zorder=4)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.xaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text"],
              fontsize=9.5, framealpha=0.9)
    ax.text(0.98, 0.10,
            f"AUC médio (por classe): {np.mean(auc_por_classe):.3f}",
            ha="right", va="bottom",
            color=PALETTE["accent"], fontsize=9, fontweight="bold",
            transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig("fig3_curva_roc.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("    ✔ fig3_curva_roc.png salvo")


# ══════════════════════════════════════════════════════════════════════
# BLOCO 7 — VISUALIZAÇÃO 4: PAINEL DE PREDIÇÃO DO NOVO PACIENTE
# Gráfico específico para exibir o resultado da predição de forma
# clara e didática: probabilidades em barras + classificação final.
# ══════════════════════════════════════════════════════════════════════
def plot_predicao_paciente(paciente: dict, probs: np.ndarray, classe_pred: int, acuracia_modelo: float = None):
    """
    Gera um gráfico visual com o resultado da predição do paciente.

    Parâmetros:
        paciente   : dicionário com os dados clínicos do paciente
        probs      : array [prob_baixo, prob_medio, prob_alto]
        classe_pred: classe prevista (0, 1 ou 2)
    """
    cor_pred, label_pred = PALETTE["risk"][classe_pred]

    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Painel esquerdo: dados do paciente ───────────────────────────
    ax_info = fig.add_subplot(gs[0])
    ax_info.set_facecolor(PALETTE["panel"])
    ax_info.axis("off")

    linhas = [
        ("DADOS DO PACIENTE", None, PALETTE["text"], 14, "bold"),
        ("", None, PALETTE["text"], 1, "normal"),
        (f"Nome",             paciente["nome"],           PALETTE["sub"],  10, "normal"),
        (f"Idade",            f"{paciente['idade']} anos", PALETTE["sub"], 10, "normal"),
        (f"Glicose",          f"{paciente['glicose']} mg/dL", PALETTE["sub"], 10, "normal"),
        (f"Pressão Arterial", f"{paciente['pressao_arterial']} mmHg", PALETTE["sub"], 10, "normal"),
        (f"IMC",              f"{paciente['imc']} kg/m²", PALETTE["sub"],  10, "normal"),
        (f"Colesterol",       f"{paciente['colesterol']} mg/dL", PALETTE["sub"], 10, "normal"),
        ("", None, PALETTE["text"], 1, "normal"),
        ("CLASSIFICAÇÃO FINAL", None, PALETTE["text"], 13, "bold"),
        (label_pred, None, cor_pred, 18, "bold"),
        ("", None, PALETTE["text"], 1, "normal"),
        (f"Acurácia do modelo: {acuracia_modelo*100:.2f}%" if acuracia_modelo else "Acurácia: N/A",
         None, PALETTE["sub"], 10, "normal"),
    ]

    y_pos = 0.97
    for item in linhas:
        rotulo, valor, cor, size, weight = item
        texto = f"{rotulo}: {valor}" if valor else rotulo
        ax_info.text(0.05, y_pos, texto, transform=ax_info.transAxes,
                     color=cor, fontsize=size, fontweight=weight, va="top")
        y_pos -= (0.04 if size >= 13 else 0.09)

    # Borda colorida indicando o risco
    for spine in ax_info.spines.values():
        spine.set_edgecolor(cor_pred)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

    # ── Painel direito: barras de probabilidade ──────────────────────
    ax_bar = fig.add_subplot(gs[1])
    estilizar_eixo(ax_bar,
                   "Probabilidades por Classe de Risco",
                   "Classe de Risco", "Probabilidade (%)")

    cores_barras = [PALETTE["roc"][i] for i in range(3)]
    vals_pct     = [p * 100 for p in probs]

    bars = ax_bar.barh(CLASS_NAMES, vals_pct, color=cores_barras,
                       alpha=0.85, edgecolor=PALETTE["bg"], linewidth=0.8,
                       height=0.5)

    # Rótulo de porcentagem à direita de cada barra
    for bar, v in zip(bars, vals_pct):
        ax_bar.text(min(v + 1.5, 97), bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", ha="left",
                    color=PALETTE["text"], fontsize=11, fontweight="bold")

    # Borda extra na barra da classe prevista
    bars[classe_pred].set_edgecolor(cor_pred)
    bars[classe_pred].set_linewidth(3)

    ax_bar.set_xlim(0, 105)
    ax_bar.xaxis.grid(True, color=PALETTE["border"], linestyle="--", alpha=0.4)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis="y", labelsize=10)

    fig.suptitle("Resultado da Predição — Risco Clínico",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig("fig4_predicao_paciente.png", dpi=150,
                bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print("    ✔ fig4_predicao_paciente.png salvo")


# ══════════════════════════════════════════════════════════════════════
# BLOCO 8 — ENTRADA INTERATIVA: NOVO PACIENTE
# Coleta os dados do paciente via input do terminal, valida cada campo
# e retorna um dicionário pronto para predição.
# ══════════════════════════════════════════════════════════════════════
def coletar_dados_paciente() -> dict:
    """
    Solicita os dados clínicos do novo paciente via terminal.
    Valida cada campo dentro dos limites fisiológicos aceitáveis.
    Retorna um dicionário com os valores validados.
    """
    print(f"\n{SEP1}")
    print("  ENTRADA DE DADOS — NOVO PACIENTE")
    print(f"{SEP1}")
    print("  Preencha os dados clínicos do paciente.")
    print("  Os valores serão validados automaticamente.\n")

    # Cada campo: (label, min, max, unidade, dica)
    campos = {
        "nome": {
            "label":   "Nome do paciente",
            "tipo":    "str",
            "dica":    "Ex: Maria"
        },
        "idade": {
            "label":   "Idade",
            "tipo":    "int",
            "min":     18, "max": 99,
            "unidade": "anos",
            "dica":    "Entre 18 e 99"
        },
        "glicose": {
            "label":   "Glicose",
            "tipo":    "float",
            "min":     60.0, "max": 400.0,
            "unidade": "mg/dL",
            "dica":    "Normal < 100 | Pré-diabético 100–125 | Diabético ≥ 126"
        },
        "pressao_arterial": {
            "label":   "Pressão Arterial Sistólica",
            "tipo":    "float",
            "min":     80.0, "max": 220.0,
            "unidade": "mmHg",
            "dica":    "Normal < 120 | Elevada 120–129 | Hipertensão ≥ 130"
        },
        "imc": {
            "label":   "IMC",
            "tipo":    "float",
            "min":     14.0, "max": 55.0,
            "unidade": "kg/m²",
            "dica":    "Normal 18.5–24.9 | Sobrepeso 25–29.9 | Obeso ≥ 30"
        },
        "colesterol": {
            "label":   "Colesterol Total",
            "tipo":    "float",
            "min":     100.0, "max": 400.0,
            "unidade": "mg/dL",
            "dica":    "Desejável < 200 | Limítrofe 200–239 | Alto ≥ 240"
        },
    }

    dados = {}
    for campo, cfg in campos.items():
        while True:
            try:
                dica  = cfg.get("dica", "")
                unit  = cfg.get("unidade", "")
                label = cfg["label"]
                prompt = f"  {label}"
                if unit:
                    prompt += f" ({unit})"
                if dica:
                    prompt += f"\n    [{dica}]"
                prompt += "\n  → "

                entrada = input(prompt).strip()

                # Tipo string (apenas para nome)
                if cfg["tipo"] == "str":
                    if not entrada:
                        print("    ⚠ Nome não pode ser vazio.\n")
                        continue
                    dados[campo] = entrada
                    break

                # Tipos numéricos
                valor = int(entrada) if cfg["tipo"] == "int" else float(entrada)
                mn, mx = cfg["min"], cfg["max"]

                if not (mn <= valor <= mx):
                    print(f"    ⚠ Valor fora do intervalo permitido: [{mn} – {mx}]\n")
                    continue

                dados[campo] = valor
                break

            except ValueError:
                tipo_str = "número inteiro" if cfg.get("tipo") == "int" else "número decimal"
                print(f"    ⚠ Por favor, insira um {tipo_str} válido.\n")

    return dados


# ══════════════════════════════════════════════════════════════════════
# BLOCO 9 — PREDIÇÃO E EXIBIÇÃO DO RESULTADO
# Recebe o paciente, aplica o scaler, prediz a classe e as
# probabilidades, e exibe um painel completo no terminal.
# ══════════════════════════════════════════════════════════════════════
def prever_paciente(paciente: dict, modelo, scaler: StandardScaler, acuracia_modelo: float = None):
    """
    Realiza a predição de risco clínico para um novo paciente.

    Fluxo:
        1. Monta DataFrame com os dados do paciente
        2. Normaliza com o mesmo scaler ajustado no treino
        3. Prediz a classe e as probabilidades
        4. Exibe resultado formatado no terminal
        5. Gera gráfico visual da predição
    """
    print(f"\n{SEP1}")
    print("  ETAPA 5 — PREDIÇÃO DO NOVO PACIENTE")
    print(SEP1)

    # ── Prepara entrada do modelo ────────────────────────────────────
    X_novo = pd.DataFrame([{
        "idade":            paciente["idade"],
        "glicose":          paciente["glicose"],
        "pressao_arterial": paciente["pressao_arterial"],
        "imc":              paciente["imc"],
        "colesterol":       paciente["colesterol"],
    }])

    # Aplica EXATAMENTE o mesmo scaler do treino (não faz novo fit!)
    X_novo_sc = scaler.transform(X_novo)

    # Predição da classe e das probabilidades por classe
    classe_pred = modelo.predict(X_novo_sc)[0]
    probs       = modelo.predict_proba(X_novo_sc)[0]  # [prob_0, prob_1, prob_2]

    # ── Exibição no terminal ─────────────────────────────────────────
    cor_label, label_risco = PALETTE["risk"][classe_pred]

    print(f"\n  {'─'*66}")
    print(f"  {'PACIENTE':12}: {paciente['nome']}")
    print(f"  {'─'*66}")
    print(f"  {'Idade':20}: {paciente['idade']} anos")
    print(f"  {'Glicose':20}: {paciente['glicose']} mg/dL")
    print(f"  {'Pressão Arterial':20}: {paciente['pressao_arterial']} mmHg")
    print(f"  {'IMC':20}: {paciente['imc']} kg/m²")
    print(f"  {'Colesterol':20}: {paciente['colesterol']} mg/dL")
    print(f"  {'─'*66}")
    print(f"\n  PROBABILIDADES PREVISTAS:")
    print(f"    {'Baixo Risco (0)':22}: {probs[0]*100:>6.2f}%  {'█' * int(probs[0]*40)}")
    print(f"    {'Risco Médio (1)':22}: {probs[1]*100:>6.2f}%  {'█' * int(probs[1]*40)}")
    print(f"    {'Risco Alto  (2)':22}: {probs[2]*100:>6.2f}%  {'█' * int(probs[2]*40)}")

    acc_str = f"{acuracia_modelo*100:.2f}%" if acuracia_modelo is not None else "N/A"
    print(f"\n  {'═'*66}")
    print(f"  CLASSIFICAÇÃO FINAL → {label_risco}")
    print(f"  Acurácia do modelo  : {acc_str}  (Random Forest — conjunto de teste)")
    print(f"  {'═'*66}\n")

    # ── Gráfico de predição ──────────────────────────────────────────
    plot_predicao_paciente(paciente, probs, classe_pred, acuracia_modelo)

    return classe_pred, probs


# ══════════════════════════════════════════════════════════════════════
# PROGRAMA PRINCIPAL — EXECUÇÃO SEQUENCIAL DO PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── ETAPA 1: Geração do dataset ──────────────────────────────────
    print(f"\n{SEP1}")
    print("  ETAPA 1 — GERAÇÃO DO DATASET (2000 registros)")
    print(SEP1)

    df = gerar_dataset(n=N_REGISTROS, seed=42)
    print(f"\n  Registros gerados : {len(df):,}")
    print(f"  Colunas           : {list(df.columns)}")
    print(f"\n  Distribuição de risco:")
    rotulos = {0: "Baixo (0)", 1: "Médio (1)", 2: "Alto (2)"}
    for k, v in df[TARGET].value_counts().sort_index().items():
        barra = "█" * int(v / 20)
        print(f"    {rotulos[k]}: {v:>5} ({v/len(df)*100:.1f}%)  {barra}")

    # ── ETAPA 2: Limpeza ─────────────────────────────────────────────
    print(f"\n{SEP1}")
    print("  ETAPA 2 — LIMPEZA E VALIDAÇÃO DOS DADOS")
    print(SEP1)
    print("\n  Verificando valores nulos e zeros inválidos...")
    df = limpar_dataset(df)
    print("  ✔ Dataset validado. Nenhuma anomalia residual detectada.")

    # ── ETAPA 3: Separação X e y ─────────────────────────────────────
    print(f"\n{SEP1}")
    print("  ETAPA 3 — SEPARAÇÃO DE FEATURES E TARGET")
    print(SEP1)

    X = df[FEATURES]     # matriz de entrada: 5 variáveis clínicas
    y = df[TARGET]       # vetor alvo: 0 = baixo, 1 = médio, 2 = alto

    print(f"\n  X shape: {X.shape}  (registros × features)")
    print(f"  y shape: {y.shape}  (classes: {sorted(y.unique())})")

    # Divisão treino/teste com estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n  Treino  : {len(X_train):>5} registros  (80%)")
    print(f"  Teste   : {len(X_test):>5} registros  (20%)")

    # Normalização: fit APENAS no treino, transform em ambos
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n  StandardScaler aplicado.")
    print(f"  (fit no treino → transform no treino e no teste)")

    # ── ETAPA 4: Treinamento e avaliação ────────────────────────────
    print(f"\n{SEP1}")
    print("  ETAPA 4 — TREINAMENTO, AVALIAÇÃO E VISUALIZAÇÕES")
    print(SEP1)

    modelos = {
        "Regressão Logística": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, metric="euclidean", weights="distance"
        ),
    }

    kfold      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}

    for nome, modelo in modelos.items():
        print(f"\n  ▶ Treinando: {nome}...")

        modelo.fit(X_train_sc, y_train)         # treina o modelo
        y_pred     = modelo.predict(X_test_sc)  # predição no teste
        cv_scores  = cross_val_score(           # validação cruzada k=5
            modelo, X_train_sc, y_train,
            cv=kfold, scoring="f1_weighted", n_jobs=-1
        )

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

        r = resultados[nome]
        print(f"    Acurácia  : {r['acuracia']:.4f}")
        print(f"    F1-Score  : {r['f1']:.4f}")
        print(f"    CV F1 μ   : {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")

    # Ranking final por CV F1 médio
    ranking = sorted(resultados.items(), key=lambda x: x[1]["cv_mean"], reverse=True)
    print(f"\n  {'─'*66}")
    print(f"  RANKING FINAL (por CV F1-Score médio):")
    medalhas = ["🥇", "🥈", "🥉"]
    for i, (nome, info) in enumerate(ranking):
        print(f"  {medalhas[i]} {i+1}º  {nome:<26}  CV-F1 = {info['cv_mean']:.4f}")
    print(f"  {'─'*66}")

    # Acessa o melhor modelo pelo nome definido na constante
    modelo_rf    = resultados[MELHOR_MODELO_NOME]["modelo"]
    y_pred_rf    = resultados[MELHOR_MODELO_NOME]["y_pred"]
    y_prob_rf    = modelo_rf.predict_proba(X_test_sc)

    # ── Gráficos ─────────────────────────────────────────────────────
    print(f"\n  Gerando visualizações...")
    plot_comparacao_modelos(resultados)
    plot_matriz_confusao(y_test, y_pred_rf, MELHOR_MODELO_NOME)
    plot_curva_roc(y_test, y_prob_rf, MELHOR_MODELO_NOME)

    # ── ETAPA 5: Coleta e predição de novo paciente ──────────────────
    paciente = coletar_dados_paciente()
    prever_paciente(paciente, modelo_rf, scaler, resultados[MELHOR_MODELO_NOME]["acuracia"])

    print(f"\n{SEP1}")
    print("  PIPELINE COMPLETO EXECUTADO COM SUCESSO ✅")
    print(f"  Arquivos gerados:")
    print(f"    fig1_comparacao_modelos.png")
    print(f"    fig2_matriz_confusao.png")
    print(f"    fig3_curva_roc.png")
    print(f"    fig4_predicao_paciente.png")
    print(SEP1 + "\n")