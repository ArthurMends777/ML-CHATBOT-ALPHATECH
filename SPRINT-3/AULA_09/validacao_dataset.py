import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# CARREGAMENTO
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("pacientes.csv")

SEP = "─" * 60

print("\n" + "═" * 60)
print("  CARREGAMENTO DO DATASET")
print("═" * 60)
print(f"  Registros carregados : {len(df):,}")
print(f"  Colunas              : {list(df.columns)}")
print(f"  Shape                : {df.shape}")


# ═════════════════════════════════════════════════════════════
# ETAPA 1 — DETECÇÃO E CORREÇÃO DE VALORES INVÁLIDOS
# ═════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  ETAPA 1 — DETECÇÃO E CORREÇÃO DE VALORES INVÁLIDOS")
print("═" * 60)

# Colunas onde zero ou nulo é clinicamente impossível
COLUNAS_CRITICAS = {
    "glicose":           "mg/dL  — zero incompatível com vida",
    "pressao_arterial":  "mmHg   — zero incompatível com vida",
    "imc":               "kg/m²  — zero impossível para adulto vivo",
}

# ── 1A. VALORES NULOS ─────────────────────────────────────────
print(f"\n{'─'*60}")
print("  1A › Verificação de Valores NULOS (NaN)")
print(f"{'─'*60}")

nulos_total = df.isnull().sum()
nulos_criticos = nulos_total[list(COLUNAS_CRITICAS.keys())]

print(f"\n  {'Coluna':<22} {'Nulos encontrados':>18}   Unidade")
print(f"  {'─'*22} {'─'*18}   {'─'*30}")
for col, descricao in COLUNAS_CRITICAS.items():
    n = nulos_criticos[col]
    flag = " ⚠ REQUER CORREÇÃO" if n > 0 else " ✓ OK"
    print(f"  {col:<22} {n:>18}   {descricao}{flag}")

# ── 1B. VALORES ZERO ──────────────────────────────────────────
print(f"\n{'─'*60}")
print("  1B › Verificação de Valores ZERO (clinicamente inválidos)")
print(f"{'─'*60}")

zeros = {col: (df[col] == 0).sum() for col in COLUNAS_CRITICAS}

print(f"\n  {'Coluna':<22} {'Zeros encontrados':>18}   Unidade")
print(f"  {'─'*22} {'─'*18}   {'─'*30}")
for col, descricao in COLUNAS_CRITICAS.items():
    n = zeros[col]
    flag = " ⚠ REQUER CORREÇÃO" if n > 0 else " ✓ OK"
    print(f"  {col:<22} {n:>18}   {descricao}{flag}")

# ── 1C. INJEÇÃO CONTROLADA DE PROBLEMAS (demonstração) ────────
# Para tornar o exercício demonstrável, injeta 15 zeros e 10 NaNs
# aleatórios nas colunas críticas (em linhas distintas)
print(f"\n{'─'*60}")
print("  1C › Injeção controlada de anomalias para demonstração")
print(f"{'─'*60}")

df_original = df.copy()  # guarda cópia limpa para comparação

rng = np.random.default_rng(seed=99)

anomalias_injetadas = {}
for col in COLUNAS_CRITICAS:
    idx_zero = rng.choice(df.index, size=8, replace=False)
    idx_nan  = rng.choice(df.index, size=5, replace=False)
    df.loc[idx_zero, col] = 0
    df.loc[idx_nan,  col] = np.nan
    anomalias_injetadas[col] = {"zeros": len(idx_zero), "nulos": len(idx_nan)}
    print(f"  • {col:<22}: {len(idx_zero)} zeros + {len(idx_nan)} NaNs injetados")

# ── 1D. AUDITORIA PÓS-INJEÇÃO ─────────────────────────────────
print(f"\n{'─'*60}")
print("  1D › Auditoria — Estado do dataset ANTES da correção")
print(f"{'─'*60}")

print(f"\n  {'Coluna':<22} {'Nulos':>8} {'Zeros':>8}  {'Total anomalias':>16}")
print(f"  {'─'*22} {'─'*8} {'─'*8}  {'─'*16}")
for col in COLUNAS_CRITICAS:
    n_nulos = df[col].isnull().sum()
    n_zeros = (df[col] == 0).sum()
    total   = n_nulos + n_zeros
    print(f"  {col:<22} {n_nulos:>8} {n_zeros:>8}  {total:>16}")

# ── 1E. CORREÇÃO PELA MEDIANA ──────────────────────────────────
print(f"\n{'─'*60}")
print("  1E › Correção — Substituição pela MEDIANA")
print(f"{'─'*60}")
print()
print("  Estratégia: zeros são primeiro convertidos para NaN,")
print("  depois todos os NaN são imputados com a mediana da coluna.")
print("  A mediana é calculada EXCLUINDO os valores inválidos,")
print("  preservando a distribuição real dos dados.")
print()

relatorio_correcao = []

for col in COLUNAS_CRITICAS:
    # Converte zeros para NaN
    n_zeros_antes = (df[col] == 0).sum()
    df[col] = df[col].replace(0, np.nan)

    # Calcula mediana excluindo NaN
    mediana = df[col].median()
    n_nulos_antes = df[col].isnull().sum()

    # Aplica imputação
    df[col] = df[col].fillna(mediana)

    relatorio_correcao.append({
        "coluna":   col,
        "zeros_corrigidos": n_zeros_antes,
        "nulos_corrigidos":  n_nulos_antes,
        "mediana_usada":     round(mediana, 2)
    })

    print(f"  ✔ {col}")
    print(f"      Zeros convertidos : {n_zeros_antes}")
    print(f"      NaNs imputados     : {n_nulos_antes}")
    print(f"      Mediana aplicada   : {mediana:.2f}")
    print()

# ── 1F. VERIFICAÇÃO FINAL ──────────────────────────────────────
print(f"{'─'*60}")
print("  1F › Verificação APÓS correção — deve ser tudo ZERO")
print(f"{'─'*60}\n")

tudo_ok = True
for col in COLUNAS_CRITICAS:
    n_nulos = df[col].isnull().sum()
    n_zeros = (df[col] == 0).sum()
    status  = "✓ OK" if (n_nulos + n_zeros) == 0 else "✗ AINDA COM PROBLEMAS"
    print(f"  {col:<22}  nulos={n_nulos}  zeros={n_zeros}   {status}")
    if (n_nulos + n_zeros) > 0:
        tudo_ok = False

print()
if tudo_ok:
    print("  ✅ Dataset íntegro — nenhuma anomalia residual detectada.")
else:
    print("  ❌ Atenção: ainda existem anomalias. Revisar pipeline.")

# Salva dataset corrigido
df.to_csv("pacientes_corrigido.csv", index=False, encoding="utf-8-sig")
print(f"\n  💾 Dataset corrigido salvo como: pacientes_corrigido.csv")


# ═════════════════════════════════════════════════════════════
# ETAPA 2 — RANKING: TOP 10 GLICOSE E TOP 10 COLESTEROL
# ═════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  ETAPA 2 — RANKING DE PACIENTES POR BIOMARCADORES")
print("═" * 60)

COLUNAS_RANKING = {
    "glicose":    ("mg/dL", "🩸 TOP 10 — MAIORES NÍVEIS DE GLICOSE"),
    "colesterol": ("mg/dL", "🫀 TOP 10 — MAIORES NÍVEIS DE COLESTEROL"),
}

LABEL_RISCO = {0: "Baixo", 1: "Médio", 2: "Alto ⚠"}

for col, (unidade, titulo) in COLUNAS_RANKING.items():
    print(f"\n{'─'*60}")
    print(f"  {titulo}")
    print(f"{'─'*60}")
    print(f"\n  {'Pos':<5} {'Nome':<14} {'Idade':>5} {'Valor':>10}  {'Risco':<10}")
    print(f"  {'─'*5} {'─'*14} {'─'*5} {'─'*10}  {'─'*10}")

    top10 = (
        df[["nome", "idade", col, "risco"]]
        .sort_values(by=col, ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    for i, row in top10.iterrows():
        pos        = f"#{i+1}"
        nome       = row["nome"]
        idade      = int(row["idade"])
        valor      = f"{row[col]:.1f} {unidade}"
        risco_str  = LABEL_RISCO[int(row["risco"])]
        print(f"  {pos:<5} {nome:<14} {idade:>5} {valor:>10}  {risco_str}")

    media_top10 = top10[col].mean()
    media_geral = df[col].mean()
    delta       = ((media_top10 - media_geral) / media_geral) * 100

    print(f"\n  Média do grupo TOP 10 : {media_top10:.1f} {unidade}")
    print(f"  Média geral do dataset: {media_geral:.1f} {unidade}")
    print(f"  Desvio relativo       : +{delta:.1f}% acima da média")

print("\n" + "═" * 60)
print("  PIPELINE CONCLUÍDO COM SUCESSO")
print("═" * 60 + "\n")
