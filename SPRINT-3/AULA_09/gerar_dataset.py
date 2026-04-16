import numpy as np
import pandas as pd

# Seed para reprodutibilidade
np.random.seed(42)

N = 2000

# ─────────────────────────────────────────────────
# 1. NOMES FICTÍCIOS
# ─────────────────────────────────────────────────
nomes_masculinos = [
    "Carlos", "João", "Pedro", "Lucas", "Marcos", "Rafael", "Bruno",
    "Diego", "Felipe", "Gabriel", "Henrique", "Igor", "Jorge", "Klaus",
    "Leonardo", "Mateus", "Nilson", "Otávio", "Paulo", "Roberto",
    "Samuel", "Thiago", "Ulisses", "Vinícius", "Wagner", "Xavier",
    "Yago", "Zeca", "André", "Breno", "César", "Davi", "Edson",
    "Fábio", "Guilherme", "Hugo", "Ivan", "Júlio", "Leandro", "Mário"
]

nomes_femininos = [
    "Ana", "Beatriz", "Carla", "Daniela", "Elisa", "Fernanda", "Gabriela",
    "Helena", "Isabela", "Juliana", "Karen", "Larissa", "Márcia", "Natália",
    "Olivia", "Patrícia", "Quésia", "Renata", "Sandra", "Tatiana",
    "Ursula", "Valéria", "Wanda", "Xênia", "Yara", "Zilda", "Alice",
    "Bruna", "Cláudia", "Débora", "Érica", "Flávia", "Giovana", "Hanna",
    "Íris", "Jéssica", "Luana", "Mônica", "Nara", "Olga"
]

todos_nomes = nomes_masculinos + nomes_femininos
nomes = np.random.choice(todos_nomes, size=N)

# ─────────────────────────────────────────────────
# 2. IDADE (18–99 anos)
# ─────────────────────────────────────────────────
# Distribuição realista: mais concentrada entre 30–70 anos
idade = np.clip(
    np.random.normal(loc=50, scale=17, size=N).astype(int),
    18, 99
)

# ─────────────────────────────────────────────────
# 3. GLICOSE (mg/dL)
# Normal: 70–99 | Pré-diabético: 100–125 | Diabético: 126+
# ─────────────────────────────────────────────────
glicose_base = np.random.normal(loc=105, scale=28, size=N)
# Idosos tendem a ter glicose mais alta
glicose = np.clip(glicose_base + (idade - 50) * 0.3, 60, 400).round(1)

# ─────────────────────────────────────────────────
# 4. PRESSÃO ARTERIAL SISTÓLICA (mmHg)
# Normal: <120 | Elevada: 120–129 | Hipertensão: 130+
# ─────────────────────────────────────────────────
pressao_base = np.random.normal(loc=120, scale=18, size=N)
pressao = np.clip(pressao_base + (idade - 50) * 0.35, 80, 220).round(1)

# ─────────────────────────────────────────────────
# 5. IMC (kg/m²)
# Baixo peso: <18.5 | Normal: 18.5–24.9 | Sobrepeso: 25–29.9 | Obeso: 30+
# ─────────────────────────────────────────────────
imc_base = np.random.normal(loc=26.5, scale=5.2, size=N)
imc = np.clip(imc_base, 14.0, 55.0).round(1)

# ─────────────────────────────────────────────────
# 6. COLESTEROL TOTAL (mg/dL)
# Desejável: <200 | Limítrofe: 200–239 | Alto: 240+
# ─────────────────────────────────────────────────
colesterol_base = np.random.normal(loc=210, scale=42, size=N)
colesterol = np.clip(colesterol_base + (idade - 50) * 0.4, 100, 400).round(1)

# ─────────────────────────────────────────────────
# 7. CÁLCULO DO RISCO CLÍNICO
# Sistema de pontuação baseado em limiares clínicos reais
# ─────────────────────────────────────────────────

def calcular_risco(idade, glicose, pressao, imc, colesterol):
    """
    Regra clínica de pontuação de risco:
    - Cada fator de risco contribui com pontos
    - 0–2 pts → Baixo risco (0)
    - 3–5 pts → Risco médio (1)
    - 6+  pts → Risco alto (2)
    """
    pontos = np.zeros(len(idade), dtype=int)

    # Idade
    pontos += (idade >= 45).astype(int)          # +1 se 45+
    pontos += (idade >= 65).astype(int)          # +1 adicional se 65+

    # Glicose
    pontos += (glicose >= 100).astype(int)       # pré-diabético
    pontos += (glicose >= 126).astype(int)       # diabético (+1 a mais)

    # Pressão arterial
    pontos += (pressao >= 130).astype(int)       # hipertensão estágio 1
    pontos += (pressao >= 160).astype(int)       # hipertensão estágio 2

    # IMC
    pontos += (imc >= 25).astype(int)            # sobrepeso
    pontos += (imc >= 30).astype(int)            # obesidade

    # Colesterol
    pontos += (colesterol >= 200).astype(int)    # limítrofe
    pontos += (colesterol >= 240).astype(int)    # alto

    # Classificação final
    risco = np.where(pontos <= 2, 0,             # baixo risco
             np.where(pontos <= 5, 1, 2))        # médio ou alto risco

    return risco, pontos

risco, pontos = calcular_risco(idade, glicose, pressao, imc, colesterol)

# ─────────────────────────────────────────────────
# 8. MONTAR DATAFRAME
# ─────────────────────────────────────────────────
df = pd.DataFrame({
    "nome":            nomes,
    "idade":           idade,
    "glicose":         glicose,
    "pressao_arterial": pressao,
    "imc":             imc,
    "colesterol":      colesterol,
    "risco":           risco
})

# ─────────────────────────────────────────────────
# 9. SALVAR CSV
# ─────────────────────────────────────────────────
output_path = "pacientes.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

# ─────────────────────────────────────────────────
# 10. RELATÓRIO DE VALIDAÇÃO
# ─────────────────────────────────────────────────
print("=" * 55)
print("  DATASET SINTÉTICO — RISCO CLÍNICO GERADO COM SUCESSO")
print("=" * 55)
print(f"\n📁 Arquivo salvo: {output_path}")
print(f"📊 Total de registros: {len(df):,}")

print("\n── Estatísticas Descritivas ──────────────────────────")
print(df[["idade","glicose","pressao_arterial","imc","colesterol"]].describe().round(2).to_string())

print("\n── Distribuição da Variável Alvo (risco) ─────────────")
dist = df["risco"].value_counts().sort_index()
labels = {0: "Baixo risco (0)", 1: "Risco médio (1)", 2: "Risco alto (2)"}
for k, v in dist.items():
    pct = v / len(df) * 100
    print(f"  {labels[k]:<22}: {v:>5} registros  ({pct:.1f}%)")

print("\n── Primeiras 5 linhas ────────────────────────────────")
print(df.head().to_string(index=False))
print("\n" + "=" * 55)
