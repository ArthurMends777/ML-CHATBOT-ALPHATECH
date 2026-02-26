import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

dados = {
    'qtd_exclamacao': [3, 0, 1, 5, 0, 2, 4, 0, 1, 6],
    'tem_palavra_alerta': [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    'tamanho_msg': [120, 45, 60, 200, 30, 150, 180, 25, 70, 220],
    'rotulo': [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(dados)

# 1. Preparação (Usando as features da Aula 02)
# Supondo que o df já tenha as colunas: 'qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg'
X = df[['qtd_exclamacao', 'tem_palavra_alerta', 'tamanho_msg']]
y = df['rotulo'] # 1: Alta Prioridade, 0: Comum

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Treinamento
clf = DecisionTreeClassifier(max_depth=3) # Limitamos a profundidade para ser legível
clf.fit(X_train, y_train)

# 3. Visualização da Lógica (Explicação para os alunos)
# Diferente de uma "caixa preta", a Árvore nos mostra as regras que criou
regras = export_text(clf, feature_names=['Exclamações', 'Palavra Alerta', 'Tamanho'])
print("--- Regras Decididas pelo Modelo ---")
print(regras)