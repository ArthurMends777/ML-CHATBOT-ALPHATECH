from collections import Counter

texto = "chatbot chatbot inteligência artificial chatbot aprendizado"

# TODO: transformar texto em lista de palavras
palavras = texto.split()

# TODO: calcular frequência
frequencia = Counter(palavras)

print(frequencia)

# Resultado esperado: {'chatbot':3, 'inteligência':1, 'artificial':1, 'aprendizado':1}
