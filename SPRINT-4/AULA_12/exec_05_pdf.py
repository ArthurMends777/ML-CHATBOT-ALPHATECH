# ============================================================
# EXERCÍCIO 5 — Intent + Entity Extraction
# Chatbot de Pedido de Pizza
# ============================================================
import re

SABORES = ['calabresa', 'frango', 'queijo', 'portuguesa', 'vegetariana', 'pepperoni']
TAMANHOS = ['pequena', 'media', 'média', 'grande', 'gigante', 'família', 'familia']

SINONIMOS_TAMANHO = {
    'pequena':   ['pequena', 'pequeno', 'mini', 'pequenina', 'pequenininha', 'pizzinha p'],
    'média':     ['media', 'média', 'médio', 'medio', 'mediana'],
    'grande':    ['grande', 'grandão', 'grandona', 'grandao'],
    'gigante':   ['gigante', 'enorme', 'gigantesca', 'grandíssima'],
    'família':   ['família', 'familia', 'familiar', 'para família'],
}

SINONIMOS_SABOR = {
    'calabresa':   ['calabresa', 'linguiça', 'linguica'],
    'frango':      ['frango', 'galinha', 'chicken'],
    'queijo':      ['queijo', 'mussarela', 'mozarela', 'mozza', '4 queijos', 'quatro queijos'],
    'portuguesa':  ['portuguesa', 'portugues'],
    'vegetariana': ['vegetariana', 'vegana', 'sem carne', 'veggie'],
    'pepperoni':   ['pepperoni', 'pepperoni clássico'],
}


def detectar_intencao(mensagem: str) -> str:
    msg = mensagem.lower()
    if any(p in msg for p in ['quero', 'pedir', 'pedido', 'comprar', 'queria']):
        return 'FAZER_PEDIDO'
    elif any(p in msg for p in ['cancelar', 'desistir', 'não quero']):
        return 'CANCELAR'
    elif any(p in msg for p in ['cardápio', 'cardapio', 'opções', 'opcoes', 'tem']):
        return 'VER_CARDAPIO'
    return 'DESCONHECIDO'

def extrair_entidades(mensagem: str) -> dict:
    msg = mensagem.lower()

    sabor_encontrado = None
    tamanho_encontrado = None

    # Busca sabor: percorre o dicionário de sinônimos
    for canonico, variacoes in SINONIMOS_SABOR.items():
        for variacao in variacoes:
            if variacao in msg:
                sabor_encontrado = canonico
                break
        if sabor_encontrado:
            break
    for canonico, variacoes in SINONIMOS_TAMANHO.items():
        for variacao in variacoes:
            if variacao in msg:
                tamanho_encontrado = canonico
                break
        if tamanho_encontrado:
            break

    return {'sabor': sabor_encontrado, 'tamanho': tamanho_encontrado}

def confirmar_pedido(entidades: dict) -> str:
    sabor   = entidades.get('sabor')
    tamanho = entidades.get('tamanho')

    if sabor and tamanho:
        return (
            f"✅ Pedido confirmado!\n"
            f"   🍕 Pizza {tamanho} de {sabor}\n"
            f"   Seu pedido está sendo preparado. Aguarde!"
        )
    elif sabor and not tamanho:
        tamanhos_disponiveis = ', '.join(['Pequena', 'Média', 'Grande', 'Gigante', 'Família'])
        return (
            f"Ótima escolha! Pizza de {sabor} 😋\n"
            f"   Qual tamanho você quer? ({tamanhos_disponiveis})"
        )
    elif tamanho and not sabor:
        sabores_disponiveis = ', '.join(SABORES)
        return (
            f"Pizza {tamanho}, certo!\n"
            f"   Qual sabor você prefere? ({sabores_disponiveis})"
        )
    else:
        return (
            "Não consegui identificar o sabor nem o tamanho. 🤔\n"
            f"   Sabores disponíveis: {', '.join(SABORES)}\n"
            f"   Tamanhos: Pequena, Média, Grande, Gigante, Família\n"
            "   Tente: 'quero uma pizza grande de calabresa'"
        )

def chatbot_pizza():
    print('=== PizzaBot — Faça seu pedido! ===')
    print('(Digite "sair" ou "tchau" para encerrar)\n')

    while True:
        entrada = input('Você: ').strip()
        if not entrada:
            continue
        if entrada.lower() in ('sair', 'tchau'):
            print('Bot: Pedido cancelado. Até logo! 👋')
            break

        intencao = detectar_intencao(entrada)
        print(f'  [DEBUG] Intenção: {intencao}')

        if intencao == 'VER_CARDAPIO':
            print(f'Bot: Sabores disponíveis: {", ".join(SABORES)}')
            print(f'Bot: Tamanhos: Pequena, Média, Grande, Gigante, Família')

        elif intencao == 'FAZER_PEDIDO':
            entidades = extrair_entidades(entrada)
            print(f'  [DEBUG] Entidades: {entidades}')
            resposta = confirmar_pedido(entidades)
            print(f'Bot: {resposta}')

        elif intencao == 'CANCELAR':
            print('Bot: Seu pedido foi cancelado.')

        else:
            print('Bot: Não entendi. Você pode pedir uma pizza ou ver o cardápio.')

        print()


if __name__ == '__main__':
    chatbot_pizza()