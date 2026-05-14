# ============================================================
# EXERCÍCIO 4 — Chatbot com Memória de Contexto
# ============================================================

from datetime import datetime

historico = []

RESPOSTAS = {
    'nome': 'Meu nome é ByteBot, seu assistente digital!',
    'ajuda': 'Posso responder sobre: nome, turno, historico, hora.',
    'hora': lambda: f'Agora são {datetime.now().strftime("%H:%M:%S")}.',
}

def processar_mensagem(mensagem: str, turno: int) -> str:
    msg = mensagem.lower().strip()

    if 'nome' in msg:
        return RESPOSTAS['nome']
    elif 'ajuda' in msg:
        return RESPOSTAS['ajuda']
    elif 'hora' in msg:
        return RESPOSTAS['hora']()
    elif 'turno' in msg:
        # ✅ TODO 1: retorna quantos turnos já ocorreram
        quantidade = len(historico)
        if quantidade == 0:
            return 'Este é o nosso primeiro turno!'
        return f'Já tivemos {quantidade} turno(s) de conversa.'
    elif 'historico' in msg or 'histórico' in msg:
        # ✅ TODO 2: resumo das últimas 3 mensagens do usuário
        if not historico:
            return 'Ainda não temos histórico.'
        ultimas = historico[-3:]
        resumo = ' | '.join([f"T{t['turno']}: '{t['usuario']}'" for t in ultimas])
        return f'Últimas mensagens: {resumo}'
    else:
        return 'Não entendi. Digite "ajuda" para ver o que posso fazer.'

def registrar_turno(turno: int, usuario: str, bot: str):
    # ✅ TODO 3: adiciona o turno ao histórico
    historico.append({
        'turno': turno,
        'usuario': usuario,
        'bot': bot
    })

def chatbot_contextual():
    print('=== ByteBot — Chatbot com Memória ===')
    turno = 0
    while True:
        entrada = input('Você: ').strip()
        if entrada.lower() in ('sair', 'tchau'):
            print('Bot: Até mais! Foram', turno, 'turnos de conversa.')
            break
        turno += 1
        resposta = processar_mensagem(entrada, turno)
        registrar_turno(turno, entrada, resposta)
        print(f'Bot: {resposta}')
        print()

chatbot_contextual()