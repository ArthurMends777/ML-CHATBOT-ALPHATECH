# ============================================================
# EXERCÍCIO 6 — Chatbot Multi-Personalidade
# ============================================================

class Personalidade:
    def __init__(self, nome, tom, prefixo, vocab_proibido):
        # ✅ TODO 1: inicializa os atributos
        self.nome = nome
        self.tom = tom
        self.prefixo_resposta = prefixo
        self.vocabulario_proibido = vocab_proibido

PERSONALIDADES = {
    'formal': Personalidade(
        nome='Assistente Formal',
        tom='cordial e profissional',
        prefixo='Prezado usuário, ',
        vocab_proibido=['cara', 'mano', 'oi', 'beleza', 'vlw']
    ),
    'casual': Personalidade(
        nome='ChatZinho',
        tom='descontraído e jovem',
        prefixo='Oi! ',
        vocab_proibido=['prezado', 'solicito', 'outrossim', 'conforme']
    ),
    'tecnico': Personalidade(
        nome='TechBot',
        tom='técnico e direto',
        prefixo='RESPOSTA: ',
        vocab_proibido=['acho', 'talvez', 'quem sabe']
    ),
}

RESPOSTAS_GENERICAS = {
    'python': 'Python é uma linguagem de programação interpretada e de alto nível.',
    'chatbot': 'Chatbot é um sistema que simula conversas humanas de forma automatizada.',
    'ia': 'Inteligência Artificial é a capacidade de máquinas realizarem tarefas cognitivas.',
    'ajuda': 'Posso falar sobre: python, chatbot, ia. Troque personalidade com /modo.',
}

class ChatbotMultiPersonalidade:
    def __init__(self):
        self.personalidade_ativa = PERSONALIDADES['formal']
        self.historico = []

    def trocar_personalidade(self, nome: str) -> str:
        # ✅ TODO 2: troca a personalidade e confirma
        if nome in PERSONALIDADES:
            self.personalidade_ativa = PERSONALIDADES[nome]
            return f'Personalidade alterada para: {self.personalidade_ativa.nome}'
        return f'Personalidade "{nome}" não encontrada. Opções: formal, casual, tecnico.'

    def gerar_resposta(self, mensagem: str) -> str:
        # ✅ TODO 3: busca resposta, aplica prefixo e filtra vocabulário
        msg = mensagem.lower().strip()

        # 1. Busca resposta base
        resposta_base = None
        for chave in RESPOSTAS_GENERICAS:
            if chave in msg:
                resposta_base = RESPOSTAS_GENERICAS[chave]
                break
        if not resposta_base:
            resposta_base = 'Não entendi. Digite "ajuda" para ver os tópicos disponíveis.'

        # 2. Aplica o prefixo da personalidade ativa
        resposta = self.personalidade_ativa.prefixo_resposta + resposta_base

        # 3. Filtra vocabulário proibido
        for palavra in self.personalidade_ativa.vocabulario_proibido:
            resposta = resposta.replace(palavra, '***')

        return resposta

    def executar(self):
        print('=== Chatbot Multi-Personalidade ===')
        print('Comandos: /modo formal | /modo casual | /modo tecnico | sair')
        print(f'Personalidade ativa: {self.personalidade_ativa.nome}')
        print()
        while True:
            entrada = input('Você: ').strip()
            if entrada.lower() == 'sair':
                print('Encerrando. Até logo!')
                break
            elif entrada.lower().startswith('/modo '):
                nome = entrada.split(' ', 1)[1].lower()
                print(f'Bot: {self.trocar_personalidade(nome)}')
            else:
                resposta = self.gerar_resposta(entrada)
                print(f'Bot: {resposta}')
            print()

bot = ChatbotMultiPersonalidade()
bot.executar()