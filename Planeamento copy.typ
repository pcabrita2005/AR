#set text(lang: "pt")

#import "@preview/fine-lncs:0.4.0": lncs, institute, author, theorem, proof

#let inst_minho = institute("Universidade do Minho")

#show: lncs.with(
  title: "Self-Play Connect Four",
  authors: (
    author("Afonso Sousa", insts: (inst_minho)),
    author("Luís Cunha", insts: (inst_minho)),
    author("Paulo Cabrita", insts: (inst_minho)),
    author("Vasco Macedo", insts: (inst_minho)),
  ),
)

#set page(
  header: context {
    let p = counter(page).get().first()

    if calc.even(p) {
      align(left)[Aprendizagem por Reforço]
    } else {
      align(right)[Self-Play Connect Four]
    }
  },
  footer: context {
    align(center)[#counter(page).display()]
  }
)


= Enquadramento do Problema

O Connect Four é um jogo de dois jogadores com informação perfeita, jogado numa grelha 6×7. Cada jogador insere alternadamente uma peça numa coluna; o primeiro a alinhar quatro peças consecutivas (horizontal, vertical ou diagonal) vence. A combinação de informação perfeita e dinâmica de soma nula torna-o ideal para _self-play_ com aprendizagem por reforço. O objetivo é treinar um agente competente exclusivamente através de auto-interação, sem dados humanos.

= Metodologia e Cronograma

O trabalho será executado por etapas, com objetivos incrementais e critérios de conclusão (_done_) explícitos. A estratégia do grupo é concluir primeiro os dois algoritmos principais (DQN e PPO), de modo a garantir uma solução base sólida, e apenas depois avançar para o AlphaZero simplificado caso exista folga temporal.

== Cronograma proposto

#figure(caption: [Cronograma do Trabalho])[
  #table(
    columns: (90pt, 130pt, auto),
    align: left + bottom,
    inset: (x: 6pt, y: 4pt),
    table.hline(),
    [*Etapa*], [*Período*], [*Resultado esperado*],
    table.hline(),
    [Planeamento], [até 30/04/2026], [Documento de planeamento submetido no GitHub do grupo.],
    [DQN], [01/05/2026 a 15/05/2026], [Pipeline de treino e avaliação do DQN funcional.],
    [PPO], [16/05/2026 a 31/05/2026], [Pipeline de treino e avaliação do PPO funcional.],
    [AlphaZero (opcional)], [até 31/05/2026], [Protótipo funcional com MCTS + rede política/valor, caso exista folga temporal.],
    [Apresentação do trabalho], [02/06/2026], [Apresentação final, abordagem e resultados obtidos até à data. Recolha de feedback para o relatório final.],
    [Relatório final], [03/06/2026 a 15/06/2026], [Relatório consolidado com metodologia, resultados e discussão.],
    [Entrega final], [até 16/06/2026], [Submissão final do relatório e do código desenvolvido.],
    table.hline(),
  )
]

== Critérios de conclusão (_done_)

- DQN e PPO: agentes treináveis com avaliação contra baselines.
- AlphaZero (opcional): versão simplificada funcional.
- Relatório: metodologia, configuração experimental, resultados e conclusões.
- Código: repositório organizado e executável.

= Ambiente e Dados

#figure(caption: [Características do Ambiente])[
  #table(
    columns: (120pt, auto),
    align: left + bottom,
    inset: (x: 6pt, y: 4pt),
    table.hline(),
    [*Componente*], [*Descrição*],
    table.hline(),
    [Tabuleiro], [Matriz 6 × 7; cada célula assume o valor 0 (vazia), 1 (jogador 1) ou 2 (jogador 2).],
    [Representação do Estado], [Tensor binário 2 × 6 × 7: um canal por jogador, com 1 nas posições ocupadas e 0 nas restantes.],
    [Ações], [Espaço discreto com 7 ações possíveis (uma por coluna); as colunas completas são marcadas como inválidas.],
    [Recompensa], [+1 em caso de vitória, –1 em caso de derrota, 0 em caso de empate ou durante o jogo.],
    [Estados Terminais], [Quatro peças alinhadas pelo jogador ativo ou tabuleiro completamente preenchido.],
    [Implementação], [PettingZoo (connect_four_v3) ou implementação própria compatível com a API Gymnasium.], 
    table.hline(),
  )
]

Será utilizada a implementação _Connect Four_ do PettingZoo (`connect_four_v3`) ou alternativa compatível com Gymnasium. Para reprodutibilidade: sementes fixas, hiperparâmetros versionados e _checkpoints_ registados por iteração.

= Algoritmos a implementar

Serão implementados e comparados dois algoritmos principais. Um terceiro algoritmo poderá ser implementado, caso seja viável até à data da entrega do trabalho.

== Deep Q-Network com Self-Play (DQN)

O DQN é o algoritmo base. Aprende Q-valores via rede convolucional com _experience replay_, rede-alvo e _opponent pool_ para _self-play_.

== Proximal Policy Optimization (PPO)

Algoritmo de gradiente de política com _clipping_ para estabilidade. O agente assume ambos os papéis em cada episódio com recompensa invertida. Maior estabilidade que DQN em ambientes estocásticos.

== AlphaZero Simplificado (opcional)

Se viável, versão simplificada combinando rede política/valor com MCTS. MCTS guia exploração durante treino e inferência. Objetivo secundário.

= Ferramentas e Packages de Software

#figure(caption: [Ferramentas e Packages])[
  #table(
    columns: 3,
    align: left + bottom,
    table.hline(),
    [*Package/Ferramenta*], [*Versão*], [*Utilização*],
    table.hline(),
    [Python], [>= 3.10], [Linguagem principal de implementação.],
    [Pytorch], [>= 2.0], [Definição e treino das redes neuronais.],
    [PettingZoo], [>= 1.24], [Ambiente multi-agente do Connect Four.],
    [Numpy], [>= 1.24], [Manipulação de estados e vetores de recompensa.],
    [Matplotlib], [>= 3.7], [Visualização de curvas de treino e métricas.],
    [Github], [---], [Controlo de versões e entrega do projeto.],
    [Gymnasium], [>= 0.29], [API de ambiente para integração e testes de compatibilidade.],
    table.hline(),
  )
]

= Validação das Soluções

A avaliação será conduzida a vários níveis, de forma a verificar tanto a correção da implementação como a qualidade efetiva do agente.

== *Baselines de referência*
- Aleatório: seleciona uniformemente entre colunas válidas (objetivo: >= 95% vitória em 200+ jogos).
- Heurístico: bloqueia ameaças e completa alinhamentos próprios.

== Métricas de Treino

- Taxa de vitória móvel (_rolling win rate_) contra a versão anterior do agente e contra os baselines, ao longo das iterações de treino.
- _Elo rating_ calculado por torneio entre _checkpoints_ do agente em diferentes fases do treino, permitindo quantificar a evolução relativa.
- Curvas de aprendizagem: _Loss_ da rede (_policy loss_, _value loss_), recompensa média por episódio e entropia da política.

== Comparação entre algoritmos

Torneio _round-robin_: DQN vs PPO vs AlphaZero (se implementado). Mínimo 200 partidas por par, com papéis alternados para eliminar viés posicional.

== Análise qualitativa

Exame de partidas representativas: controlo do centro, construção de ameaças duplas (_forks_) e bloqueio preventivo.