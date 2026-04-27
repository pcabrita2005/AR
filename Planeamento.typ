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
  }
)


= Enquadramento do Problema

O Connect Four é um jogo de dois jogadores com informação perfeita e espaço de estados finito, jogado num tabuleiro de 6 linhas por 7 colunas. Cada jogador joga alternadamente inserindo uma peça numa coluna; o primeiro a alinhar quatro peças consecutivas, na horizontal, vertical ou diagonal, vence. O jogo termina em empate caso o tabuleiro fique completamente preenchido sem vencedor.

A combinação de informação perfeita, espaço de estados e dinâmica de soma nula torna o Connect Four um jogo ideal para aplicar técnicas de _self-play_ com aprendizagem por reforço. O objetivo deste trabalho é treinar um agente capaz de jogar a um nível competente exclusivamente através de interações consigo próprio, sem recurso a dados humanos.

= Metodologia e Cronograma

O trabalho será executado por etapas, com objetivos incrementais e critérios de conclusão (_done_) explícitos. A estratégia do grupo é concluir primeiro os dois algoritmos principais (DQN e PPO), de modo a garantir uma solução base sólida, e apenas depois avançar para o AlphaZero simplificado caso exista folga temporal.

== Etapas de trabalho

1. Entrega do planeamento (este documento).
2. Desenvolvimento do DQN.
3. Desenvolvimento do PPO.
4. Desenvolvimento do AlphaZero simplificado (opcional).
5. Apresentação do trabalho.
6. Elaboração do relatório final.
7. Entrega do relatório e do código desenvolvido.

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

O grupo dispõe de aproximadamente um mês para desenvolvimento técnico. Em termos de planeamento base, esse período é distribuído por cerca de duas semanas para DQN e duas semanas para PPO. Se alguma implementação for concluída antes do prazo previsto e sobrar aproximadamente uma semana, essa folga será alocada ao desenvolvimento do AlphaZero simplificado.

== Critérios de conclusão (_done_)

- DQN concluído: agente treinável, avaliação contra baselines e resultados registados.
- PPO concluído: agente treinável, avaliação contra baselines e resultados registados.
- AlphaZero concluído (se aplicável): versão simplificada funcional e comparável com DQN/PPO.
- Relatório final concluído: documento com metodologia, configuração experimental, resultados e conclusões.
- Código concluído: repositório organizado e executável; a entrega pode ser faseada ao longo do projeto ou realizada integralmente na entrega final.

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

Não serão utilizados datasets externos. O único âmbito de treino é o ambiente sintético gerado pelas próprias partidas simuladas. 

Como fonte do ambiente, será utilizada a implementação _Connect Four_ do PettingZoo (`connect_four_v3`), podendo ser usada uma implementação própria desde que mantenha compatibilidade com a API Gymnasium e comportamento equivalente nas regras do jogo. Para melhorar a reprodutibilidade, os ensaios serão executados com sementes (_random seeds_) fixas, configuração de hiperparâmetros versionada e registo de _checkpoints_ por iteração.

= Algoritmos a implementar

Serão implementados e comparados dois algoritmos principais. Um terceiro algoritmo poderá ser implementado, caso seja viável até à data da entrega do trabalho.

== Deep Q-Network com Self-Play (DQN)

O DQN será o algoritmo de base (_baseline_). O agente aprende uma função de valor Q através de uma rede neuronal convolucional que recebe o estado do tabuleiro e devolve os valores esperados para cada ação. O _self-play_ é concretizado mantendo uma cópia ‘congelada’ da rede (_opponent pool_), atualizada periodicamente. Serão utilizadas as técnicas _standard_ de _experience replay_ e rede-alvo (_target network_).

== Proximal Policy Optimization (PPO)

O PPO é um algoritmo de gradiente de política que otimiza diretamente a probabilidade das ações tomadas, com um mecanismo de _clipping_ que estabiliza o treino. Em contexto de dois jogadores alternados, o agente assume os dois papéis em cada episódio, com a recompensa a ser invertida para o segundo jogador. O PPO tende a apresentar melhor estabilidade de treino do que o DQN em ambientes estocásticos.

== AlphaZero Simplificado (opcional)

Se o progresso do trabalho o permitir, será implementada uma versão simplificada do AlphaZero, combinando uma rede de política/valor com Monte Carlo Tree Search (MCTS). O MCTS guia a exploração durante o treino e a inferência, produzindo tipicamente agentes substancialmente mais fortes. Esta componente é considerada um objetivo secundário (_stretch goal_).

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
- Agente aleatório: Seleciona uniformemente entre as colunas válidas. O objetivo é atingir taxa de vitória elevada e estável (por exemplo, >= 95% em pelo menos 200 jogos), evitando conclusões com base em poucas amostras.
- Agente heurístico: Bloqueia as ameaças imediatas do adversário e completa alinhamentos próprios quando possível. Constitui um teste de dificuldade intermediária.

== Métricas de Treino

- Taxa de vitória móvel (_rolling win rate_) contra a versão anterior do agente e contra os baselines, ao longo das iterações de treino.
- _Elo rating_ calculado por torneio entre _checkpoints_ do agente em diferentes fases do treino, permitindo quantificar a evolução relativa.
- Curvas de aprendizagem: _Loss_ da rede (_policy loss_, _value loss_), recompensa média por episódio e entropia da política.

== Comparação entre algoritmos

Será realizado um torneio _round-robin_ entre os agentes finais treinados com DQN e PPO (e AlphaZero, se este vier a ser implementado). Cada par de agentes disputará um mínimo de 200 partidas, com os papéis, primeiro ou segundo jogador, alternados equitativamente para eliminar viés posicional. Os resultados serão apresentados numa tabela de resultados e discutidos à luz das diferenças arquiteturais e de otimização entre os algoritmos.

== Análise qualitativa

Serão examinadas as partidas representativas para verificar se o agente aprendeu padrões estratégicos conhecidos do Connect Four, nomeadamente: controlo do centro do tabuleiro, construção de ameaças duplas (_fork_) e bloqueio preventivo de ameaças do adversário.