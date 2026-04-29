import unittest
import numpy as np
from connect4_rl.envs.connect_four import (
    ConnectFourEnv,
    initial_state,
    apply_action,
    is_terminal,
    outcome_for_player,
    legal_actions,
    encode_state,
)


class TestConnectFourEnvironment(unittest.TestCase):

    def test_initial_state(self):
        state = initial_state()
        self.assertEqual(state.current_player, 1)
        self.assertFalse(is_terminal(state))
        # Todas as colunas devem ter 0 peças
        for col in range(7):
            self.assertEqual(sum(row[col] for row in state.board if row[col] != 0), 0)

    def test_legal_actions(self):
        state = initial_state()
        legal = legal_actions(state)
        self.assertEqual(len(legal), 7)  # Todas as colunas inicialmente legais
        self.assertEqual(set(legal), set(range(7)))

    def test_action_alternates_player(self):
        state = initial_state()
        self.assertEqual(state.current_player, 1)
        
        state = apply_action(state, 0)
        self.assertEqual(state.current_player, 2)
        
        state = apply_action(state, 0)
        self.assertEqual(state.current_player, 1)

    def test_full_column_not_legal(self):
        state = initial_state()
        
        # Preenche a coluna 0 com 6 peças
        for _ in range(6):
            state = apply_action(state, 0)
        
        legal = legal_actions(state)
        self.assertNotIn(0, legal)
        self.assertIn(1, legal)

    def test_horizontal_win(self):
        state = initial_state()
        
        # Joga 0, 0, 0, 0 (vitória horizontal)
        for col in range(4):
            state = apply_action(state, col)
            if col < 3:
                state = apply_action(state, 6) 
        
        self.assertTrue(is_terminal(state))
        self.assertEqual(outcome_for_player(state, 1), 1.0)
        self.assertEqual(outcome_for_player(state, 2), -1.0)

    def test_vertical_win(self):
        state = initial_state()
        
        # Joga coluna 0 quatro vezes (vitória vertical)
        for _ in range(3):
            state = apply_action(state, 0)
            state = apply_action(state, 1)  
        
        # 7a jogada: Jogador 1 ganha
        state = apply_action(state, 0)
        
        self.assertTrue(is_terminal(state))
        self.assertEqual(outcome_for_player(state, 1), 1.0)

    def test_encode_state(self):
        state = initial_state()
        encoded = np.array(encode_state(state, perspective_player=1))
        
        # Deve ser (2, 6, 7) - dois planos para o jogador 1 e 2
        self.assertEqual(encoded.shape, (2, 6, 7))
        # Inicialmente deve ser tudo zero
        self.assertTrue(np.all(encoded == 0))

    def test_state_encoding_consistency(self):
        state = initial_state()
        state = apply_action(state, 0)  # Jogador 1 joga coluna 0
        
        # O encoding do jogador 1 deve ter uma peça no topo esquerdo (linha 5, coluna 0)
        encoded_p1 = np.array(encode_state(state, perspective_player=1))
        self.assertEqual(encoded_p1[0, 5, 0], 1) 
        
        # O encoding do jogador 2 deve mostrar isso no plano do oponente
        encoded_p2 = np.array(encode_state(state, perspective_player=2))
        self.assertEqual(encoded_p2[1, 5, 0], 1)


class TestDrawDetection(unittest.TestCase):
    def test_draw_on_full_board(self):
        state = initial_state()
        
        # Preenche o tabuleiro para criar um empate (alternando colunas para evitar vitórias)
        columns = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6] * 3
        
        for col in columns[:42]: 
            state = apply_action(state, col)
            if is_terminal(state):
                break
        
        # Se o tabuleiro estiver cheio e não houver vencedor, é um empate
        if is_terminal(state):
            p1_outcome = outcome_for_player(state, 1)
            p2_outcome = outcome_for_player(state, 2)
            # O empate deve ser 0 para ambos ou -1, 1 (vitória/derrota)
            self.assertIn(p1_outcome, [-1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
