import unittest
from connect4_rl.agents.baselines import HeuristicAgent, MinimaxAgent, RandomAgent, StrongHeuristicAgent, WeakHeuristicAgent
from connect4_rl.envs.connect_four import (
    initial_state,
    apply_action,
    is_terminal,
    legal_actions,
)


class TestRandomAgent(unittest.TestCase):

    def test_selects_legal_action(self):
        agent = RandomAgent(seed=42)
        state = initial_state()
        
        for _ in range(50):
            legal = legal_actions(state)
            action = agent.select_action(state, legal)
            self.assertIn(action, legal, f"Agent selected illegal action {action} from {legal}")
            
            state = apply_action(state, action)
            if is_terminal(state):
                break

    def test_deterministic_with_seed(self):
        state1 = initial_state()
        state2 = initial_state()
        
        agent1 = RandomAgent(seed=12345)
        agent2 = RandomAgent(seed=12345)
        
        actions1 = []
        actions2 = []
        
        for _ in range(20):
            legal1 = legal_actions(state1)
            legal2 = legal_actions(state2)
            
            action1 = agent1.select_action(state1, legal1)
            action2 = agent2.select_action(state2, legal2)
            
            actions1.append(action1)
            actions2.append(action2)
            
            state1 = apply_action(state1, action1)
            state2 = apply_action(state2, action2)
            
            if is_terminal(state1) or is_terminal(state2):
                break
        
        self.assertEqual(actions1, actions2, "Same seed should produce same actions")

    def test_different_seed_different_actions(self):
        state = initial_state()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)
        
        actions1 = []
        actions2 = []
        
        for _ in range(30):
            legal = legal_actions(state)
            action1 = agent1.select_action(state, legal)
            action2 = agent2.select_action(state, legal)
            actions1.append(action1)
            actions2.append(action2)
            
            state = apply_action(state, action1)
            if is_terminal(state):
                break
        
        self.assertNotEqual(actions1, actions2, "Different seeds should likely produce different actions")


class TestHeuristicAgent(unittest.TestCase):

    def test_selects_legal_action(self):
        agent = HeuristicAgent(seed=42)
        state = initial_state()
        
        for _ in range(50):
            legal = legal_actions(state)
            action = agent.select_action(state, legal)
            self.assertIn(action, legal, f"Agent selected illegal action {action} from {legal}")
            
            state = apply_action(state, action)
            if is_terminal(state):
                break

    def test_blocks_opponent_win(self):

        agent = HeuristicAgent(seed=42)
        
        state = initial_state()
        
        state = apply_action(state, 0) 
        state = apply_action(state, 3) 
        state = apply_action(state, 1) 
        state = apply_action(state, 4) 
        state = apply_action(state, 2) 
        
        legal = legal_actions(state)
        if 3 in legal:
            action = agent.select_action(state, legal)


class TestAgentMatchup(unittest.TestCase):

    def test_heuristic_beats_random(self):
        from connect4_rl.envs.connect_four import outcome_for_player
        
        heuristic = HeuristicAgent(seed=42)
        random_agent = RandomAgent(seed=99)
        
        heuristic_wins = 0
        games = 10
        
        for game in range(games):
            state = initial_state()
            
            while not is_terminal(state):
                legal = legal_actions(state)
                
                if state.current_player == 1:
                    action = heuristic.select_action(state, legal)
                else:
                    action = random_agent.select_action(state, legal)
                
                state = apply_action(state, action)
            
            outcome = outcome_for_player(state, 1)
            if outcome == 1.0:
                heuristic_wins += 1
        
        self.assertGreaterEqual(heuristic_wins, games // 2, 
                               f"Heuristic won {heuristic_wins}/{games}, expected at least {games//2}")

    def test_strong_heuristic_beats_weak(self):
        from connect4_rl.envs.connect_four import outcome_for_player

        strong = StrongHeuristicAgent(seed=42)
        weak = WeakHeuristicAgent(seed=99)

        strong_wins = 0
        games = 10
        for _game in range(games):
            state = initial_state()
            while not is_terminal(state):
                legal = legal_actions(state)
                if state.current_player == 1:
                    action = strong.select_action(state, legal)
                else:
                    action = weak.select_action(state, legal)
                state = apply_action(state, action)
            if outcome_for_player(state, 1) == 1.0:
                strong_wins += 1

        self.assertGreaterEqual(strong_wins, games // 2)

    def test_minimax_selects_legal_action(self):
        agent = MinimaxAgent(depth=2, seed=42)
        state = initial_state()
        for _ in range(20):
            legal = legal_actions(state)
            action = agent.select_action(state, legal)
            self.assertIn(action, legal)
            state = apply_action(state, action)
            if is_terminal(state):
                break


if __name__ == "__main__":
    unittest.main()
