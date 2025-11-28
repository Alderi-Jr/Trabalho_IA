import numpy as np


class QLearning:
    def __init__(self, env, grid_size, num_actions,
                 learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=0.3):

        self.env = env
        self.grid_size = grid_size
        self.num_actions = num_actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Q-table com flags binárias (f0,f1,f2)
        self.q_table = np.zeros((grid_size, grid_size, 2, 2, 2, num_actions))


    # ------------------------------------------------------------------
    #   TREINAMENTO COM CAPTURA DO MELHOR CAMINHO REAL
    # ------------------------------------------------------------------

    def train(self, num_episodes=3000, callback=None):

        best_reward = -1e9
        best_qtable = None
        best_actions = []     # <- sequência do melhor episódio REAL

        reward_goal = 205
        patience = 150
        wait = 0

        for ep in range(1, num_episodes + 1):

            state = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0

            episode_actions = []   # <- guarda as ações deste episódio

            # decaimento epsilon
            self.exploration_rate = max(0.02, self.exploration_rate * 0.995)

            while not terminated and not truncated:

                s = tuple(int(x) for x in state)

                # ação (exploração vs exploração)
                if np.random.random() < self.exploration_rate:
                    action = np.random.randint(self.num_actions)
                else:
                    action = int(np.argmax(self.q_table[s]))

                # salva ação real
                episode_actions.append(action)

                # executa
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                s2 = tuple(int(x) for x in next_state)

                # Q-learning
                target = reward + self.discount_factor * np.max(self.q_table[s2])
                self.q_table[s][action] += self.learning_rate * (target - self.q_table[s][action])

                state = next_state
                total_reward += reward

            # ---------------- MELHOR EPISÓDIO REAL ----------------
            if total_reward > best_reward:
                best_reward = total_reward
                best_qtable = self.q_table.copy()
                best_actions = episode_actions[:]       # <- salva caminho REAL
                wait = 0
            else:
                wait += 1

            # ---------------- Callback do HUD ----------------
            if callback is not None:
                ctrl = callback(ep, self.exploration_rate, total_reward, best_reward)
                if ctrl == "STOP":
                    return best_qtable, best_reward, "STOP", best_actions, ep
                if ctrl == "NEW_MAP":
                    return best_qtable, best_reward, "NEW_MAP", best_actions, ep

            # ---------------- Critérios de parada ----------------
            if best_reward >= reward_goal:
                return best_qtable, best_reward, None, best_actions, ep

            if wait >= patience:
                return best_qtable, best_reward, None, best_actions, ep

        return best_qtable, best_reward, None, best_actions, num_episodes
