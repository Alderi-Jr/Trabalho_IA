import pickle
import numpy as np
import pygame
from environment import ApocalypseEnvironment
from qlearning import QLearning


# ------------------------------- TREINAMENTO -------------------------------

def train_with_hud(env, qlearn, episodes=3000):
    """
    Treina usando qlearn.train(), atualizando o HUD.
    Espera que qlearn.train retorne:
      q_table, best_reward, ctrl, best_actions, episodes_run
    """

    def callback(ep, eps, ep_reward, best_reward):
        env.training_episode = ep
        env.training_max = episodes
        env.training_last_reward = ep_reward
        env.training_best_reward = best_reward

        sig = env.render()

        if sig == "EXIT":
            return "STOP"
        if sig == "NEW_MAP":
            return "NEW_MAP"
        return None

    q_table, best_reward, ctrl, best_actions, episodes_run = qlearn.train(
        num_episodes=episodes,
        callback=callback
    )

    return q_table, best_reward, ctrl, best_actions, episodes_run


# ------------------------------- REPLAY -------------------------------

def run_best_path(env, best_actions, fps=2, max_steps=5000):
    """
    Reproduce exactly the best_actions sequence.
    If best_actions is empty, falls back to greedy argmax (limited steps).
    """

    # If no best_actions -> fallback
    if not best_actions:
        # fallback greedy (not recommended)
        state = env.reset()
        terminated = False
        truncated = False
        steps = 0
        while not terminated and not truncated and steps < max_steps:
            idx = (int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]))
            action = int(np.argmax(q_learning.q_table[idx]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state

            env.clock.tick(fps)
            sig = env.render()
            if sig == "EXIT":
                return "EXIT"
            if sig == "NEW_MAP":
                return "NEW_MAP"
            if sig == "REPLAY":
                return "REPLAY"
            if sig == "STEP":
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                waiting = False
                            if event.key == pygame.K_r:
                                return "REPLAY"
                            if event.key == pygame.K_n:
                                return "NEW_MAP"
                            if event.key == pygame.K_ESCAPE:
                                return "EXIT"
                    env.clock.tick(10)
            steps += 1

        # after fallback, wait for user
        while True:
            env.clock.tick(5)
            sig = env.render()
            if sig == "REPLAY":
                return "REPLAY"
            if sig == "NEW_MAP":
                return "NEW_MAP"
            if sig == "EXIT":
                return "EXIT"

    # --- Play recorded best_actions exactly ---
    env.reset()
    terminated = False
    truncated = False

    for action in best_actions:
        next_state, reward, terminated, truncated, _ = env.step(action)

        env.clock.tick(fps)
        sig = env.render()

        if sig == "EXIT":
            return "EXIT"
        if sig == "NEW_MAP":
            return "NEW_MAP"
        if sig == "REPLAY":
            return "REPLAY"
        if sig == "STEP":
            # step-by-step mode
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            waiting = False
                        if event.key == pygame.K_r:
                            return "REPLAY"
                        if event.key == pygame.K_n:
                            return "NEW_MAP"
                        if event.key == pygame.K_ESCAPE:
                            return "EXIT"
                env.clock.tick(10)

        if terminated or truncated:
            break

    # finished: wait for user command
    while True:
        env.clock.tick(5)
        sig = env.render()
        if sig == "REPLAY":
            return "REPLAY"
        if sig == "NEW_MAP":
            return "NEW_MAP"
        if sig == "EXIT":
            return "EXIT"


# ------------------------------- LOOP PRINCIPAL -------------------------------

def main_loop():

    global q_learning  # used by fallback greedy in run_best_path
    env = ApocalypseEnvironment()
    env.generate_random_map()

    q_learning = QLearning(
        env=env,
        grid_size=env.grid_size,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.93,
        exploration_rate=0.25
    )

    # initial HUD show
    env.training_episode = 0
    env.training_max = 3000
    env.training_last_reward = 0.0
    env.training_best_reward = -1e9
    env.render()

    while True:

        print("Starting training (adaptive)...")
        q_table, best_reward, ctrl, best_actions, episodes_run = train_with_hud(
            env, q_learning, episodes=3000
        )

        if ctrl == "STOP":
            env.close()
            return

        if ctrl == "NEW_MAP":
            # new map requested while training
            env.generate_random_map()
            q_learning.q_table = np.zeros_like(q_learning.q_table)
            continue

        print(f"Training finished! Best reward: {best_reward:.2f} (episodes run: {episodes_run})")

        # Save Q-table and best_actions
        try:
            with open("q_table_stateflags.pkl", "wb") as f:
                pickle.dump(q_table, f)
        except Exception:
            pass
        try:
            with open("best_actions.pkl", "wb") as f:
                pickle.dump(best_actions, f)
        except Exception:
            pass

        # Replay loop (R -> replay best_actions only; N -> new map + retrain)
        while True:
            result = run_best_path(env, best_actions, fps=2)

            if result == "EXIT":
                env.close()
                return

            if result == "NEW_MAP":
                env.generate_random_map()
                q_learning.q_table = np.zeros_like(q_learning.q_table)
                break  # back to training

            if result == "REPLAY":
                # just repeat replay
                continue


if __name__ == "__main__":
    main_loop()
