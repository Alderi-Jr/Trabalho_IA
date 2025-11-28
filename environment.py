import numpy as np
import gym
from gym import spaces
import pygame
import os


class ApocalypseEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Grid
        self.grid_size = 10
        self.max_steps = 500

        # Actions
        self.action_space = spaces.Discrete(4)
        self.action_arrows = ["↑", "→", "↓", "←"]

        # Observation (x, y, f0, f1, f2)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(5,), dtype=np.int32
        )

        # Start position (fixed)
        self.agent_start = np.array([0, 0])
        self.agent_pos = self.agent_start.copy()

        # Safe zone default (will be randomized)
        self.safe_zone = np.array([9, 9])

        # placeholders (will be replaced by generate_random_map)
        self.walls = []
        self.rocks = []
        self.initial_supplies = []
        self.supplies_flags = []
        self.zombies = []

        # Pygame / rendering
        self.screen = None
        self.clock = None
        self.font = None
        self.cell = 600 // self.grid_size

        # Sprites lazy-loaded after pygame.init()
        self.sprites_loaded = False
        self.img_floor = None
        self.img_wall = None
        self.img_rock = None
        self.img_supply = None
        self.img_zombie = None
        self.img_agent = None
        self.img_safe = None

        # HUD/training info (updated externally from main)
        self.training_episode = 0
        self.training_max = 0
        self.training_last_reward = 0.0
        self.training_best_reward = -1e9

        # runtime trackers
        self.last_action = None
        self.total_reward = 0.0
        self.step_count = 0
        self.steps = 0

        # create an initial random map
        self.generate_random_map()

    # ----------------------- Map randomization (FULL RANDOM, no overlaps) -----------------------
    def generate_random_map(self, seed=None,
                            pct_walls=0.15,
                            num_rocks=2,
                            num_supplies=4,
                            num_zombies=5):
        """
        Generate a fully-random map:
        - Walls: pct_walls fraction of total cells (rounded)
        - Rocks: num_rocks
        - Supplies: num_supplies
        - Zombies: num_zombies
        - Safe zone: 1 random cell
        Guarantees: no overlap between any elements or agent_start.
        Does NOT guarantee path connectivity (as requested).
        """
        rng = np.random.RandomState(seed)

        total_cells = self.grid_size * self.grid_size
        num_walls = int(total_cells * pct_walls)

        # set of forbidden coordinates (tuples)
        forbidden = set()
        forbidden.add((int(self.agent_start[0]), int(self.agent_start[1])))

        # helper to sample a free position and reserve it
        def sample_free():
            # pick random until free
            while True:
                x = int(rng.randint(0, self.grid_size))
                y = int(rng.randint(0, self.grid_size))
                tup = (x, y)
                if tup not in forbidden:
                    forbidden.add(tup)
                    return np.array([x, y])

        # reset lists
        self.walls = []
        self.rocks = []
        self.initial_supplies = []
        self.zombies = []

        # sample safe_zone first (so it's not used by others)
        self.safe_zone = sample_free()

        # generate walls
        for _ in range(num_walls):
            self.walls.append(sample_free())

        # generate rocks
        for _ in range(num_rocks):
            self.rocks.append(sample_free())

        # generate supplies
        for _ in range(num_supplies):
            self.initial_supplies.append(sample_free())

        # generate zombies
        for _ in range(num_zombies):
            self.zombies.append(sample_free())

        # ensure lists are numpy arrays (consistent with rest of code)
        self.walls = [np.array(p) for p in self.walls]
        self.rocks = [np.array(p) for p in self.rocks]
        self.initial_supplies = [np.array(p) for p in self.initial_supplies]
        self.zombies = [np.array(p) for p in self.zombies]

        # reset flags and agent state
        self.supplies_flags = [False] * len(self.initial_supplies)
        self.agent_pos = self.agent_start.copy()
        self.steps = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action = None

    # ----------------------- Sprites loading -----------------------
    def _load_sprites(self):
        if self.sprites_loaded:
            return
        asset_path = "assets"
        try:
            def load(name):
                img = pygame.image.load(os.path.join(asset_path, name)).convert_alpha()
                return pygame.transform.scale(img, (self.cell, self.cell))
            self.img_floor  = load("floor.png")
            self.img_wall   = load("wall.png")
            self.img_rock   = load("rock.png")
            self.img_supply = load("supply.png")
            self.img_zombie = load("zombie.png")
            self.img_agent  = load("agent.png")
            self.img_safe   = load("safe.png")
        except Exception as e:
            print("Aviso: não foi possível carregar sprites (usando retângulos).", e)
            self.img_floor = self.img_wall = self.img_rock = self.img_supply = self.img_zombie = self.img_agent = self.img_safe = None
        self.sprites_loaded = True

    # ----------------------- Gym API -----------------------
    def _get_state_tuple(self):
        flags = tuple(1 if f else 0 for f in self.supplies_flags)
        return (int(self.agent_pos[0]), int(self.agent_pos[1]), flags[0], flags[1], flags[2])

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            try:
                super().reset(seed=seed)
            except TypeError:
                pass

        self.agent_pos = self.agent_start.copy()
        self.steps = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.last_action = None
        self.supplies_flags = [False] * len(self.initial_supplies)
        return self._get_state_tuple()

    def step(self, action):
        self.last_action = action

        new_pos = self.agent_pos.copy()
        if action == 0:         # UP
            new_pos[1] -= 1
        elif action == 1:       # RIGHT
            new_pos[0] += 1
        elif action == 2:       # DOWN
            new_pos[1] += 1
        elif action == 3:       # LEFT
            new_pos[0] -= 1

        # boundaries
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            new_pos = self.agent_pos

        # collisions with walls/rocks
        if any(np.array_equal(new_pos, w) for w in self.walls):
            new_pos = self.agent_pos
        if any(np.array_equal(new_pos, r) for r in self.rocks):
            new_pos = self.agent_pos

        self.agent_pos = new_pos
        self.steps += 1
        self.step_count += 1

        reward = -0.2
        terminated = False
        truncated = False

        # zombie collision -> terminal
        if any(np.array_equal(self.agent_pos, z) for z in self.zombies):
            reward = -15.0
            terminated = True
            self.total_reward += reward
            return self._get_state_tuple(), reward, terminated, truncated, {}

        # collect supplies
        for i, pos in enumerate(self.initial_supplies):
            if not self.supplies_flags[i] and np.array_equal(self.agent_pos, pos):
                self.supplies_flags[i] = True
                reward += 30.0
                break

        # safe zone
        if np.array_equal(self.agent_pos, self.safe_zone):
            if all(self.supplies_flags):
                reward += 120.0
                terminated = True
            else:
                reward -= 20.0

        if self.steps >= self.max_steps:
            truncated = True

        self.total_reward += reward
        return self._get_state_tuple(), reward, terminated, truncated, {}

    # ----------------------- Render + HUD + key handling -----------------------
    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Apocalypse RL - Random Maps (full random)")
            # make window a bit wider to fit HUD
            self.screen = pygame.display.set_mode((900, 600))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 20)
            self._load_sprites()

        # input handling: return signals to main
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "EXIT"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "REPLAY"
                if event.key == pygame.K_SPACE:
                    return "STEP"
                if event.key == pygame.K_ESCAPE:
                    return "EXIT"
                if event.key == pygame.K_n:
                    return "NEW_MAP"

        # background
        self.screen.fill((40, 40, 40))

        # draw floor
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.img_floor:
                    self.screen.blit(self.img_floor, (x * self.cell, y * self.cell))
                else:
                    pygame.draw.rect(self.screen, (50, 50, 50), (x * self.cell, y * self.cell, self.cell, self.cell))

        # walls
        for p in self.walls:
            if self.img_wall:
                self.screen.blit(self.img_wall, (p[0] * self.cell, p[1] * self.cell))
            else:
                pygame.draw.rect(self.screen, (100, 100, 100), (p[0] * self.cell, p[1] * self.cell, self.cell, self.cell))

        # rocks
        for p in self.rocks:
            if self.img_rock:
                self.screen.blit(self.img_rock, (p[0] * self.cell, p[1] * self.cell))
            else:
                pygame.draw.rect(self.screen, (150, 120, 80), (p[0] * self.cell, p[1] * self.cell, self.cell, self.cell))

        # supplies
        for i, p in enumerate(self.initial_supplies):
            if not self.supplies_flags[i]:
                if self.img_supply:
                    self.screen.blit(self.img_supply, (p[0] * self.cell, p[1] * self.cell))
                else:
                    pygame.draw.rect(self.screen, (0, 255, 0), (p[0] * self.cell, p[1] * self.cell, self.cell, self.cell))

        # zombies
        for p in self.zombies:
            if self.img_zombie:
                self.screen.blit(self.img_zombie, (p[0] * self.cell, p[1] * self.cell))
            else:
                pygame.draw.rect(self.screen, (255, 0, 0), (p[0] * self.cell, p[1] * self.cell, self.cell, self.cell))

        # safe
        if self.img_safe:
            self.screen.blit(self.img_safe, (self.safe_zone[0] * self.cell, self.safe_zone[1] * self.cell))
        else:
            pygame.draw.rect(self.screen, (0, 120, 255), (self.safe_zone[0] * self.cell, self.safe_zone[1] * self.cell, self.cell, self.cell))

        # agent
        if self.img_agent:
            self.screen.blit(self.img_agent, (self.agent_pos[0] * self.cell, self.agent_pos[1] * self.cell))
        else:
            pygame.draw.rect(self.screen, (255, 255, 0), (self.agent_pos[0] * self.cell, self.agent_pos[1] * self.cell, self.cell, self.cell))

        # HUD panel (right)
        hud_x = 640
        self.screen.blit(self.font.render("HUD", True, (255,255,255)), (hud_x, 10))
        self.screen.blit(self.font.render(f"Passos: {self.step_count}", True, (255,255,255)), (hud_x, 40))
        self.screen.blit(self.font.render(f"Reward total: {round(self.total_reward,1)}", True, (255,255,255)), (hud_x, 70))

        # training info (if any)
        self.screen.blit(self.font.render(f"Ep: {self.training_episode}/{self.training_max}", True, (255,255,255)), (hud_x, 110))
        self.screen.blit(self.font.render(f"Última reward: {round(self.training_last_reward,1)}", True, (255,255,255)), (hud_x, 140))
        self.screen.blit(self.font.render(f"Melhor reward: {round(self.training_best_reward,1)}", True, (255,255,255)), (hud_x, 170))

        # supplies flags display (use simple ASCII to avoid font issues)
        # guard against lists shorter than 3 (shouldn't happen if num_supplies=4 but keep safe)
        flags_display = []
        for i in range(min(3, len(self.supplies_flags))):
            flags_display.append('[X]' if self.supplies_flags[i] else '[ ]')
        while len(flags_display) < 3:
            flags_display.append('[ ]')

        flags = f"{flags_display[0]}  {flags_display[1]}  {flags_display[2]}"
        self.screen.blit(self.font.render("Suprimentos:", True, (255,255,255)), (hud_x, 210))
        self.screen.blit(self.font.render(flags, True, (0,255,0)), (hud_x+10, 235))

        # action arrow
        if self.last_action is not None:
            self.screen.blit(self.font.render(f"Ação: {self.action_arrows[self.last_action]}", True, (255,255,0)), (hud_x, 270))

        # instructions
        self.screen.blit(self.font.render("R = replay melhor episódio", True, (200,200,200)), (hud_x-20, 520))
        self.screen.blit(self.font.render("N = novo mapa + treinar", True, (200,200,200)), (hud_x-20, 540))
        self.screen.blit(self.font.render("ESPAÇO = passo-a-passo", True, (200,200,200)), (hud_x-20, 560))
        self.screen.blit(self.font.render("ESC = sair", True, (200,200,200)), (hud_x-20, 580))

        pygame.display.flip()
        self.clock.tick(10)  # HUD updates quicker than replay for responsiveness
        return None

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass
