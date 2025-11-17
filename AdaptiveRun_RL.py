from panda3d.core import PointLight, CardMaker, loadPrcFileData, TextNode
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectButton, DirectFrame
import random
import numpy as np
import sys
import time

# ------------------------------------------------------
# WINDOW CONFIG - FULLSCREEN
# ------------------------------------------------------
# Must be called BEFORE ShowBase() is constructed
loadPrcFileData("", "fullscreen 1")
loadPrcFileData("", "win-size 1280 720")  # used if fullscreen is off

# ======================================================
# CONFIG
# ======================================================
LANES = [-1.5, 0, 1.5]   # 3 lanes

BASE_SCROLL_SPEED = 0.18
INIT_OBSTACLE_FREQ = 0.02
INIT_COIN_FREQ = 0.25

JUMP_FORCE = 0.35
GRAVITY = -0.02

RL_UPDATE_TIME = 1.0  # make RL update more frequently (every 1s)

# Collision sensitivity
COLLISION_X = 0.4
COLLISION_Y = 0.6
COLLISION_Z = 0.5


# ======================================================
# RL Difficulty Agent
# ======================================================
class RLAgent:
    def __init__(self):
        self.actions = 7
        self.w = np.zeros((self.actions, 5))  # 5 features in state
        self.last_state = None
        self.last_action = 0
        self.eps = 0.3  # 30% exploration

    def featurize(self, s):
        return np.array(s, dtype=float)

    def choose_action(self, state):
        # ε-greedy: sometimes explore
        if random.random() < self.eps:
            return random.randint(0, self.actions - 1)

        features = self.featurize(state)
        q_values = features @ self.w.T
        return int(np.argmax(q_values))

    def update(self, new_state, reward):
        if self.last_state is None:
            self.last_state = new_state
            return

        old_f = self.featurize(self.last_state)
        new_f = self.featurize(new_state)

        gamma = 0.95
        alpha = 0.01
        td_target = reward + gamma * np.max(new_f @ self.w.T)
        td_error = td_target - (old_f @ self.w[self.last_action])
        self.w[self.last_action] += alpha * td_error * old_f
        self.last_state = new_state


# ======================================================
# 3D Game
# ======================================================
class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.disableMouse()

        # Camera: slightly above & behind player, looking forward
        self.camera.setPos(0, -15, 7)
        self.camera.lookAt(0, 5, 4)

        # --------- GAME STATE FLAGS ----------
        self.game_over = False
        self.game_over_frame = None  # overlay UI

        # Game variables
        self.score = 0.0
        self.coins = 0
        self.start_time = time.time()
        self.agent = RLAgent()
        self.last_rl_update = time.time()

        # Difficulty variables
        self.difficulty = 1.0
        self.scroll_speed = BASE_SCROLL_SPEED
        self.coin_freq = INIT_COIN_FREQ
        self.obstacle_freq = INIT_OBSTACLE_FREQ

        # Warmup period (easy mode, RL off)
        self.rl_warmup = 10.0  # seconds

        # Lighting
        plight = PointLight("plight")
        plight_node = render.attachNewNode(plight)
        plight_node.setPos(0, -30, 20)
        render.setLight(plight_node)

        # Simple colored card for player
        cm_player = CardMaker("player_card")
        cm_player.setFrame(-0.5, 0.5, -0.5, 0.5)  # square
        self.player = render.attachNewNode(cm_player.generate())
        self.player.setBillboardPointEye()  # always face camera
        self.player.setColor(0.2, 0.4, 1.0, 1)  # blue
        self.player.setScale(0.8)
        self.player_lane = 1  # middle lane index
        self.player.setPos(LANES[self.player_lane], 0, 2)
        self.vz = 0.0  # vertical velocity for jump

        # Dynamic objects (coins + obstacles)
        self.objects = []  # list of NodePaths

        # HUD (Score + Coins) at top-right
        self.hud_score = OnscreenText(
            text="Score: 0",
            parent=base.a2dTopRight,
            pos=(-0.1, -0.1),
            align=TextNode.ARight,
            scale=0.07,
            fg=(1, 1, 1, 1),
            mayChange=True
        )
        self.hud_coins = OnscreenText(
            text="Coins: 0",
            parent=base.a2dTopRight,
            pos=(-0.1, -0.2),
            align=TextNode.ARight,
            scale=0.06,
            fg=(1, 1, 0.5, 1),
            mayChange=True
        )

        # Controls
        self.accept("escape", sys.exit)
        self.accept("arrow_left", self.move_left)
        self.accept("arrow_right", self.move_right)
        self.accept("space", self.jump)

        # Main update loop
        self.taskMgr.add(self.update, "updateGame")

        # Initial HUD update
        self.update_hud()

    # =======================================================
    # PLAYER MOVEMENT
    # =======================================================
    def move_left(self):
        if not self.game_over and self.player_lane > 0:
            self.player_lane -= 1

    def move_right(self):
        if not self.game_over and self.player_lane < 2:
            self.player_lane += 1

    def jump(self):
        if self.game_over:
            return
        # Jump only if on "ground" (z ~= 2)
        if self.player.getZ() <= 2.05:
            self.vz = JUMP_FORCE

    # =======================================================
    # HUD UPDATE
    # =======================================================
    def update_hud(self):
        self.hud_score.setText(f"Score: {int(self.score)}")
        self.hud_coins.setText(f"Coins: {self.coins}")

    # =======================================================
    # MAIN UPDATE LOOP
    # =======================================================
    def update(self, task):

        # If game is over, stop updating world (but keep UI alive)
        if self.game_over:
            return task.cont

        # Apply lane x-position
        self.player.setX(LANES[self.player_lane])

        # Jump physics
        self.vz += GRAVITY
        self.player.setZ(self.player.getZ() + self.vz)
        if self.player.getZ() < 2:
            self.player.setZ(2)
            self.vz = 0

        # Time since start
        elapsed = time.time() - self.start_time

        # Move all objects towards the player
        move_speed = self.scroll_speed * self.difficulty
        for obj in self.objects[:]:
            obj.setY(obj.getY() - move_speed)
            if obj.getY() < -10:
                obj.removeNode()
                self.objects.remove(obj)

        # --- Spawning Logic ---

        # During warmup: even fewer obstacles ⇒ super easy start
        if elapsed < self.rl_warmup:
            obstacle_freq = self.obstacle_freq * 0.3
        else:
            obstacle_freq = self.obstacle_freq

        # Spawn obstacles (with difficulty factor)
        if random.random() < obstacle_freq * self.difficulty:
            self.spawn_obstacle()

        # Spawn coins
        if random.random() < self.coin_freq:
            self.spawn_coin()

        # Handle collisions
        self.check_collisions()

        # RL difficulty update (only after warmup)
        self.update_rl()

        # Increase score over time
        self.score += self.difficulty * 0.5

        # Update HUD every frame
        self.update_hud()

        return task.cont

    # =======================================================
    # SPAWN OBJECTS
    # =======================================================
    def spawn_obstacle(self):
        spawn_y = 30

        # Avoid impossible patterns:
        # Don't allow more than 2 obstacles very close in Y (same "row")
        nearby_obstacles = []
        for obj in self.objects:
            if obj.getTag("type") == "obstacle" and abs(obj.getY() - spawn_y) < 4:
                nearby_obstacles.append(obj)

        # Count how many lanes already blocked in this "row"
        used_lanes = set()
        for o in nearby_obstacles:
            used_lanes.add(round(o.getX(), 1))

        if len(used_lanes) >= 2:
            # Already 2 lanes blocked here → skip spawning a 3rd
            return

        lane = random.choice(LANES)
        cm_obs = CardMaker("obs_card")
        cm_obs.setFrame(-0.5, 0.5, -0.5, 0.5)
        obj = render.attachNewNode(cm_obs.generate())
        obj.setBillboardPointEye()
        obj.setColor(1, 0.2, 0.2, 1)  # red = obstacle
        obj.setScale(0.9)
        obj.setPos(lane, spawn_y, 2)
        obj.setTag("type", "obstacle")
        self.objects.append(obj)

    def spawn_coin(self):
        lane = random.choice(LANES)
        cm_coin = CardMaker("coin_card")
        cm_coin.setFrame(-0.4, 0.4, -0.4, 0.4)
        obj = render.attachNewNode(cm_coin.generate())
        obj.setBillboardPointEye()
        obj.setColor(1, 1, 0, 1)  # yellow = coin
        obj.setScale(0.6)
        obj.setPos(lane, 30, 2.4)
        obj.setTag("type", "coin")
        self.objects.append(obj)

    # =======================================================
    # COLLISION CHECK
    # =======================================================
    def check_collisions(self):
        px, py, pz = self.player.getPos()

        for obj in self.objects[:]:
            ox, oy, oz = obj.getPos()

            # More forgiving collision box
            if (abs(px - ox) < COLLISION_X and
                abs(py - oy) < COLLISION_Y and
                abs(pz - oz) < COLLISION_Z):

                obj_type = obj.getTag("type")

                if obj_type == "coin":
                    self.score += 100
                    self.coins += 1

                    # (Optional) coin-based bump — still only affects difficulty, not speed
                    if self.coins % 50 == 0:
                        self.difficulty *= 1.2
                        if self.difficulty > 3.0:
                            self.difficulty = 3.0

                    obj.removeNode()
                    self.objects.remove(obj)

                elif obj_type == "obstacle":
                    self.on_game_over()
                    return

    # =======================================================
    # GAME OVER + RESTART
    # =======================================================
    def on_game_over(self):
        if self.game_over:
            return

        self.game_over = True

        # Create a semi-transparent overlay
        self.game_over_frame = DirectFrame(
            parent=self.aspect2d,
            frameColor=(0, 0, 0, 0.7),
            frameSize=(-1.3, 1.3, -1, 1)
        )

        # Game Over text
        OnscreenText(
            text="GAME OVER",
            parent=self.game_over_frame,
            pos=(0, 0.3),
            scale=0.12,
            fg=(1, 1, 1, 1),
            align=TextNode.ACenter
        )

        # Final score
        OnscreenText(
            text=f"Score: {int(self.score)}",
            parent=self.game_over_frame,
            pos=(0, 0.1),
            scale=0.08,
            fg=(1, 1, 0.8, 1),
            align=TextNode.ACenter
        )

        # Final coins
        OnscreenText(
            text=f"Coins: {self.coins}",
            parent=self.game_over_frame,
            pos=(0, -0.02),
            scale=0.07,
            fg=(1, 0.9, 0.4, 1),
            align=TextNode.ACenter
        )

        # Try Again button
        DirectButton(
            text="Try Again",
            parent=self.game_over_frame,
            scale=0.08,
            pos=(0, 0, -0.25),
            command=self.restart_game,
            text_fg=(0, 0, 0, 1),
            frameColor=(1, 1, 1, 1),
            pad=(0.3, 0.15)
        )

    def restart_game(self):
        # Destroy the game-over UI
        if self.game_over_frame is not None:
            self.game_over_frame.destroy()
            self.game_over_frame = None

        # Remove all dynamic objects
        for obj in self.objects:
            obj.removeNode()
        self.objects.clear()

        # Reset core variables
        self.score = 0.0
        self.coins = 0
        self.start_time = time.time()
        self.difficulty = 1.0
        self.scroll_speed = BASE_SCROLL_SPEED
        self.coin_freq = INIT_COIN_FREQ
        self.obstacle_freq = INIT_OBSTACLE_FREQ
        self.last_rl_update = time.time()

        # New RL agent (fresh start)
        self.agent = RLAgent()

        # Reset player
        self.player_lane = 1
        self.player.setPos(LANES[self.player_lane], 0, 2)
        self.vz = 0.0

        # Clear game over flag
        self.game_over = False

        # Update HUD
        self.update_hud()

    # =======================================================
    # RL DIFFICULTY UPDATE
    # =======================================================
    def update_rl(self):
        now = time.time()
        if now - self.last_rl_update < RL_UPDATE_TIME:
            return

        # Don't let RL change difficulty during warmup
        if now - self.start_time < self.rl_warmup:
            return

        # Build current state
        state = [
            self.difficulty,
            self.score / 10000.0,
            self.scroll_speed,
            self.obstacle_freq,
            self.coin_freq,
        ]

        # Choose action and remember it
        action = self.agent.choose_action(state)
        self.agent.last_action = action
        self.agent.last_state = state

        # -------- RL ACTION EFFECTS --------
        # 0 = slightly easier, 1–6 = variations of harder
        if action == 0:
            # ✅ Only reduce difficulty, DO NOT touch speed
            self.difficulty -= 0.05
        elif action == 1:
            self.scroll_speed += 0.02
        elif action == 2:
            self.obstacle_freq += 0.01
        elif action == 3:
            self.coin_freq -= 0.01
        elif action == 4:
            self.obstacle_freq += 0.015
        elif action == 5:
            self.coin_freq += 0.015
        elif action == 6:
            self.difficulty += 0.05

        # Clamp values to keep game easy/moderate
        self.coin_freq = max(0.1, min(self.coin_freq, 0.6))
        self.obstacle_freq = max(0.01, min(self.obstacle_freq, 0.18))
        self.difficulty = max(0.8, min(self.difficulty, 2.8))
        # ✅ Speed never below BASE_SCROLL_SPEED
        self.scroll_speed = max(BASE_SCROLL_SPEED, min(self.scroll_speed, 0.35))

        # Reward prefers moderate difficulty band
        reward = 1.0
        if self.difficulty > 2.4 or self.difficulty < 0.9:
            reward = -1.0

        self.agent.update(state, reward)
        self.last_rl_update = now

        # Print to console so you can show RL working
        print(
            f"[RL] action={action}, diff={self.difficulty:.2f}, "
            f"speed={self.scroll_speed:.2f}, obs_freq={self.obstacle_freq:.3f}, coin_freq={self.coin_freq:.3f}"
        )


# ======================================================
# RUN GAME
# ======================================================
if __name__ == "__main__":
    game = Game()
    game.run()
