AdaptiveRun – An RL-Based Endless Runner Game
📌 Problem Statement

Traditional endless-runner games use fixed or manually tuned difficulty levels. This makes the game either too easy or too hard, and does not adapt to the player's skill.
The goal of this project is to build a 3D endless-runner where the game difficulty automatically adjusts in real-time using Reinforcement Learning (RL).

🧠 Approach

The system uses a lightweight Q-learning style RL agent that observes the game state and periodically adjusts difficulty parameters.

State Features

Current difficulty value

Player score (scaled)

Scroll speed

Obstacle spawn frequency

Coin spawn frequency

Actions
The RL agent selects from 7 possible difficulty adjustments, such as:

Increase/decrease scroll speed

Increase/decrease obstacle frequency

Increase/decrease coin frequency

Adjust overall difficulty factor

Reward Function
The agent receives:

Positive reward for staying in a moderate difficulty band

Negative reward for making the game too easy or too hard

Game Engine

Built using Panda3D

Simple 3-lane runner

Player can move left/right and jump

Coins increase score

Obstacles end the game

On “Game Over,” a UI screen shows:

Final score

Coins collected

Try Again button

🎮 Key Features

Adaptive difficulty using RL

Dynamic spawning of coins and obstacles

Smooth 3D rendering using Panda3D

Full-screen gameplay experience

Restartable game loop with clean UI

Simple physics (jump + gravity)

📊 Results

The RL agent successfully learns to keep the game within a balanced difficulty range.

The difficulty becomes dynamic, not static—making the gameplay feel smoother and more responsive.

The agent visibly changes:

speed

obstacle frequency

coin frequency

…which creates different difficulty patterns during the run.

Players experience a progressively challenging yet fair gameplay curve.
