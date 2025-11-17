AdaptiveRun – An RL-Based Endless Runner Game
📌 Problem Statement

Traditional endless-runner games rely on fixed or manually tuned difficulty levels, which often makes gameplay either too easy or too hard. They do not adapt to the player's skill in real time. This project solves that limitation by creating a 3D endless-runner where the game difficulty automatically adjusts using a lightweight Reinforcement Learning (RL) agent. The goal is to deliver a smoother, more balanced gameplay experience that becomes progressively challenging based on player performance.

🧠 Approach

The game uses a simple Q-learning style RL agent that periodically observes the game state and adjusts difficulty parameters. The agent receives a compact state representation containing the current difficulty value, scaled player score, scroll speed, and the spawn frequencies of obstacles and coins. Based on this state, the RL model selects one of seven actions, such as increasing or decreasing speed, modifying spawn frequencies, or adjusting the overall difficulty factor. The reward function encourages the agent to maintain a balanced difficulty range, penalizing extreme or unfair difficulty spikes. Over time, this allows the game to “self-tune” difficulty dynamically during gameplay.

🎮 Results

The RL agent successfully adapts game difficulty while maintaining smooth playability. Players experience a more personalized difficulty curve, where speed, obstacle density, and coin frequency adjust based on their skill. The game noticeably shifts difficulty patterns during a run, but never becomes impossible thanks to safe clamping. Overall, AdaptiveRun demonstrates how even a simple RL agent can create engaging, responsive gameplay without pre-programmed difficulty stages.
