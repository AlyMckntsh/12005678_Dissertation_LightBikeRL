# 12005678_Dissertation_LightBikeRL

## Reinforcement Learning in a Competitive Light-Bike Game

This repository contains the Unity project developed for my BSc (Hons) Computing dissertation. The project investigates how a Proximal Policy Optimisation (PPO) reinforcement learning agent performs in a competitive light-bike environment built using Unity and Unity ML-Agents.

The environment is inspired by light-cycle style games, where two bikes move around a bounded grid while leaving permanent trails behind them. A bike loses if it collides with a wall, a trail, or the opposing bike. The project uses this environment to explore reinforcement learning, reward design, training stability, and post-training evaluation.

## Project Overview

The final system includes:

- A custom Unity light-bike game environment
- Grid-based bike movement and trail generation
- Shared-step gameplay so both bikes move fairly at the same time
- A PPO-controlled learner agent using Unity ML-Agents
- A fixed heuristic opponent
- Two reward conditions: `SurvivalOnly` and `ArenaControl`
- Training metric recording for TensorBoard
- Evaluation mode for testing trained models
- CSV export of evaluation results

## Research Aim

The aim of the dissertation was to investigate how a PPO reinforcement learning agent performs in a competitive light-bike environment under different fixed reward conditions.

The project compares whether a survival-focused reward design and a centre-control reward design lead to different learning outcomes.

## Technologies Used

- Unity
- C#
- Unity ML-Agents
- PPO
- TensorBoard
- ONNX model inference
- GitHub for version control

## Repository Contents

The main Unity project files are included in this repository. Key folders include:

```text
Assets/
Packages/
ProjectSettings/
```

The project may also include supporting files such as evaluation outputs, trained models, or documentation where relevant.

Unity-generated folders such as `Library`, `Temp`, `Obj`, `Logs`, and build output folders are intentionally excluded from the repository.

## Running the Project

To open the project:

1. Clone or download this repository.
2. Open the folder in Unity.
3. Allow Unity to regenerate the required project files.
4. Open the main scene used for training or evaluation.
5. Run the project in the Unity Editor.

The project is designed around Unity ML-Agents, so training requires the ML-Agents Python environment to be installed separately.

## Training and Evaluation

Training was carried out using Unity ML-Agents with PPO. The learner agent was trained against a fixed heuristic opponent under two reward conditions:

- `SurvivalOnly`
- `ArenaControl`

After training, models were evaluated separately against the same heuristic opponent. Evaluation results were recorded using win rate, wins, draws, losses, average reward, and average episode length.

## Author

Alasdair Mackintosh  
Student Number: 12005678  
BSc (Hons) Computing Software  
University of the Highlands and Islands

## Academic Use

This project was created as part of a dissertation submission. It is intended to demonstrate the implementation, training, and evaluation workflow used in the study.
