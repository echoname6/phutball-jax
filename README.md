# Phutball JAX

AlphaZero-style training infrastructure for [Philosopher's Football (Phutball)](https://en.wikipedia.org/wiki/Phutball) using JAX.

## Overview

Self-play reinforcement learning pipeline for training neural network agents to play Phutball, built on JAX for hardware-accelerated training on GPU/TPU.

- MCTS-based self-play with batched inference
- Policy-value network training
- Distributed training support
- Curriculum learning and round-robin evaluation
- Supports additional games (Gomoku, Halma)

## Acknowledgments

This project was built with substantial contributions from AI coding assistants. Claude Opus 4.6 and Gemini Flash 2.5 Preview were instrumental throughout development, with Opus 4.6 driving much of the final sprint. Models from Anthropic, Google, OpenAI, and DeepSeek all served their purposes at various stages of the project.
