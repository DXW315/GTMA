# GTMA
This repository contains the Python implementation for the paper:
“Game-Theoretic Multi-Agent Dynamic Traffic Assignment Using Congestion-Aware Adaptive Learning.” This paper is submitted to the” which has been submitted to IET Intelligent Transport Systems.

Description：
Implement a dynamic traffic assignment model that can be rapidly tested in Python and SUMO environments, providing a clear demonstration of the route-choice process for multiple agents. By controlling time-varying parameters (such as learning rate and congestion coefficient), this model analyzes how vehicles choose routes under different adjustment strategies and examines how the system’s average travel time evolves.

Dependencies：
See requirements.txt

Running the code：

Python and SUMO: 
Python GTMA SUMO.py

Just python: 
Python GTMA.py

Plot:
Python Gap.py

Baseline or benchmark:
In total, four baseline frameworks are considered: two game-theoretic (GAME1, GAME2) and two external benchmarks (LTM, CTM). The game-theoretic frameworks differ in their treatment of the dynamic route choice parameters γ(t) and ψ(t): GAME1 omits them entirely, while GAME2 employs fixed parameters. Meanwhile, the LTM implementation is sourced from https://github.com/ZijianHu-polyu/DynamicUserEquilibrium, and the CTM implementation is obtained from https://github.com/byq-luo/Dynamic-traffic-assignment-node-based-with-CTM.

