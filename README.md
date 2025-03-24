# GTMA
This repository provides the Python implementation for a paper submitted to IET Intelligent Transport Systems, titled 
## Game-Theoretic Multi-Agent Dynamic Traffic Assignment Using Congestion-Aware Adaptive Learning

### Description
Implement a dynamic traffic assignment model that can be rapidly tested in Python and SUMO environments, providing a clear demonstration of the route-choice process for multiple agents. By controlling time-varying parameters (such as learning rate and congestion coefficient), this model analyzes how vehicles choose routes under different adjustment strategies and examines how the system’s average travel time evolves.

### Dependencies
See requirements.txt

### Running the code
Python and SUMO:<br>
``
Python GTMA SUMO.py
``

Just python:<br> 
``
Python GTMA.py
``

### Baseline or benchmark
Four baseline frameworks are considered: two game-theoretic (GAME1, GAME2) and two external benchmarks (LTM, CTM). The game-theoretic frameworks differ in treating the dynamic route choice parameters γ(t) and ψ(t): GAME1 omits them entirely, while GAME2 employs fixed parameters. Meanwhile, the LTM implementation is sourced from (https://github.com/ZijianHu-polyu/DynamicUserEquilibrium), and the CTM implementation is obtained from (https://github.com/byq-luo/Dynamic-traffic-assignment-node-based-with-CTM).

## Traffic Network
This experiment is conducted on multiple networks of increasing complexity (Sioux Falls, Chicago, and Anaheim). The city road network data is sourced from (https://github.com/bstabler/TransportationNetworks). 
