
# A Benders Decomposition Method for Markov Decision Processes

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The source code and data in this repository are a snapshot of the software and data
that were used in the research reported on in the manuscript under review 
[A Benders Decomposition Method for Markov Decision Processes](https://www.researchgate.net/publication/353295872_A_Benders_Decomposition_Method_for_Markov_Decision_Processes) by Z. Liu et. al. The data generated for this study are included with the codes.

## Cite

To cite this repository, please cite the [manuscript](https://www.researchgate.net/publication/353295872_A_Benders_Decomposition_Method_for_Markov_Decision_Processes).

Below is the BibTex for citing the manuscript.

```
@article{Liu2021,
  title={A Benders Decomposition Method for Markov Decision Processes},
  author={Liu, Zeyu and Li, Xueping and Khojandi, Anahita},
  journal={Preprint},
  year={2021},
  url={https://www.researchgate.net/publication/353295872_A_Benders_Decomposition_Method_for_Markov_Decision_Processes}
}
```

## Description

The goal of this repository is to solve Markov decision processes (MDP) with the Benders decomposition. Our proposed algorithm, the MDP-Benders algorithm, decomposes the linear programming formulation of MDP into a master problem and multiple subproblems with respect to the states of MDP. Please refer to the manuscript for further details.

The data used in this study contains a randomly generated MDP problem and five other problems from the literature:
1. A queueing problem [[1]](#1);
2. A bandit machine problem [[2]](#2);
3. An inventory management problem [[3]](#3);
4. A machine maintenance problem [[4]](#4);
5. A data transmission problem [[5]](#5).


## Python Prerequisite

The following Python libraries are required to run the source codes:
1. `numpy`;
2. `scipy`;
3. `pickle`;
4. `gurobipy`;
5. `mdptoolbox`.

## Usage

### General MDP

Run `decomposition()` function in the `main.py` file in the `scripts/constrained_MDP` folder. Different problems can be set up using the provided variables `instance` and `params`. All six problems are available. The following describes the parameters for each problem:
1. `random`: `params = ( number of states, number of actions )`;
2. `queue`: `params = ( length of queue, number of service modes, arrival probability )`;
3. `bandit`: `params = ( number of arms, number of states for each arm )`;
4. `inventory`: `params = ( inventory capacity )`;
5. `replace`: `params = ( number of states of the machine, number of maintenance options )`;
6. `transmission` `params = ( number of channels, number of data packages, number of transmission modes )`.

### MDP with Monotone Optimal Policy

Run `decomposition_monotone()` function in the `main.py` file in the `scripts/constrained_MDP` folder. Different problems can be set up using the provided variables `instance` and `params`. The `queue`, `maintain` and `transmit` problems are available.

### Constrained MDP

Run the `main.py` file in the `scripts/constrained_MDP` folder. Different problems can be set up using the provided variables `instance` and `params`. All six problems are available.

## Support

For support in using this software, submit an
[issue](https://github.com/ILABUTK/Benders_Decomposition_for_MDP/issues/new).

## Reference
<a id="1">[1]</a> de Farias DP, Van Roy B (2003) The linear programming approach to approximate dynamic programming. Operations research 51(6):850-865.

<a id="2">[2]</a> Bertsimas D, Misic VV (2016) Decomposable markov decision processes: A fluid optimization approach. Operations Research 64(6):1537-1555.

<a id="3">[3]</a> Lee I, Epelman MA, Romeijn HE, Smith RL (2017) Simplex algorithm for countable-state discounted markov decision processes. Operations Research 65(4):1029-1042.

<a id="4">[4]</a> Puterman ML (2014) Markov decision processes: discrete stochastic dynamic programming (John Wiley & Sons).

<a id="5">[5]</a> Krishnamurthy V (2016) Partially observed Markov decision processes (Cambridge university press).