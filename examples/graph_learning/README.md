#### Graph Neural Network

- Graph Neural Network for robot design value prediction:
  - input: graph representation of a robot
  - output: predicted reward
  - varying data distribution is a critical problem:
    - Exp1: random shuffling dataset and take the first 2/3 as training data and the second 1/3 as testing data:
      - training MSE error: 0.32
      - testing MSE error: 0.34
    - Exp2: take the first 2/3 as training data and the second 1/3 as testing data without shuffling, and then shuffle the training data only:
      - training MSE error: 0.15
      - testing MSE error: 4.0
  - TODO: use this GNN to help predict for each searched design and avoid running the MPC for those designs with low prediction reward.
- TODO: Graph Neural Network for searching heuristic function: need to handle nonterminals.