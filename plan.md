1. Game Environment
- Create some sort of representation for the 6x7 board, move legality, and win conditions.
- Represent board state for NN somehow

2. Design NN
- Probably use small transformer or CNN model for spatial patterns
- Input should be board state
- Outputs should be:
    - Policy Head: Provide probabilities for best move to guide search process
    - Value Head: Estimate the expected outcome (W, D, L) from a given state
- Outputs are used in MCTS to bias exploration

3. Implement MCTS
- Traverse tree by selecting moves that maximize balance between exploration and exploitation
- Use NN value head to evaluate position.
- Backprop: Update the node in tree with eval to improve move selection

4. Self-Play and Data Generation
5. Train NN
6. Eval and Fine-tuning
