## Monte carlo simulation for basic multi-armed bandit approximations with graduate school expectation storyline.

Included methods are:

 - Max reward (for comparison)
 - Explore only
 - Exploit only
 - $\epsilon$-Greedy

Regret is also calculated.

Example result with 1000 MC iterations:
```
$ python run_simulation.py 

MAX_REWARD: 100%|██████████████████████████████████████| 1000/1000 [00:08<00:00, 122.90it/s]
Expected max reward: 8815

EXPLORE ONLY: 100%|█████████████████████████████████████| 1000/1000 [00:10<00:00, 95.89it/s]
Expected total reward: 4629
Expected total regret: 4181

EXPLOIT ONLY: 100%|████████████████████████████████████| 1000/1000 [00:08<00:00, 123.97it/s]
Expected total reward: 3268
Expected total regret: 5546

EPSILON GREEDY(0.1): 100%|██████████████████████████████| 1000/1000 [00:12<00:00, 81.07it/s]
Expected total reward: 6348
Expected total regret: 2469
```