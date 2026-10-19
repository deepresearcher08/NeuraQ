# NeuraQ barren-plateau benchmark leaderboard

Entries are ranked by resilience to barren plateaus: a local-cost slope closer to 0 is better, a smaller global-cost slope magnitude is better, and so is a higher success rate under an equal budget.

| name | cost | init | n range | depth M | local slope (log Var per qubit) | global slope (log Var per qubit) | notes |
|---|---|---|---|---|---|---|---|
| baseline-hea-singleaxis | local and global | random | 4-10 | O(log n) | -0.446 (E2a) | -1.275 (E1) | committed baseline |
