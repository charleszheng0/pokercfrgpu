# Research notes

These sources guide future implementation. In milestone 1, only the need for explicit information states, finite action abstraction, and independent CPU reference informed the architecture; the published poker systems are **not reproduced**.

| Source | Core contribution and relevant location | Used here | Departure / pending work |
|---|---|---|---|
| [Zinkevich et al., 2007](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf) | Counterfactual regret, reach probabilities, regret matching, and average strategy in the CFR algorithm. | Historical `small_games/kuhn.py` and `small_games/compare_cfr_nfsp.py` contain tabular regret matching. | New Hold'em CFR is pending; historical convergence and update schedule require validation. |
| [Heinrich and Silver, 2016](https://arxiv.org/abs/1603.01121) | NFSP combines an approximate best response with supervised learning of the average policy; anticipatory mixing and reservoir sampling preserve historical behavior. | Historical `small_games/train_nfsp.py` has Q/target and policy networks, replay and reservoir buffers. | No new training or reproducibility claim yet; exact Leduc exploitability must be measured. |
| [Moravčík et al., 2017](https://arxiv.org/abs/1701.01724) | Continual re-solving, depth-limited lookahead, and learned hand-dependent counterfactual values. | Motivates a future range-vector value model. | No depth-limited solver or opponent counterfactual-value constraints yet; no DeepStack safety guarantee. |
| [Brown and Sandholm, Libratus](https://noambrown.com/papers/17-Science-Superhuman.pdf) | Blueprint abstraction, nested subgame solving, and self-improvement. | Motivates separating offline blueprint from online local solve. | None of these three modules is implemented in the new Hold'em path. |
| [Brown et al., ReBeL, 2020](https://papers.nips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf) | Public belief state representation joins public actions with distributions over private states for search and learning. | Motivates future explicit range representation. | No learned belief-state search is implemented. |
| [Brown and Sandholm, Safe and Nested Subgame Solving, 2017](https://arxiv.org/abs/1705.02955) | Opponent value constraints in safe subgame solving. | Flags a safety condition for future continual re-solving. | No safe subgame solver exists yet. |

At a future CFR information set `I`, the intended regret increment for action `a` is `π₋ᵢ(I) π_c(I) [vᵢ(I,a) − vᵢ(I)]`; average strategy weight uses the acting player's reach `πᵢ(I)`. The CPU implementation must test chance weighting and the information-set grouping explicitly before interpreting its strategies.

GPU-CFR literature is still under review. Existing repository CUDA snippets are preserved but are not cited as research evidence or treated as validated accelerators. Candidate original work includes [Parallelizing Counterfactual Regret Minimization](https://arxiv.org/abs/2605.14277), which needs detailed review before any algorithmic claim or implementation choice.
