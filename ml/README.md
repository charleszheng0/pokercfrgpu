# Future value network

This directory defines a PyTorch interface only. The network is **untrained** and
is not used to report solver quality.

The intended flow is: encode a game/public-belief state, predict continuation
values, then call that evaluator at a depth limit during CFR re-solving. The C++
CPU solver already accepts a `LeafEvaluator` callback as the integration seam;
the current Kuhn implementation still traverses every branch to a terminal state.

Before enabling this path, generate solver-supervised training data, define a
held-out error metric, calibrate values by information state, and verify that
depth-limited strategies remain low-exploitability.
