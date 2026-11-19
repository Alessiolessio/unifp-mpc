# UniFP-MPC — Research Codebase

## Goal
Build a fast policy-in-the-loop sampling MPC for the pretrained UniFP-Spot
whole-body controller. Use SUMO's ReLIC-based MPC and Judo as references, but
design the controller around UniFP's own commands, history, and actuators.
Target MuJoCo execution and native C++ CPU rollouts; verify this architecture
against the existing IsaacLab/PhysX baseline rather than assuming it is best.
The actual tasks are pushing a box to a target location and reorienting a traffic
cone; prioritize box pushing. Wall contact is only an optional diagnostic.

## Sources of Truth
- Inspect the actual repositories and checkpoint before deciding how to integrate.
  UniFP-Spot (/home/nexus/unifp-spot) defines policy behavior; the existing UniFP-MPC defines the current
  planner; SUMO/Judo (/home/nexus/judo) (/home/nexus/sumo) are infrastructure and design references, not specifications.
- Treat sibling repositories as read-only unless explicitly told otherwise.
  Make implementation changes only in this repository.
- Prefer adapting working code over rewriting it or copying entire frameworks.
  Check licensing and attribution before reusing third-party source/assets.
- Separate verified source facts from assumptions, estimates, and hypotheses.
  Challenge proposed architecture choices when a simpler or faster path exists.

## Dependencies and Environment
- This repository starts without a virtual environment. Audit the actual Python,
  MuJoCo, ONNX Runtime, Judo, SUMO, and UniFP-Spot versions/build requirements;
  establish a minimal, reproducible local environment and native C++ build.
- Resolve incompatible dependencies deliberately; do not blindly combine existing
  venvs or change reference repositories. Document required setup and versions.

## Coding Rules
- Write minimal, clear research code: no code golf, unnecessary abstractions,
  compatibility layers, feature flags, redundant modules, or defensive scaffolding.
- Prefer short, direct implementations and descriptive names. Avoid comments
  except for non-obvious mathematics, frames, units, timing, and state semantics.
- Preserve determinism and existing behavior. Keep changes focused and remove
  temporary experimental machinery when finished.
- Python: 4-space indentation, approximately 100-character line limit; avoid
  one-line loops/conditionals. C++: follow the local project's established style.
- Do not silently change pretrained weights, training code, or baseline physics.
  Do not start training, expensive sweeps, or long benchmarks without a request.

## Policy and Rollout Correctness
- Preserve the verified UniFP observation order/scales, 32-frame history,
  encoder+actor inference, command frames, joint mapping, action processing,
  actuator behavior, and policy/physics timing. Re-check dimensions in source.
- Each candidate must branch from the same *executed* state and own its
  physics state, observation history, previous action, gait phase, and actuator
  delays/state. Never feed a hypothetical rollout state into the live controller.
- Match MuJoCo and IsaacLab robot dynamics as closely as practical; measure
  mismatches in joint response, locomotion, and contact before trusting MPC.
- Force *commands* and learned CSE outputs are not measured contact forces.
  Use correctly transformed simulator contact forces for force-related costs,
  and test position-only / matched-effective-position baselines.
- Reuse the current MPC formulation when useful, but **no task cost function is
  validated yet**. Design and test task-specific costs instead of assuming SUMO's
  ReLIC-oriented rewards transfer. Do not force ReLIC's interface onto UniFP.

## Performance and Validation
- Prioritize end-to-end MPC latency and task performance, not isolated policy FPS.
  Profile before optimizing; compare like-for-like workloads and contact fidelity.
- Prefer native rollout loops without per-step Python callbacks, repeated
  allocation, or CPU/GPU synchronization. Benchmark CPU threading and ONNX
  settings rather than assuming more threads are better.
- Use only small, targeted smoke/parity checks when needed. Put temporary
  probes, benchmarks, and test code in root `tests/` (disposable).
- Report what changed, what was actually checked, measured timings if any, and
  remaining uncertainty. Never describe unrun tests or simulated results as facts.

## Workflow / Git
- For meaningful tasks: inspect relevant code, choose the simplest sound plan,
  implement in focused increments, and briefly explain consequential tradeoffs.
- Do not stage, commit, push, reset, or discard changes unless explicitly asked.
