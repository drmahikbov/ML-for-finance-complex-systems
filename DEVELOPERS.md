# Developer guide

Architecture notes and the recipe for adding new optimal-control problems.
For setup and the reference simulation, see [README.md](README.md).

---

Implicit-network policy training for finite-horizon optimal control problems
(JFB / FullAD), built around a small set of single-purpose abstractions:

| Layer            | Path                                  | Owns                                              |
| ---------------- | ------------------------------------- | ------------------------------------------------- |
| Math contract    | `jfb-for-implicit-oc/core/ImplicitOC` | Abstract OC problem (dynamics, costs, IC sampler) |
| Run identity     | `jfb-for-implicit-oc/core/run_io`     | `tag`, `run_id`, every artifact filename          |
| Disk layout      | `jfb-for-implicit-oc/core/paths`      | `results/<ProblemClassName>/.../` directories     |
| Training loop    | `jfb-for-implicit-oc/core/OptimalControlTrainer` | Save / reload / finalize, plotting dispatch |
| Plotting service | `jfb-for-implicit-oc/benchmarking/`   | `Trajectory`, `Panel`, `BenchmarkPlotter`         |
| Concrete problem | `jfb-for-implicit-oc/models/`         | Math, plus `panels()` + `to_trajectory()`         |
| Runner          | `jfb-for-implicit-oc/examples/`       | Wiring only — *no* paths, *no* filenames          |

## Running an existing model

```bash
cd jfb-for-implicit-oc
python examples/example_liquidationportfolio.py
```

Every run deterministically writes the following bundle (timestamps differ
per invocation, so re-running never overwrites):

```
results/LiquidationPortfolioOC/
├── training/
│   ├── best_policy_<tag>_<YYYYMMDD_HHMMSS>.pth
│   ├── history_<tag>_<YYYYMMDD_HHMMSS>.csv
│   ├── loss_curve_<tag>_<YYYYMMDD_HHMMSS>.png
│   └── training-plots/
│       └── rollout_<tag>_<YYYYMMDD_HHMMSS>_NNNN.png   # one per plot_frequency
└── rollouts/
    ├── policy_rollout_<tag>_<YYYYMMDD_HHMMSS>.png      # final rollout (best policy)
    └── trajectory_<tag>_<YYYYMMDD_HHMMSS>.pth          # raw tensor for replay
```

`<tag>` defaults to `"JFB"`; pass `tag="FullAD"` to `OptimalControlTrainer`
to compare two strategies in the same `results/` tree without collisions.

## Recipe: adding a new model

The recipe is **three files**, in order. The trainer + `RunIO` +
`BenchmarkPlotter` combo guarantees the artifact bundle above appears
without you writing any path code.

> **Quick start:** copy `jfb-for-implicit-oc/examples/example_TEMPLATE.py`
> as `examples/example_<myproblem>.py` and resolve the two `TODO` blocks
> (one import, one constructor). Everything else — `sys.path` bootstrap,
> network wiring, optimizer, trainer call — is already in place. Running
> the unmodified template fails fast with a clear message pointing at the
> exact lines to edit.

### Checklist

- [ ] **1. Define the problem in `jfb-for-implicit-oc/models/MyProblem.py`**
  - [ ] `class MyProblemOC(ImplicitOC)` with `state_dim`, `control_dim`
        passed to `super().__init__`
  - [ ] Implement the math contract:
    - [ ] `compute_lagrangian(t, z, u)`
    - [ ] `compute_grad_lagrangian(t, z, u)`
    - [ ] `compute_f(t, z, u)`
    - [ ] `compute_grad_f_u(t, z, u)`
    - [ ] `compute_grad_f_z(t, z, u)`
    - [ ] `compute_G(z)`
    - [ ] `compute_grad_G_z(z)`
    - [ ] `sample_initial_condition()`
    - [ ] `generate_trajectory(u, z0, nt, return_full_trajectory=False)`
  - [ ] Plug into the plotting service (recommended):
    - [ ] `panels(self) -> list[Panel]` — declarative subplot spec
    - [ ] `to_trajectory(self, z_traj, policy=None, path_index=0, label=...) -> Trajectory`
          — pack `(batch, state_dim, nt+1)` tensor into a `Trajectory`
  - [ ] **No paths. No filenames. No matplotlib code.**
  - [ ] If you can't supply `panels()` / `to_trajectory()` yet, the trainer
        falls back to a legacy `plot_position_trajectories(z_traj, save_path=...)`
        method (Quadcopter / MultiBicycle still work this way).

- [ ] **2. Write the runner in `jfb-for-implicit-oc/examples/example_myproblem.py`**
  - [ ] Copy `examples/example_TEMPLATE.py` as the starting point (it
        already has the `sys.path` bootstrap, network wiring, optimizer,
        scheduler, and `trainer.train(...)` call in place)
  - [ ] Resolve **TODO[1]**: import your `MyProblemOC` from `models/`
  - [ ] Resolve **TODO[2]**: instantiate `MyProblemOC(...)` with concrete
        hyperparameters (dynamics, costs, IC distribution); delete the
        `raise NotImplementedError(...)` immediately below it
  - [ ] (Optional) override `Phi(3, 50, ...)` width, `u_min` / `u_max`,
        optimizer, LR scheduler, `epochs`, `plot_frequency` if the
        template defaults are wrong for your problem
  - [ ] Sanity-check the constructor signature: the trainer is built with
        `tag=...` only — no `save_name`, no output paths
  - [ ] **No `os.path.join`. No `save_path=`. No `save_name=`.** The runner
        is purely declarative.

- [ ] **3. Run it**
  - [ ] `cd jfb-for-implicit-oc && python examples/example_myproblem.py`
  - [ ] Confirm the six-artifact bundle landed under
        `results/MyProblemOC/training/` and `results/MyProblemOC/rollouts/`
  - [ ] Open `loss_curve_*.png` and `policy_rollout_*.png` for sanity

### Optional: comparison-vs-reference plots

If your problem has an analytical or BVP reference solution and you want
overlay figures, copy the liquidation comparison utilities:

- [ ] **4. Add a `MyProblemBenchmark`** modeled on
      `liquidation_benchmark.LiquidationBenchmark` (provides the reference
      curves)
- [ ] **5. Add `plot_myproblem_jfb.py`** modeled on `plot_liquidation_jfb.py`
      — it writes to `results/<cls>/benchmark/`, distinct from the trainer's
      `rollouts/` (the trainer plot shows the trained policy alone; the
      benchmark plot overlays it against the reference)

This step is purely additive. Pure JFB research with no reference solution
does *not* need it — the example file already produces a final rollout.

## Smell test: when SoC is being violated

If you find yourself writing any of the following, stop — something is
leaking out of `RunIO` and the right fix is to extend `RunIO`, not to
bypass it:

- `os.path.join(...)` inside a model or example
- `save_path=...` literal inside a model or example
- A `save_name` / `output_dir` argument added to `train()` or
  `OptimalControlTrainer.__init__`
- A new `*_dir` constant defined in a model or example file
- `matplotlib.pyplot` imported inside a model class (panels go through
  `BenchmarkPlotter`, not direct plt calls)

## Where the legacy lives

- `models/Quadcopter.py`, `models/MultiBicycle.py` keep their bespoke
  `plot_position_trajectories(z_traj, save_path=...)` and ride the
  trainer's fallback branch. Migrate them to `panels()` /
  `to_trajectory()` opportunistically.
- `liquidation_benchmark.py` and `plot_liquidation_jfb.py` sit at the
  package top level (rather than in `examples/`) because they are
  comparison-vs-reference utilities — distinct from the trainer's
  trained-policy-alone rollouts.
- `examples/example_liquidationportfolio.py`, `example_multibicycle.py`,
  and `example_multi_quadcopter.py` are fully-instantiated examples, not
  copy templates. Use `examples/example_TEMPLATE.py` for that purpose;
  copying a fully-instantiated example also works but you'll have more
  problem-specific wiring to delete.

## Verifying the recipe

After implementing a new model, the cheapest sanity check is:

```bash
cd jfb-for-implicit-oc
python -m py_compile models/MyProblem.py examples/example_myproblem.py
python examples/example_myproblem.py        # short run, e.g. num_epochs=3
ls results/MyProblemOC/training results/MyProblemOC/rollouts
```

You should see exactly six files in the two directories, with timestamps
inside the run window.
