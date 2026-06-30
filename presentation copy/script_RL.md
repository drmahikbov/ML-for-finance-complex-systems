## Marin et Hector (10min)

---

## 2.1: Setup/motivation (45sec)

- "In Part I : built an implicit policy from a value function φ_θ. Now = drop a key assumption: **dynamics f are unknown**."
- "We still know the cost L, terminal cost G, and initial distribution ρ. cannot differentiate f anymore."
- "Goal: keep the same architecture, replace what we lost with data."

---

## 2.2 & 2.3: Discretization (45sec)

- "We move from continuous time to discrete rollouts: z\_{k+1} = F(t_k, z_k, u_k)."
- "We don't have F as a formula, we only see samples (z*k, u_k, z*{k+1})."
- "The objective is unchanged: sum of running costs plus terminal cost, averaged over initial states."

---

## 2.4 to 2.7: Local Jacobian estimation (1min30)

**main part of pres**

- "Since we can't differentiate F, we **estimate its Jacobians locally** along the current trajectory (just a linearization regression)."
- "A_k ≈ ∂F/∂z and B_k ≈ ∂F/∂u -> these replace the missing analytical derivatives."
- "**How?** Perturb the controls around the rollout with Gaussian noise, observe Δz\_{k+1}, and fit by least squares."
- use **recursive least squares (RLS)**, sample-efficient, reuses past rollouts."
- "The result is a **local linear model** -> F̂_k valid near the current trajectory point."

---

## 2.8 & 2.9: Estimated Hamiltonian (1min)

- "Plug the local model into the Hamiltonian. We get Ĥ_k, an estimated Hamiltonian."
- "**Key simplification** (slide 2.9): the gradient of Ĥ w.r.t. u is just −B_k^T p − ∇_u L."
- "So **inside the fixed-point operator, we only need B_k**. We never need to evaluate f itself."
- "This means the implicit policy still works, we just replaced ∇_u f with B_k."

---

## 2.10 & 2.11: Adjoint with estimated dynamics (1min)

- "For the gradient, we need a costate. We compute it backward using A_k:"
- "p*N = ∇G(z_N), then p_k = A_k^T p*{k+1} + Δt ∇_z L."
- "Two costates appear in the pipeline (point to the table):"
  - "**One for the control**: ∇*z φ*θ, used inside the fixed point."
  - "**One for the gradient**: p̂_k, from the backward recursion."
- "Same Pontryagin structure, but A_k replaces ∇_z f."

---

## 2.12 & 2.13: JFB gradient (1min)

- "The exact policy gradient has two problems: ∇_u f is unknown, and the implicit derivative requires inverting (I − ∂T/∂u). Expensive."
- "**JFB = Jacobian-Free Backpropagation**, replaces that inverse with just ∂T/∂θ. One step instead of a full implicit solve."
- "Combined with B_k ≈ ∇_u f, we get the **central training formula**."
- "This is what we actually optimize."

---

## 2.14: Algorithm summary (1min)

1. "Sample initial state from ρ."
2. "Roll out in the environment."
3. "Compute controls via the implicit fixed point."
4. "Update Jacobian estimates A_k, B_k."
5. "Backward adjoint pass."
6. "JFB parameter update."

> if no more time, jump here and skip 2.10–2.13

---

## 2.16 & 2.17: Benchmark problems (45sec)

**Van der Pol (2.16):**

- "Nonlinear oscillator, scalar control u acts as an external force, goal is to drive (x_1, x_2) to the origin."

**Portfolio (2.17):**

- "Wealth dynamics with risky and risk-free assets. The drift μ and rate r are **unknown**, exactly the RL setting."
- "Trade-off: maximize log terminal wealth, penalize aggressive positions via exp(π²)."

---

## 2.21 & 2.22: Results (1min)

**Van der Pol (2.21):**

- "The RL-trained controller matches the **oracle** (which has full access to f). Convergence to the origin with no dynamics model."

**Portfolio (2.22):**

- "Left: training loss decreasing smoothly."
- "Right: learned policy rollout, wealth trajectory."
- "Mid-training snapshot at epoch 200 shows policy is already shaping correctly."

---

## 2.23: Practical challenges (45 sec)

- "**Locality**: RLS estimates become stale if rollouts drift."
- "**Exploration**: we need enough perturbation σ to identify B_k."
- "**Long-horizon sensitivity**: A_k errors compound through the backward adjoint."

---

## 2.24: Outlook (30sec)

- "Same Gelphman pipeline: φ*θ → ∇φ*θ → u\*\_θ → J(θ)."
- "The only structural change: f and ∇f are replaced by local estimates A_k, B_k."
- "Next: unknown running cost L, stochastic dynamics, formal descent guarantees, multi-asset portfolios."

---
