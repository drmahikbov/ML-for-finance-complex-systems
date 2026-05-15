---
title: "Part II — Reinforcement learning extension"
theme: default
class: text-left
highlighter: shiki
transition: fade
mdc: true
---

# 2.1. Part II — Reinforcement learning extension

## 2.2. From implicit OC to learned policies

- **State** $z$ (Markov), **initial** $z(0)=x$; **policy** $\pi_\theta(u\mid t,z)$ or deterministic $u_\theta(t,z)$
- **Trajectory** still $z_x(t)$ under chosen controls; **objective** $\mathbb{E}\bigl[\sum_t \gamma^t L(t,z_x(t),u_t) + G(z_x(T))\bigr]$

<!--
This file is merged after section 1 (subsections **1.1**–**1.9**); titles here use section 2 subnumbering (2.1–2.9). When you run `slidev slides-part2-rl.md` alone, numbering still reflects “Part II” of the full talk. Notation: z is state, x is only the initial label in z(0)=x.
-->

---

## 2.3. MDP view (discrete time)

$$
z_{k+1} = F(k, z_k, u_k, \xi_k), \qquad z_0 = x,
$$

with i.i.d. shocks $\xi_k$ (drop $\xi_k$ for deterministic dynamics).

$$
J(\theta; x) = \mathbb{E}_{\pi_\theta,\,\xi}\left[ \sum_{k=0}^{K-1} \gamma^k L(k, z_k, u_k) + G(z_K) \right].
$$

- **Goal:** optimize $\theta$ from **rollouts** (model-free) or **differentiable simulators** (model-based)

<!--
Connect to Part I: continuous-time limit relates to HJB; here we emphasize sampling and expectations. Gamma is discount; can be set to 1 for finite-horizon OC analogues.
-->

---

## 2.4. Bellman vs Hamilton–Jacobi (informal)

$$
\phi_\theta(k,z) = \max_u \mathbb{E}_\xi\left[ L(k,z,u) + \gamma\, \phi_\theta\bigl(k+1, F(k,z,u,\xi)\bigr) \right].
$$

- **Continuous-time limit:** DPP $\to$ HJB; **co-state** $p=\nabla_z \phi_\theta$ plays the role of **value gradient** in smooth regimes

<!--
Speaker bridge: RL tabular/value iteration targets the same optimality object as the HJB, but with noise and function approximation. Neural phi_theta as critic is the natural OC analogue.
-->

---

## 2.5. Policy gradient (baseline sketch)

$$
\nabla_\theta J \approx \mathbb{E}\left[ \sum_k \nabla_\theta \log \pi_\theta(u_k\mid k, z_k)\, \hat{A}_k \right].
$$

- **REINFORCE:** high variance; **advantage** $\hat{A}_k$ from critic or TD reduces variance

<!--
Mention pathwise derivatives when the simulator is reparameterizable and dynamics smooth—then chain rule resembles implicit differentiation through the rollout, but without the fixed-point inner solve unless control is implicit.
-->

---

## 2.6. Actor–critic and the OC dictionary

| RL | OC / this project |
|----|-------------------|
| Critic $\phi_\psi$ or $Q_\psi$ | $\phi_\theta(t,z)$ |
| Target / Bellman residual | HJB residual (PDE side) |
| Actor $\pi_\theta$ | $u_\theta^*(t,z)$ from **$\nabla_u H=0$** (implicit) |

- **Key twist:** actor output may be **defined implicitly** (fixed point), not a free map $G(\cdot)$

<!--
This is the conceptual link to Part I: the actor is not always an explicit feedforward map; JFB-style training is about differentiating through that implicit definition without unrolling the solver.
-->

---

## 2.7. RL when the Hamiltonian maximizer is implicit

$$
\nabla_u H\bigl(t, z, \nabla_z \phi_\theta(t,z), u_\theta^*\bigr) = 0
\quad \leadsto \quad
u_\theta^* = T_\theta(u_\theta^*; t,z).
$$

- **Rollout:** simulate $z_x(t)$ using $u_\theta^*$ from the **equilibrium** of $T_\theta$
- **Update:** policy/value losses use **implicit** $\partial u_\theta^*/\partial\theta$ (e.g. JVP / JFB), not full unrolling of the inner loop in the tape

<!--
Emphasize memory and stability: same story as Part I, but data now come from stochastic trajectories and possibly unknown dynamics; exploration and off-policy corrections become central, which we only name here.
-->

---

## 2.8. Practical challenges (Part II focus)

- **Sample complexity** vs deterministic PDE/HJB residuals on a grid
- **Exploration** when $u$ is constrained (execution, liquidation)
- **Off-policy / distribution shift** if replay or behavior policy differs from $u_\theta^*$
- **Stability:** advantage estimation + implicit layers + long horizons

<!--
Keep this slide as the honest “open problems” slide for RL extension; speaker can give one concrete example (e.g. liquidation with noisy fills) if time permits.
-->

---

## 2.9. Outlook

- Combine **Monte Carlo rollouts** with **implicit Hamiltonian** stationarity for the actor
- **Critic:** $\phi_\theta$ with **diffusion** terms (stochastic HJB) if dynamics are SDEs
- **Research hooks:** variance reduction for $\nabla_{zz}\phi$, multi-step TD, and contractivity of $T_\theta$ under policy updates

<!--
Close Part II by pointing back to Part I takeaways: structure (e.g. removing redundant coordinates) remains crucial; RL adds data-driven uncertainty and exploration on top of implicit differentiation.
-->
