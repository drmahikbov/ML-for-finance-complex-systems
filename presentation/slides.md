---
title: "Implicit Hamiltonian Learning for Optimal Control: Replication and Stochastic Extension"
theme: default
class: text-left
highlighter: shiki
transition: fade
mdc: true
---

# Implicit Hamiltonian learning for high dimensional optimal control

### RL and Stochastic control Extension

<em class="paper-ref">End-to-End Training of High-Dimensional Optimal Control with Implicit<br />Hamiltonians via Jacobian-Free Backpropagation</em>

<!--
Title and motivation together: many OC problems admit a Hamiltonian whose maximizer has no closed form, so standard “differentiate through an explicit policy map” pipelines break. We still want a neural surrogate phi_theta for the value, recover feedback implicitly from H_u=0, roll trajectories z_x(t) forward from sampled initial conditions x, and train against the objective. Roadmap: HJB/PMP recap, implicit vs explicit Hamiltonians, JFB, Almgren–Chriss reproduction, why full state (q,S,X) failed, reduced (q,S) results, stochastic outlook, conclusions.
-->

---

## 1.1 - Classical optimal control - PMP perspective


$$
J(u; x) = \int_0^T L\bigl(t, z_x(t), u(t)\bigr)\,dt + G\bigl(z_x(T)\bigr),
$$

$$
\dot z_x(t) = f\bigl(t, z_x(t), u(t)\bigr), \qquad z_x(0) = x.
$$

**Generalized Hamiltonian**

$$
\mathcal{H}(t, z_x, p_x, u) = -\,p_x^\top f(t, z_x, u) - L(t, z_x, u).
$$

**State and adjoint** along an optimal control $u^\star$

$$
\dot z_x = -\nabla_{p_x} \mathcal{H}(t, z_x, p_x, u^\star), \qquad z_x(0) = x,
$$

$$
\dot p_x = \nabla_{z_x} \mathcal{H}(t, z_x, p_x, u^\star), \qquad p_x(T) = \nabla G\bigl(z_x(T)\bigr).
$$
**Optimality**
$$
u^\star(t) \in \arg\max_{u} \mathcal{H}\bigl(t, z_x(t), p_x(t), u\bigr)
\quad \Longrightarrow \quad
\nabla_u \mathcal{H}\bigl(t, z_x(t), p_x(t), u^\star(t)\bigr) = 0.
$$


---

## 1.2 - Classical optimal control — HJB formulation


**Value surrogate** $\phi_\theta(t,z)$

$$
\phi_\theta(t, z) = \sup_{u} \left[ \int_t^T L\bigl(s, z(s), u(s)\bigr)\,ds + G\bigl(z(T)\bigr) \right].
$$

**HJB**

$$
\partial_t \phi_\theta(t, z) + \sup_{u \in U} \mathcal{H}(t, z, p, u). = 0.
$$

**Bridge with original formulation**

$$
p_x(t) = \nabla_z \phi_\theta\bigl(t, z_x(t)\bigr).
$$



---

## 1.3. Paper's contribution

- **Problem:** **no closed form** for $u$ in $\nabla_u \mathcal{H} = 0$


<PipelineBox title="End to end training pipeline">

$$
\phi_\theta(t,z) \;\longrightarrow\; p = \nabla_z \phi_\theta(t,z) \;\longrightarrow\; \text{INN } u_\theta^*(t,z) \;\longrightarrow\; z_x(t) \;\longrightarrow\; \text{objective}
$$

</PipelineBox>


<div class="grid grid-cols-2 gap-8 items-center">

<div>

**Fixed-point characterization**

$$
u_\theta^* = T_\theta(u_\theta^*; t, z).
$$

</div>

<div>

**JFB approximation**

$$
\frac{\partial u_\theta^*}{\partial \theta}
=
\underbrace{\left(I - \tfrac{\partial T_\theta}{\partial u}\right)^{-1}}_{m\times m \text{ solve}}
\frac{\partial T_\theta}{\partial \theta}
\;\approx\;
\frac{\partial T_\theta}{\partial \theta}
$$

</div>

</div>


<!-- **Gains** ($m=\dim u$, $K$ inner iterations) -->

|              | Exact IFT          | JFB                |
|:-------------|:-------------------|:-------------------|
| **Compute**  | $\mathcal{O}(m^3)$ | $\mathcal{O}(m^2)$ |
| **Memory**   | $\mathcal{O}(K)$   | $\mathcal{O}(1)$   |

<!--
Paper: “End-to-End Training of High-Dimensional Optimal Control with Implicit Hamiltonians via Jacobian-Free Backpropagation” (arXiv:2510.00359). Eq. (10) is the exact implicit derivative with J_\theta=(I-\partial_u T)^{-1}; the m^3 cost is from solving that m\times m linear system at each (t,z). JFB Eq. (11) replaces J_\theta^{-1} by I (identity), avoiding the inversion—hence m^2 scaling in the control dimension per the paper’s Sec. III-B discussion. Unrolling stores the full inner fixed-point chain in the autograd tape (K-fold), which is a different bottleneck from the m^3 vs m^2 comparison.
-->

---

## 1.4. Training algorithm - (extra details)


<pre v-pre class="algo-box"><code><span class="algo-ln"> 1:</span>  <span class="algo-kw">Initialize</span> networks with parameters <span class="algo-math">θ</span>
<span class="algo-ln"> 2:</span>  <span class="algo-kw">for</span> iteration = 1, 2, … <span class="algo-kw">do</span>
<span class="algo-ln"> 3:</span>      Sample a batch of initial states <span class="algo-math">{x_i} ∼ ρ</span>
<span class="algo-ln"> 4:</span>      <span class="algo-kw">for</span> each trajectory <span class="algo-kw">do</span>
<span class="algo-ln"> 5:</span>         <span class="algo-kw">for</span> <span class="algo-math">k</span> = 0, …, <span class="algo-math">N_t − 1</span> <span class="algo-kw">do</span>
<span class="algo-ln"> 6:</span>              Compute grad of the value function <span class="algo-math">p ← ∇_z φ_θ (t_k, z)</span> <span class="algo-cm"># discrete adjoint</span>
<span class="algo-ln"> 7:</span>              INN solves fixed point eq for <span class="algo-math">u</span>  <span class="algo-cm"># K steps detached, then K′ steps on-graph</span>
<span class="algo-ln"> 8:</span>              Increase running loss 
<span class="algo-ln"> 9:</span>              Evolve state <span class="algo-math">z</span> numerically <span class="algo-cm"># detached, not computed in the loss</span>
<span class="algo-ln"> 10:</span>          <span class="algo-kw">end for</span>
<span class="algo-ln">11:</span>          Mix running loss with terminal <span class="algo-math">G(z)</span>
<span class="algo-ln">12:</span>      <span class="algo-kw">end for</span>
<span class="algo-ln">13:</span>      Average batch objectives for the final loss
<span class="algo-ln">14:</span>      Backprop on <span class="algo-math">θ</span>
<span class="algo-ln">14:</span>  <span class="algo-kw">end for</span>
</code></pre>




- Gradient flows **only** through the **tracked tail** of length $K'$ — not through the $K$-step convergence loop or the state transitions
- Deployment: same inner $u$-loop, with $\theta=\theta^\star$ frozen

<!--
Speaker note: Inner loop on **u**: many Hamiltonian-ascent steps with no graph tape, then a short **tracked** tail (length K′) so only that tail contributes to gradients in **θ** (Term I / JFB). Dynamics in **z** are stepped without differentiating through the transition. Costate **p** is whatever **∇_z φ_θ** implementation you couple into **H** — the narrative here is Pontryagin/HJB bookkeeping, not a claim about computing that gradient by reverse-mode autodiff.
-->

---

## 1.5. Almgren–Chriss reproduction benchmark

**State**

$$
z = (q, S, X).
$$

**Dynamics**

$$
\dot q = -u, \qquad
\dot S = \kappa u, \qquad
\dot X = -S u - \eta\,|u|^\gamma.
$$

- $q$: inventory · $S$: impacted price · $X$: cash · $u$: liquidation rate
- **Why here:** closed-form PDE/BVP reference enables quantitative comparison

<!--
This is the canonical optimal execution model with temporary price impact and liquidation costs. The benchmark is attractive because a deterministic boundary-value or PDE solution can be compared against the learned feedback without ambiguity about the “truth.”
-->

---

## 1.6. Why the first formulation failed

- Augmenting the state with $X$ made learning **harder**, even though $X$ is **structurally redundant**
- **Decomposition**

$$
\phi_\theta(t, q, S, X) = X + \widetilde{\phi}_\theta(t, q, S).
$$

- Hence $\partial_X \phi_\theta = 1$: no need to learn $X$ as an independent state coordinate

<!--
Intuition: cash is bookkeeping given inventory and price paths; the marginal value of an extra dollar in the book is one. Forcing the network to fit X as a coordinate duplicates information the model can infer analytically, inflates gradients, and couples the HJB residual to a direction that should be exact.
-->

---

## 1.7. Reduced formulation and empirical outcome

**Reduced state**

$$
z = (q, S).
$$

- Fewer dimensions; fixed costate component **removed analytically**
- Training **stabilizes**; learned $u_\theta^*$ **tracks** the BVP benchmark
- **Figures:** add `![](relative/path.png)` below when exporting plots

<!--
Insert figure: learned_control_vs_bvp.png
Insert figure: inventory_trajectory.png
Insert figure: loss_curve.png
Example after rendering: ![](jfb-for-implicit-oc/results/LiquidationPortfolioOC/benchmark/jfb_vs_exactbvp_benchmark_gamma1.5alpha1.png). The reduced formulation aligns the learned Hamiltonian structure with the analytical separability of cash, which removes a degenerate direction from the residual.
-->

---

## 1.8. Stochastic extension (outlook)

$$
dZ_t = f(t, Z_t, u_t)\,dt + \sigma(t, Z_t, u_t)\,dW_t.
$$

$$
\partial_t \phi_\theta + \max_u \left[
L + \nabla_z \phi_\theta^\top f + \tfrac{1}{2}\operatorname{Tr}\bigl(\sigma \sigma^\top \nabla_{zz}^2 \phi_\theta\bigr)
\right] = 0.
$$

- If $\sigma$ **does not** depend on $u$: fixed-point **form** for $u^*$ often **unchanged**, but $\phi_\theta$ must capture **diffusion**
- If $\sigma$ **depends** on $u$: the **algebraic** fixed-point for $u^*$ **changes**
- **Open angles:** Hessian / trace estimation, MC rollouts, contractivity of the stochastic Hamiltonian map

<!--
This is the forward-looking slide: stochastic HJB adds a trace term involving the Hessian of $\phi_\theta$. For neural $\phi_\theta$, that raises questions of variance and stability. Control-dependent diffusion couples into the implicit first-order condition for u, altering T_theta. Research questions include efficient JVP/HVP schemes, sample-based pathwise losses, and whether the inner map remains well-posed / contractive after discretization.
-->

---

## 1.9. Takeaways

1. **JFB** trains with **implicit** Hamiltonians **without** unrolling the inner fixed-point solver in the autodiff tape
2. **Almgren–Chriss** required exploiting **$\phi_\theta = X + \widetilde{\phi}_\theta$** so $X$ is not learned as a redundant coordinate
3. **Stochastic** models add **diffusion**, **Monte Carlo** rollouts, and—when $\sigma$ depends on $u$—**new** implicit structure in the control equation

<!--
Close by tying back to the opening pipeline: implicit u_theta^* from grad_u H = 0, differentiated via JFB, integrated along z_x(t). The benchmark story shows that physics/bookkeeping structure matters as much as the implicit-differentiation trick. The stochastic slide frames honest next steps rather than finished work. Section 2 (merged `slides-part2-rl.md`) covers the RL / stochastic-control extension in more detail.
-->

---
src: ./slides-part2-rl.md
---
