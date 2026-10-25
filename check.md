MPC in Koopman Space — Math Formulation
State / control

$$x\in\mathbb R^{D}\ (D=120),\qquad x=[x^{\text{pos}};,x^{\text{vel}}],\quad x^{\text{pos}}\in\mathbb R^{60},\ x^{\text{vel}}\in\mathbb R^{60}$$
$$u=(\text{cmd_dx},\text{cmd_dy})\in\mathbb R^{2}$$

Normalization (fit once on the training data, fixed at inference)

$$\bar x=\frac{x-\mu_x}{\sigma_x},\qquad \bar u=\frac{u}{c_u}$$

Learned Koopman model (encoder/decoder are MLPs, operator is affine)

$$z=E(\bar x)\in\mathbb R^{256},\qquad \hat{\bar x}=D(z),\qquad z_{k+1}=Az_k+B\bar u_k$$

Per-tick optimal-control problem
At time $t$, given current state $x_t$ and fixed goal $x_g$:

$$z_0=E(\bar x_t),\qquad \bar x_g=\frac{x_g-\mu_x}{\sigma_x}$$

Decision variables — the next $H$ commands, reparameterized to respect box constraints exactly:

$$\bar u_k=\bar u_{\max}\tanh(\theta_k),\qquad k=0,\dots,H-1,\qquad \bar u_{\max}=\min!\Big(1,\ \tfrac{u_{\max}}{c_u}\Big)$$

Rollout (exact, affine):

$$z_{k+1}=Az_k+B\bar u_k,\qquad \hat{\bar x}{k+1}=D(z{k+1})$$

Objective:

$$
\min_{\theta_0,\dots,\theta_{H-1}}\ \ J=\frac1H\sum_{k=0}^{H-1}\Big[,w_{k+1},\big|,W^{1/2}\big(\hat{\bar x}_{k+1}-\bar x_g\big)\big|2^2 ;+; r,|\bar u_k|2^2 ;+; r{\text{rate}},|\bar u_k-\bar u{k-1}|_2^2\Big]
$$

with

$$w_{k}=1+(q_f-1)\frac{k-1}{H-1}\quad(\text{linear ramp }1\to q_f),\qquad W=\operatorname{diag}\big(\underbrace{q,\dots,q}{\text{pos block}},\underbrace{q\cdot\rho,\dots,q\cdot\rho}{\text{vel block}}\big)$$

$q$ = position tracking weight (--q), $\rho$ = q_vel_ratio (velocity weighted lightly, as a soft regularizer)
$r$ = control effort weight, $r_{\text{rate}}$ = control-rate (smoothness) weight, $\bar u_{-1}\equiv$ last command actually applied
$q_f$ = terminal-weight multiplier, pushing the plan to actually arrive near $x_g$ by step $H$
Solved by unconstrained gradient descent (Adam) on $\theta$ — the $\tanh$ reparameterization makes the box constraint $|\bar u_k|\infty\le\bar u{\max}$ automatic, so no explicit QP/constrained solver is needed.

(Note: cost is evaluated on the decoded state $\hat{\bar x}$, not raw $|z_k-\bar z_g|^2$ — the encoder isn't distance-preserving, so a pure-latent quadratic cost measurably degrades position tracking in practice. Only the affine rollout $z_{k+1}=Az_k+Bu_k$ stays exact/linear; the leaf cost passes once more through the frozen decoder MLP, which costs strict convexity but is still fully differentiable.)

Receding-horizon (closed-loop) scheme
Solve the problem above from $x_t$ → get plan $\bar u_0^*,\dots,\bar u_{H-1}^*$
Apply only the first command: $u_t=c_u,\bar u_0^*$
Observe/step to $x_{t+1}$ (model self-rollout, or a real plant via plant_step)
Warm-start the next solve with the shifted plan: $\theta_k^{(0)}\leftarrow\operatorname{atanh}!\big(\bar u^*{k+1}/\bar u{\max}\big)$ for $k<H-1$, repeat the last entry for $k=H-1$
$t\leftarrow t+1$, repeat until $|\hat x_t^{\text{pos}}-x_g^{\text{pos}}|_{\text{RMS}}<\text{tol}$ or steps exhausted
Corresponds directly to KoopmanMPC.solve() (steps 1–2) and KoopmanMPC.track() (the outer loop) in mpc.py.

