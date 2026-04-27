# Kalman Filters and MapFormer — Intuition Document

A walkthrough conversation building up Kalman-filter intuition and showing
how it connects to MapFormer's Level 1.5 InEKF extension. Written so
another chat can pick up the thread without prior context.

---

## 1. Why does the Kalman filter help with the MapFormer design?

The intuition is **dead-reckoning vs. fixing position from landmarks**.

**What vanilla MapFormer does.** It builds a position estimate $\theta_t$ by
*cumulatively summing* action-driven increments:
$\theta_t = \omega \odot \mathrm{cumsum}(\Delta_t)$.
That's pure path integration — open-loop dead reckoning. Every action
contributes once and the angle just accumulates.

**The problem.** Path integration drifts:
- If actions are noisy, errors compound linearly with $T$.
- If you train at $T=128$ but evaluate at $T=512$, the $\theta$ values fall
  outside any range the network has seen.
- Observations *are* available at every step, but vanilla MapFormer only
  uses them through softmax attention — i.e. for *retrieval*, never to
  correct the position estimate itself. The internal $\theta$ never gets
  pulled back toward truth.

**What the Kalman filter adds.** A second pathway that says: "given what
you just observed, what does that imply about where you are?" The
measurement head $\hat z_t$ is the inverse model — observation → implied
angle. The innovation
$\nu_t = \mathrm{wrap}(\hat z_t - \theta_{\mathrm{path},t})$
is "how wrong does the dead-reckoning estimate look in light of this
observation." The Kalman update folds that correction back into $\theta$
before it goes into RoPE.

**Why it's the right shape for MapFormer.**
1. **Bounded error under observability.** Drift can't grow unboundedly —
   every informative observation pulls $\theta$ back. That's the
   OOD-length win.
2. **Heteroscedastic gain matches the task.**
   $K_t = \Pi/(\Pi+R_t)$ — at an aliased blank token $R_t$ is large and
   the filter ignores the observation; at a unique landmark $R_t$ is
   small and the filter snaps to it. That's the landmark win.
3. **Group-correct.** Wrapping innovations via $\mathrm{atan2}(\sin, \cos)$
   makes the scalar EKF identical to the full Lie-Group EKF on
   $\mathrm{SO}(2)$ (Markovic 2017), so corrections compose correctly with
   the rotation structure MapFormer already uses.
4. **Same parallelism class.** The correction is one scalar affine scan —
   same $\mathcal{O}(\log T)$ depth as MapFormer's cumsum.

In one line: **MapFormer had the *predict* step of a Kalman filter and was
missing the *update* step.** Level 1.5 adds it back, parallelised, and
lets the model learn how much to trust each observation per-token.

---

## 2. Doesn't MapFormer already generalise OOD via path integration?

It does — and that claim still stands. The two are about different things,
and they stack rather than compete.

**MapFormer's OOD claim is "path integration vs. learned positional
embedding."** The comparison is against RoPE / CoPE / TAPE, which encode
position as a (possibly content-gated) function of *token index*. At a
longer sequence length their position encodings either repeat, fall off
the trained manifold, or just don't say anything sensible. MapFormer
side-steps this by computing position from the *action stream* via cumsum
— so the representation of "I'm at cell $(7,3)$" is the same whether you
got there in 10 steps or 500. That's why MapFormer generalises across
length and CoPE doesn't.

**But path integration alone is open-loop.** Two limits show up once you
start pushing it:

1. **Drift under action noise.** A cumulative sum of noisy increments has
   variance that grows linearly in $T$. There's nothing in the predict
   step that pulls $\theta$ back toward truth. The MapFormer paper trains
   and tests on *clean* actions, so this never bites them.
2. **Length OOD even on clean data.** MapFormer-WM drops from 0.955 at
   $T{=}128$ to $0.913 \pm 0.037$ at $T{=}512$ — small but non-zero. Pure
   cumsum is correct in expectation, but the *learned components* (the
   action-to-Lie projection, the $\omega$ schedule, the measurement head
   decoding $\theta$ back to a vocabulary distribution) are imperfect
   approximations trained on a bounded $\theta$ range.

**Where the Kalman update fits.** It's the *correction* (update) step a
full filter has and pure path integration is missing.

The layered story:
- **No path integration** (CoPE, RoPE, MambaLike): can't represent
  position correctly at all out-of-length. Accuracy collapses.
- **Path integration alone** (MapFormer): correct in expectation, but
  drifts under noise and degrades slowly under length.
- **Path integration + Kalman update** (Level 1.5): bounded error under
  observability — the formal classical-filtering guarantee.

---

## 3. The standard Kalman filter formulation

### Setup

A linear-Gaussian state-space model:
$$x_{t+1} = F_t x_t + B_t u_t + w_t, \qquad w_t \sim \mathcal{N}(0, Q_t)$$
$$y_t = H_t x_t + v_t, \qquad v_t \sim \mathcal{N}(0, R_t)$$

You have a hidden state $x_t$ that evolves linearly under known dynamics
$F$ driven by a known control $u_t$ with Gaussian process noise $Q$. You
observe $y_t$ as a linear function of $x_t$ with Gaussian measurement
noise $R$.

### The pieces

- $x_t$ — what you want to know (e.g. position and velocity).
- $u_t$ — known control inputs.
- $F$ — dynamics matrix.
- $B$ — how controls map into the state.
- $Q$ — process noise covariance: how much you *don't* trust your
  dynamics model.
- $H$ — observation matrix.
- $R$ — measurement noise covariance.

### Two steps per timestep

**Predict** (push the prior forward):
$$\hat x_{t|t-1} = F \hat x_{t-1|t-1} + B u_t$$
$$P_{t|t-1} = F P_{t-1|t-1} F^\top + Q$$

The mean moves under $F$ and $Bu_t$. The covariance grows: stretched by
$F$, inflated by $Q$. **Predict always increases uncertainty** — you know
strictly less about where you are after time passes than before.

**Update** (fold in the observation):
$$\nu_t = y_t - H \hat x_{t|t-1} \quad \text{(innovation)}$$
$$S_t = H P_{t|t-1} H^\top + R \quad \text{(innovation covariance)}$$
$$K_t = P_{t|t-1} H^\top S_t^{-1} \quad \text{(Kalman gain)}$$
$$\hat x_{t|t} = \hat x_{t|t-1} + K_t \nu_t$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

The mean moves toward what the measurement implies, weighted by $K_t$.
The covariance shrinks. **Update always decreases uncertainty.**

### What the Kalman gain does, intuitively

Scalar case ($H{=}1$):
$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$

A **trust ratio**:
- Prior very uncertain ($P \gg R$): $K \approx 1$, trust the measurement.
- Sensor very noisy ($R \gg P$): $K \approx 0$, ignore the measurement.
- In between, blend by relative confidence.

This is exactly **inverse-variance weighting**: if you have two unbiased
estimates with variances $\sigma_1^2, \sigma_2^2$, the optimal linear
combination weights them by $1/\sigma_i^2$. The Kalman gain falls out
immediately.

### Why this is optimal

1. **Bayes-optimal under linear-Gaussian assumptions.** Posterior
   $p(x_t \mid y_{1:t})$ is *exactly* Gaussian, computed exactly.
2. **Minimum MSE among linear estimators (BLUE).** Even if noise isn't
   actually Gaussian.
3. **Innovation sequence $\{\nu_t\}$ is white** when correctly tuned.
   Useful diagnostic.

### Observability and bounded error

Bounded-error guarantee depends on **observability**: over enough time,
measurements have to be informative enough about the state that the
covariance recurrence stays bounded.

Formally: observability matrix
$$\mathcal{O} = \begin{bmatrix} H \\ HF \\ HF^2 \\ \vdots \\ HF^{n-1} \end{bmatrix}$$
must have full rank $n$. If so, $P^*$ is finite and bounded indefinitely.

For MapFormer: actions alone are not observable for absolute angle (any
starting offset gives the same dead-reckoning). Adding observations that
carry positional information *makes the system observable*. That's where
Level 1.5's bounded-error story comes from.

---

## 4. Deriving predict and update from first principles

### What we want

A hidden state $x_t$ we can't see, noisy measurements $y_1, \ldots, y_t$.
The honest thing to ask:
$$p(x_t \mid y_{1:t})$$
the distribution over the current state given everything observed so far.
Once you have this *belief*, you can read off mean, variance,
confidence intervals.

### The recursive trick

Suppose we already have $p(x_{t-1} \mid y_{1:t-1})$. Two things happen
between $t-1$ and $t$:
1. **Time passes.** State evolves under $x_t = F x_{t-1} + B u_t + w_t$.
2. **A new measurement arrives.** $y_t = H x_t + v_t$.

Each one corresponds to one rule of probability — and that's where
predict and update come from.

### Step 1 (predict): time passes, marginalise over the past

By the chain rule:
$$p(x_t \mid y_{1:t-1}) = \int p(x_t \mid x_{t-1}) \, p(x_{t-1} \mid y_{1:t-1}) \, dx_{t-1}$$

In English: "for every possible $x_{t-1}$, weight by how likely it was,
then push forward through the dynamics."

This integral is the **predict step**. Both factors are Gaussian, so the
integral is too.

If previous belief is $\mathcal{N}(\hat x_{t-1|t-1}, P_{t-1|t-1})$ and
$x_t = F x_{t-1} + B u_t + w_t$ with $w_t \sim \mathcal{N}(0, Q)$:
$$\hat x_{t|t-1} = F \hat x_{t-1|t-1} + B u_t$$
$$P_{t|t-1} = F P_{t-1|t-1} F^\top + Q$$

"Push the Gaussian belief through the dynamics, add the process-noise
covariance to the spread."

### Step 2 (update): measurement arrives, apply Bayes' rule

$$p(x_t \mid y_{1:t}) = \frac{p(y_t \mid x_t) \, p(x_t \mid y_{1:t-1})}{p(y_t \mid y_{1:t-1})}$$

Standard **prior × likelihood, normalised**.

### Working out the update for Gaussians

Build the **joint** distribution of $(x_t, y_t)$ given $y_{1:t-1}$, then
condition.

$$\begin{pmatrix} x_t \\ y_t \end{pmatrix} \;\bigg|\; y_{1:t-1} \;\sim\; \mathcal{N}\!\left( \begin{pmatrix} \hat x_{t|t-1} \\ H \hat x_{t|t-1} \end{pmatrix}, \begin{pmatrix} P_{t|t-1} & P_{t|t-1} H^\top \\ H P_{t|t-1} & H P_{t|t-1} H^\top + R \end{pmatrix} \right)$$

Cross-covariance $P_{t|t-1} H^\top$ tells how much state and measurement
co-vary; measurement variance is $H P_{t|t-1} H^\top + R$ (state
uncertainty mapped through $H$, plus sensor noise).

Apply the **Gaussian conditioning formula**: for jointly Gaussian
$(X, Y)$,
$$X \mid Y = y \sim \mathcal{N}(\mu_X + \Sigma_{XY}\Sigma_{YY}^{-1}(y - \mu_Y), \;\Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX})$$
This is just "regress $X$ on $Y$ with the optimal least-squares slope."

Plug in:
$$\hat x_{t|t} = \hat x_{t|t-1} + \underbrace{P_{t|t-1} H^\top S_t^{-1}}_{=:\, K_t} \, \underbrace{(y_t - H \hat x_{t|t-1})}_{=:\, \nu_t}$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

The Kalman gain $K_t$ wasn't pulled out of nowhere — **it's the
regression slope of state on measurement** under the joint Gaussian.

### Where predict and update come from, cleanly

| Operation | Origin |
|---|---|
| Predict | Marginalising over $x_{t-1}$ via the chain rule |
| Update  | Conditioning on $y_t$ via Bayes' rule |

Whole filter in one sentence: **"Maintain a Gaussian belief over the
state. When time passes, marginalise out the past (predict). When a
measurement arrives, condition on it (update)."**

Both operations preserve Gaussianity under linear-Gaussian dynamics, so
we only need to track mean and covariance — that's what makes the filter
cheap.

### Aesthetic notes

- Predict has no measurement; runs *open-loop*. Path integration in
  MapFormer is a predict-only filter.
- Update has no dynamics; pulls belief toward reality.
- Predict pushes uncertainty up; update brings it down. In steady state
  under observability, they balance at a fixed point — the DARE solution.
- **Both steps are local in time**: predict uses only previous belief;
  update uses only current prior + current measurement. That locality is
  what makes the recursion $\mathcal{O}(T)$ sequentially.

---

## 5. How Level 1.5 differs from textbook Kalman

| Component | Classical Kalman | Level 1.5 |
|---|---|---|
| State $x_t$ | physical (position, velocity, …) | $\theta_t$ on $\mathrm{SO}(2)$, per (head, block) |
| Dynamics $F$, $B$, $u_t$ | known from physics | "predict" is MapFormer's $\theta_{\mathrm{path},t} = \omega \cdot \mathrm{cumsum}(\Delta_t)$ |
| Process noise $Q$ | known sensor/model spec | implicit; never explicitly tracked |
| Observation model | linear, $H$ given | $\hat z_t = \pi \cdot \tanh(\mathrm{MLP}(\text{content}))$ — learned inverse model |
| Measurement noise $R$ | known sensor spec | per-token learned: $R_t = \exp(\mathrm{MLP}(\text{content}))$ |
| Prior covariance | tracked through predict-update | learnable *constant* $\Pi$ (Level 1.5's central simplification) |
| Innovation $\nu_t$ | $y_t - H\hat x_{t\mid t-1}$ | $\mathrm{atan2}(\sin(\hat z_t - \theta_{\mathrm{path},t}), \cos(\hat z_t - \theta_{\mathrm{path},t}))$ |
| Kalman gain | derived from $P, R$ | $K_t = \Pi/(\Pi + R_t)$ — same algebraic form, both learned |
| Update recurrence | sequential | parallel scalar affine scan, $\mathcal{O}(\log T)$ depth |
| Predict ↔ update interleaving | tightly alternating | **decoupled into two separate scans** |

### The conceptual changes

1. **Everything classical Kalman knows, Level 1.5 learns.** $F$, $H$, $Q$,
   $R$ are gradient-descent-trained end-to-end on next-token loss. The
   "Kalman" part is the *algebraic shape* — weighted blend with weight
   $\Pi/(\Pi+R)$ — and the *Lie-group structure* of the wrap.

2. **The covariance recurrence is dropped.** Replaced with a single
   learnable constant $\Pi$. Justification: Level 2 (which does track
   $\Pi_t$ via Möbius scan) was strictly more expressive but performed
   measurably worse. $\Pi_t$ only varied $\sim 4\times$ across tokens
   anyway.

3. **The "observation" is the content embedding.** $\hat z_t$ is a
   learned scalar in $[-\pi, \pi]$ from the token embedding. Closer to a
   learned likelihood than a real measurement.

4. **Predict and update are decoupled scans, not interleaved steps.** On
   $\mathrm{SO}(2)$ with constant $\Pi$ and group-additive corrections,
   you can compute *all* path-integrated $\theta_{\mathrm{path},t}$ first
   via cumsum, *then* compute *all* correction offsets $d_t$ via a
   scalar affine scan, and add. Both $\mathcal{O}(\log T)$.

5. **Lie-group correctness from the wrap.** Markovic et al. 2017 prove
   that on $\mathrm{SO}(2)$, EKF with $\mathrm{atan2}$-wrapped innovations
   is *exactly* the Lie-Group EKF.

---

## 6. Mapping predict-step quantities ($F$, $B$, $u_t$) to MapFormer

### What's the state, what's the input?

Concrete choices for MapFormer:

- **State** $x_t = \theta_t \in \mathbb{R}$, the angle on $\mathrm{SO}(2)$
  (per head, per block — each scalar channel evolves independently).
- **Control input** $u_t = \Delta_t$, the *Lie-algebra increment* induced
  by the current action token.
- **Process noise** $w_t = 0$ in vanilla MapFormer.

### What's $F$, what's $B$?

The path-integration recurrence:
$$\theta_t = \theta_{t-1} + \omega \cdot \Delta_t$$

Match against $x_t = F x_{t-1} + B u_t$:
- $\boxed{F = 1}$. **Pure integrator.** Right model for orientation: if
  you don't move, your heading doesn't change. The Lie-group structure
  forces $F=1$ (group-additive in the Lie algebra).
- $\boxed{B = \omega}$, the geometric frequency schedule. Learnable but
  time-invariant — same value at every timestep regardless of token.

### Where action-dependence hides

$u_t = \Delta_t = f_\Delta(\text{token}_t)$ — a learned function of the
token. The structure is:
$$\theta_t = \theta_{t-1} + \omega \cdot f_\Delta(\text{token}_t)$$

**$B$ does not depend on the action; $u_t$ does.**

This matters because:
1. Linear-Gaussian Kalman requires linearity in $u_t$. If $B = B(u_t)$,
   you've left the regime where $K = P/(P+R)$ is right.
2. $\omega$ and $f_\Delta$ play conceptually different roles: $\omega$ is
   *frequency allocation* (per-block rate), $f_\Delta$ is *direction
   allocation* (action symbol → Lie-algebra direction). Collapsing them
   hides why path integration parallelises.
3. The cumsum structure depends on $\omega$ being constant in time — you
   can pull it out of the cumsum.

### Why $F = 1$ specifically

Dynamics on $\mathrm{SO}(2)$ are *group-additive in the Lie algebra*.
Composing rotations $\theta_a$ and $\theta_b$ gives $\theta_a + \theta_b$.
That's exactly why path integration can be a parallel cumsum. $\mathrm{SE}(3)$
would *not* have $F = 1$ — translation and rotation couple, and the
equivalent of $F$ is the Adjoint $\mathrm{Ad}(R_t)$. Generalising Level 1.5
beyond $\mathrm{SO}(2)$ requires Adjoint-weighted scans (Sola 2018).

### Where action-conditioned $B$ would live: Mamba

Mamba's selective SSM:
$$h_t = A_t h_{t-1} + B_t u_t, \qquad A_t, B_t = f(\text{token}_t)$$
That's "selective" in *selective SSM*. Strictly more expressive but the
recurrence is no longer a simple cumsum.

The MapFormer paper (Appendix A.5) and our MambaLike baseline (~0.57
accuracy across configs) show: **giving the model that extra freedom in
$B_t$ doesn't help on cognitive-map learning at our scale.**
Fixed-$B = \omega$ is more discoverable from gradient descent than the
selective-SSM parameterisation.

---

## 7. Notation: $f_\Delta$ vs $\Delta_t$

- $\text{token}_t$ — symbolic input at step $t$ (action like `↑` or
  observation like `obs_5`).
- $f_\Delta(\cdot)$ — the **learned function** mapping a token to a
  Lie-algebra increment (token embedding → low-rank `action_to_lie`
  projection $W_{\text{out}} W_{\text{in}}$ with rank-2 bottleneck).
- $\Delta_t$ — the **value** of that function at step $t$, i.e.
  $\Delta_t \equiv f_\Delta(\text{token}_t) \in \mathbb{R}^{H \times B}$.

So $f_\Delta$ is a fixed piece of architecture; $\Delta_t$ is its output
at step $t$ (varies with the token).

Path integration in this notation:
$$\theta_t = \theta_{t-1} + \omega \cdot \Delta_t, \qquad \Delta_t = f_\Delta(\text{token}_t)$$

Mapping to Kalman form $x_{t+1} = F x_t + B u_t + w_t$:
| Symbol | Value | Role |
|---|---|---|
| $F$ | $1$ | identity dynamics on $\mathrm{SO}(2)$ |
| $B$ | $\omega$ | constant per-block frequency gain |
| $u_t$ | $\Delta_t$ | input fed into the predict step |
| $w_t$ | $0$ | no explicit process noise |

---

## 8. Both action and observation tokens produce $\Delta_t$ — how does that fit $x_t$ vs $u_t$?

**Every token contributes a predict-step input *and* an update-step
measurement; the model learns by gradient descent which contribution to
suppress where.**

### Mechanically

The trajectory $s = (a_1, o_1, a_2, o_2, \ldots)$ is one unified token
stream. Every token flows through the *same* three computations:

1. $\Delta_t = f_\Delta(\text{token}_t)$ — feeds the **predict** step
   (cumsum'd into $\theta_{\text{path}}$).
2. $\hat z_t = \pi \tanh(f_z(\text{token}_t))$ — the **measurement** at
   step $t$.
3. $R_t = \exp(f_R(\text{token}_t))$ — the **measurement noise** at step
   $t$.

The model has no structural switch saying "this token is an action, run
only predict." Both pathways fire. Magnitudes of $\Delta_t$ and $R_t$
determine which side dominates.

### Mapping back to Kalman

In classical Kalman the streams are separate: $u_t$ from one channel, $y_t$
from another. Here both come from the same token stream — but the "type"
of step is encoded in *learned* magnitudes:

| Token type | What model should learn | Effect |
|---|---|---|
| Action token | $\Delta_t$ large, $R_t$ large | predict dominates |
| Aliased obs (informative-ish) | $\Delta_t \approx 0$, $R_t$ moderate | partial update |
| Landmark obs (highly informative) | $\Delta_t \approx 0$, $R_t$ small | update dominates: $\hat z_t$ snaps $\theta$ |
| Blank obs (uninformative) | $\Delta_t \approx 0$, $R_t$ large | both near no-ops |

### This is how real robotic filtering works

Textbook "alternate predict / update" is a presentation convention, not a
mathematical requirement. Real systems have IMU at 200Hz (predict) and GPS
at 1Hz (update), arriving asynchronously. Standard handling:
- Always run predict at every timestep.
- When measurement available, also run update; otherwise treat $R = \infty$,
  so $K = 0$, so update is a no-op.

MapFormer's unified stream does exactly this, with the
"is-a-measurement-available-and-how-informative" decision *learned*
instead of hardcoded.

### Filter equations are unchanged

At every step:
$$\theta_{\text{path},t} = \theta_{\text{path},t-1} + \omega \cdot \Delta_t \quad \text{(predict)}$$
$$d_t = (1 - K_t) d_{t-1} + K_t \cdot \nu_t \quad \text{(update)}$$
$$\hat\theta_t = \theta_{\text{path},t} + d_t$$

with $\Delta_t, \hat z_t, R_t$ all functions of the same $\text{token}_t$.

**One-sentence framing: MapFormer treats the predict-step input and the
update-step measurement as two different views of the same token, and lets
the network learn how much weight to give each view per token.**
