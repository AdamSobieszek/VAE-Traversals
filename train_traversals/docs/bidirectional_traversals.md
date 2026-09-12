# Bidirectional Traversals with one shared scalar network

Implemented in `lib/TraversalPDE.py`. No training run was performed. Both new
architectures have exactly the original trainable parameters and state-dict
keys. The linear direction and every nonlinear layer are shared across signs.
The time sign is a numerical argument, not an embedding or another network.

## The finite-step issue

Write `h = |dt|`, `s ∈ {-1,+1}`, and suppress the traversal index k. Let

\[
g(x)=\nabla F(x),\quad q=g^Tg,\quad
I_\epsilon(g)=\frac{g}{q+\epsilon},\quad v(x)=I_\epsilon(g(x)).
\]

Euler uses `T_s(x)=x+s h v(x)`. Its forward/backward composition has leading
defect `-h² Dv(x)v(x)`. Sharing F and simply negating its gradient therefore
does not make these discrete maps inverses.

The useful fact is that spherical inversion is itself a gradient **in g**:

\[
H_\epsilon(g)=\tfrac12\log(g^Tg+\epsilon),\qquad
\nabla_g H_\epsilon=I_\epsilon(g).
\]

In particular its Jacobian `A = D_g I_epsilon` is symmetric. This is the
integrability property needed to construct directional scalar potentials;
we do not have to abandon the endpoint-potential contract.

One can even derive the implicit inverse of the original Euler step: solve
`y=x+h v(x)` for x, then define

\[
F_-(y)=-F(x)-h\,[g(x)^Tv(x)-H_\epsilon(g(x))].
\]

Differentiating gives `dF_-=-g(x)^T dy`, hence
`grad F_-(y)=-g(x)` and the backward step recovers x. This construction is
asymmetric in computation: one direction is explicit and the other implicit.
The implemented midpoint construction splits this implicit coordinate change
equally between the two directions.

## `midpoint`: structural inverse maps and scalar potentials

For an endpoint x and direction s, find m from

\[
m=x+\frac{s h}{2}v(m),\qquad
T_s(x)=x+s h v(m).
\]

Thus m is the midpoint of the segment. Equivalently, define

\[
A_s(m)=m-\frac{s h}{2}v(m),\quad
B_s(m)=m+\frac{s h}{2}v(m),\quad
T_s=B_s\circ A_s^{-1}.
\]

Since `A_-s=B_s` and `B_-s=A_s`, **`T_-s = T_s^{-1}`** on a domain where
these coordinate maps are invertible and the solver selects the corresponding
branch. The deterministic discrete path is retraced, including at finite h.
This is not a claim that midpoint exactly follows the continuous ODE solution.

Here is an explicit scalar generating potential for this map:

\[
\boxed{F_s(x)=sF(m)-\frac h2\left[g(m)^Tv(m)-H_\epsilon(g(m))\right].}
\]

For epsilon=0 this simplifies, up to an x-independent constant, to

\[
F_s(x)=sF(m)+\frac h4\log\|g(m)\|^2.
\]

Proof: because `dx=dm-s h A dg/2` and
`d(g^T v-H_epsilon)=g^T A dg`,

\[
dF_s=s g^Tdm-\frac h2 g^TA\,dg=s g^Tdx.
\]

Consequently

\[
\boxed{\nabla_x F_s(x)=s\nabla F(m),\qquad
T_s(x)=x+h I_\epsilon(\nabla_x F_s(x)).}
\]

This is precisely the directional potential interface. At `h=1, epsilon=0`
it gives the two equations in the request. At the same x, the two midpoints
normally differ, so `grad^+ F_-(x)` need not equal `-grad^+ F_+(x)`. At paired
endpoints of one segment they use the same m and cancel exactly.

For epsilon>0, the implementation retains the nonconstant rational term:
`F_s=s F(m)+h log(q+epsilon)/4-h q/(2(q+epsilon))`. Omitting it would give
an incorrect scalar-potential identity near the numerical regularizer.

### Backpropagation and actual costs

The forward fixed-point iterations run without retaining their autograd
history. Backward reconstructs one small potential graph at the solved m.
For `a=s h/2` and upstream cotangent u for m it solves

\[
(I-aD_mv(m))^T\lambda=u,
\quad \frac{\partial L}{\partial x}=\lambda,
\quad \frac{\partial L}{\partial\theta}
=\left(\partial_\theta v(m)\right)^T a\lambda.
\]

These are vector-Jacobian products; no dense D-by-D Jacobian or Hessian is
formed. Direct parameter dependence of the final `g(m)` evaluation is handled
by ordinary autograd and added to the implicit contribution. Signs and h
retain their tensor broadcasting and h retains its derivative.

This is implicit-layer differentiation, not a detached-midpoint or
straight-through approximation. Solver-history memory is constant in the
number of iterations, although potential graphs across rollout steps still
consume memory. Compared with Euler there are extra potential evaluations and
potential VJPs, plus device synchronizations for convergence checks. There is
no extra generator/recognizer evaluation or backward traversal, and no cycle
loss. Runtime optimality and training quality have not been established.

The custom implicit backward supports ordinary first-order parameter training
through the explicitly supplied input gradient. Differentiating through that
custom backward again is unsupported. Accordingly midpoint rejects PDE losses
other than `BB`, `g2orth`, and `signed_g2orth` (plus the `epsilon` setting).
This prevents silently miscomputing higher-derivative PINN losses. Do not
obtain its training input gradient by doing a second backward through the
scalar value; use `value_and_grad` / `PDEState.f_grad`, as the trainer does.

### What the structural guarantee requires

A sufficient fixed-point condition is `h ||Dv||/2 < 1` on a suitable invariant
region. The current network does not enforce this globally. Even a perfectly
valid implicit root can be difficult for Picard iteration; finite-step maps
can also fold and cease to have a single global inverse. A shared arbitrary
MLP cannot remove this issue merely by learning a direction embedding.

Both forward and backward check residuals and raise on nonconvergence. There
is no silent Euler fallback and no silently shortened step. The default cap
is 80 iterations, with absolute and relative tolerances 1e-6; most easy solves
stop much earlier. Increasing the cap helps slow contraction, not divergence.
Residual tolerances are not direct bounds on round-trip error: conditioning
amplifies them. Small MPS checks include a slowly contracting field with
round-trip error around 2.3e-5 at these tolerances.

For an unsuccessful solve, inspect gradient norms and local curvature, reduce
dt, or explicitly choose the approximate architecture. Independently adapting
dt at opposite endpoints would generally destroy the claimed inverse identity.
For exact comparisons use the same h, unchanged weights, `eval()` (or
`noise_aperture=0`), and corresponding solution branches. Independently sampled
training noise is not reversible.

## `potential_jet`: explicit approximation using the same weights

Expanding the midpoint generating potential in h gives

\[
\boxed{F_s^{\rm jet}(x)=sF(x)+\frac h4\log(\|\nabla F(x)\|^2+\epsilon).}
\]

Thus, with `Q=Hess F` and `r=q+epsilon`,

\[
\nabla F_s^{\rm jet}=s g+\frac h{2r}Qg,
\quad
I_\epsilon(\nabla F_s^{\rm jet})
=s v+\frac h2 Dv\,v+O(h^2).
\]

The induced step matches the second-order flow expansion. Its local step
error is O(h³), and it cancels Euler's leading O(h²) reversal defect. For
smooth fields and fixed h, the signed family has a forward/backward defect
O(h⁴); this is a local asymptotic statement, not finite-step invertibility.

The correction uses one Hessian-vector product through the small potential.
Training it differentiates that correction, including mixed higher
derivatives. It has no solver, extra head, auxiliary target, or cycle penalty.
It may be faster than midpoint, but that should be measured on the CUDA VM.
At large curvature or near small q it can be inaccurate; neither architecture
is presented as an empirically validated replacement for training.

This also explains the desired late-training sharing. For epsilon negligible,
scaling `F -> cF` scales the base velocity as `1/c`, but the leading asymmetric
velocity correction as `1/c²`. There is no epoch schedule or learned sign
gate: the correction shrinks with the actual local dynamics. Small displacement
alone does not prove small error if curvature simultaneously grows; the
relevant quantity is displacement relative to the spatial variation scale.

## Training integration and use

`train.py --bidirectional` samples independent Rademacher signs `[B,K,1]` once
per rollout, leaves the sampled x0 shared over K, and multiplies both historical
and adjacent antisymmetric logits by the flattened signs `[B*K,1]`. Every
class retains label k in both directions. This balances time orientation in
expectation; it does not mathematically guarantee uniform spatial coverage.
The positive-Gram orthogonality penalty removes the sampled sign before
comparing heads. Antisymmetric logits alone are an inductive bias, not a proof
that all semantically duplicate traversals are excluded.

The legacy `euler` mode remains the default, with its formerly ignored
`direction` fixed. `--bidirectional --traversal-architecture euler` is a useful
sign-negation ablation. `dt` and `direction` accept scalar, `[B]`, `[B,1]`,
`[1,K]`, `[B,K]`, or broadcastable `[B,K,1]` values. A 1-D value means B, not K;
use `[1,K,1]` for per-head signs. The last dimension must be one. Signed dt is
still supported for existing inference visualizers. Direction must be ±1.

On the CUDA VM, from the repository's `train_traversals` directory:

```bash
TRAVERSAL_ARCHITECTURE=midpoint BIDIRECTIONAL=true bash scripts/SNGAN_train.sh
TRAVERSAL_ARCHITECTURE=potential_jet BIDIRECTIONAL=true bash scripts/SNGAN_train.sh
```

Extra CLI arguments can follow the shell script, e.g. `--traversal-noise 0`
or `--midpoint-max-iter 120`. The baseline script still enables its existing
noise unless explicitly overridden. `--potential-hidden` defaults to the
existing train.py width of 32 and is saved alongside the architecture and
solver settings in experiment args. `gen_pairs.py` and
`traverse_latent_space.py` restore these settings. A bare state dict intentionally
has unchanged keys: when loading one directly, supply its architecture and
hidden width yourself. No automatic compatibility claim is made for changing
an existing optimizer/scheduler experiment into a different method; start a
new experiment for comparisons.

Python interface:

```python
S = TraversalPDE(K, T, D, n_hidden=32, architecture="midpoint",
                 lambdas={"BB": .25, "signed_g2orth": 1.}, noise_aperture=0)
signs = torch.randint(0, 2, (B, K, 1), device=z.device).float() * 2 - 1
preds, left, right, pde_loss, delta_f = S(z, t_index, dt, direction=signs)
S.eval()
x, forward_delta = S.inference(z, dt=dt, direction=signs)
y, backward_delta = S.inference(x + forward_delta, dt=dt, direction=-signs)
# Compare y + backward_delta with x, accounting for solver conditioning.
```

## Validation and references

The small MPS tests cover scalar-gradient identities (including epsilon=.01),
mixed-sign reversal, explicit vs unrolled implicit parameter/input/dt
derivatives, jet error reduction, fixed-sign multi-step rollouts, the signed
Gram correction, checkpoint parameter parity, sign/logit ordering, and solver
failure. A separate batch-1, K=2 SNGAN AnimeFaces/LeNet check backpropagates
recognizer supervision through each architecture without an optimizer step.
It verifies nonzero finite linear and nonlinear gradients, frozen generator
weights, and the same synthesis batch sizes `[1,2,2]`.

The recognizer-only gradient norms in that smoke check were:

| Architecture | Linear weights | First nonlinear layer |
| --- | ---: | ---: |
| Euler | 44.8823 | 10.6333 |
| Potential jet | 42.0606 | 10.3173 |
| Midpoint | 42.4330 | 10.3758 |

These are connectivity/numerical checks at initialization, not evidence of
better learning or loss convergence. Run the focused checks with:

```bash
/opt/anaconda3/envs/manip311/bin/python -m pytest -q \
  tests/test_bidirectional_sngan.py tests/test_bidirectional_traversal.py \
  tests/test_traversal_optimizations.py tests/test_traversal_revision.py \
  -k 'not gaussian_cone'
```

The three existing `gaussian_cone` cases are excluded because they assert a
transverse cone kernel against the pre-existing isotropic Gaussian function.
This change preserves that existing noise kernel; the new reversal tests are
deterministic. On this machine PyTorch tests must run outside the sandbox so
the MPS backend is visible.

The standard midpoint and generating-function background is in
[Hairer's geometric integration notes](https://people.math.ethz.ch/~hiptmair/Seminars/PHS_24/Hairer_LectureHamiltonianSystems.pdf)
and [Cohen's geometric integration course](https://www.math.chalmers.se/~cohend/Teach/gni-Cohen.pdf).
The inverse and midpoint scalar formulas above are the derivation for this
repository's specific spherical-inversion field, rather than a claim that a
source already publishes this Traversal architecture.

For memory-efficient implicit derivatives see the authors' tutorial on
[implicit functions and automatic differentiation](https://implicit-layers-tutorial.org/implicit_functions/).
[Invertible Residual Networks](https://arxiv.org/abs/1811.00995) provides related
context for contraction-based inversion. The original potential-flow
baseline is [Song et al. (2023)](https://arxiv.org/abs/2304.12944).
The construction uses established mathematical and DL techniques; no unsupported
claim of a new 2026 state of the art or globally optimal runtime is intended.
