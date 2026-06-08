"""
Non-Markovian collective motion with self-regulated perceptual dynamics
in NON-PERIODIC domains, for robotic systems and drone swarms.

Extension of:
    J. Beuria, "Non-Markovian Collective Motion from Self-Regulated
    Perceptual Dynamics", arXiv:2510.23688v2 (physics.soc-ph).

WHAT IS KEPT FROM THE PAPER
---------------------------
Each agent carries two internal registers evolving on separated timescales:

  * a FAST perceptual register, one two-state channel (i,j) per neighbour j,
    whose longitudinal Bloch component m_z^{ij} in [-1,1] is the *resolved*
    tendency to align with neighbour j;
  * a SLOW regulatory variable s_i (the longitudinal slow Bloch component
    s_z^i), which integrates recent alignment and feeds back as an internal
    gain on the fast channels.

The fast channels are propagated with a driven-dissipative (Bloch / GKSL)
update: transverse switching at rate Gamma, dephasing 1/T2, and longitudinal
relaxation 1/T1 toward a field-dependent equilibrium.  The longitudinal field
on channel (i,j) is

    h_z^{ij} = kappa * s_i        (slow -> fast gain,  H_sp in the paper)
             + g_align * (e_i . e_j)   (channel-channel consistency / persistence)
             + h_ext^{ij}          (external drive P(X(t)) in the paper)

The slow variable obeys a relaxation (memory horizon 1/gamma_s) driven by the
agent's mean resolved alignment -- this is the fast -> slow half of the loop.
In the fast-relaxation / weak-feedback limit (kappa, T1, T2 -> 0) this reduces
to Vicsek-type alignment, exactly as in the paper's Appendix C.

(The exact closed forms of the heading map and the fast->slow feedback are
reconstructed from the paper's described structure and its stated Vicsek limit;
the two-register Bloch core, the GKSL dissipators, and the slow->fast gain are
taken directly from Eqs. (1)-(12).)

WHAT IS NEW (the non-periodic extension)
----------------------------------------
  * boundary modes: 'soft' (repulsive walls), 'reflect' (billiard), 'open'
    (+cohesion), and 'periodic' (recovers the original paper as a sanity check);
  * circular obstacles and a hard short-range separation zone (collision
    avoidance) -- both expressed as steering cues that enter the SAME internal
    machinery via the external-drive term P(X(t));
  * a bounded turn rate omega_max (finite actuation);
  * a dimension-agnostic heading map: dim=2 (ground robots) or dim=3 (drones),
    the latter with an altitude band;
  * the slow regulatory variable additionally gates wall/obstacle avoidance,
    i.e. "adaptive caution" near boundaries -- the controller-gain reading the
    paper suggests for robotic swarms.

Author of extension: prepared for the user. MIT-style use.
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  small geometry helpers (work for dim = 2 or 3)
# --------------------------------------------------------------------------- #
def _normalize(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _rotate_toward(e, target, max_angle):
    """Rotate unit vectors `e` toward unit vectors `target` by at most
    `max_angle` (per agent).  Dimension-general (great-circle / SLERP step)."""
    dot = np.clip(np.sum(e * target, axis=1), -1.0, 1.0)
    ang = np.arccos(dot)                      # full angle to target
    step = np.minimum(ang, max_angle)
    # component of target orthogonal to e
    perp = target - (dot[:, None]) * e
    perp = _normalize(perp)
    moving = ang > 1e-9
    out = e.copy()
    out[moving] = (np.cos(step[moving])[:, None] * e[moving]
                   + np.sin(step[moving])[:, None] * perp[moving])
    return _normalize(out)


# --------------------------------------------------------------------------- #
#  main model
# --------------------------------------------------------------------------- #
class SelfRegulatedSwarm:
    def __init__(
        self,
        N=200,
        dim=2,
        box=(50.0, 50.0),           # arena size (Lx, Ly[, Lz])
        v0=1.0,                     # fixed speed
        R=4.0,                      # perceptual / sensing radius
        # ---- internal two-register dynamics (from the paper) ----
        Gamma=1.0,                  # fast transverse switching rate (H_p)
        T1=0.6,                     # fast longitudinal relaxation time
        T2=0.3,                     # fast transverse (dephasing) time
        kappa=1.2,                  # slow -> fast gain (H_sp feedback)
        g_align=2.0,                # channel-channel consistency weight
        gamma_s=0.05,               # slow relaxation rate (memory ~ 1/gamma_s)
        lam_fb=2.0,                 # fast -> slow feedback gain
        s_base=0.0,                 # slow-register baseline
        eta=0.15,                   # heading noise amplitude
        # ---- actuation ----
        omega_max=0.6,              # max turn rate (rad / time unit)
        # ---- non-periodic extension ----
        boundary='soft',            # 'soft' | 'reflect' | 'open' | 'periodic'
        wall_margin=6.0,            # distance at which walls start pushing
        wall_gain=3.0,              # base wall repulsion strength
        r_sep=1.2,                  # short-range separation (collision avoid.)
        sep_gain=4.0,
        coh_gain=0.4,               # cohesion (used in 'open' mode)
        obstacles=None,             # list of (center_vec, radius)
        alt_band=None,             # (z_lo, z_hi) drone altitude band (dim=3)
        alt_gain=3.0,
        caution=0.6,                # how much slow tone s_i raises avoidance gain
        seed=0,
    ):
        assert dim in (2, 3)
        self.rng = np.random.default_rng(seed)
        self.N, self.dim, self.v0, self.R = N, dim, v0, R
        self.box = np.array(box[:dim], float)
        self.Gamma, self.T1, self.T2 = Gamma, T1, T2
        self.kappa, self.g_align = kappa, g_align
        self.gamma_s, self.lam_fb, self.s_base = gamma_s, lam_fb, s_base
        self.eta, self.omega_max = eta, omega_max
        self.boundary = boundary
        self.wall_margin, self.wall_gain = wall_margin, wall_gain
        self.r_sep, self.sep_gain, self.coh_gain = r_sep, sep_gain, coh_gain
        self.obstacles = obstacles or []
        self.alt_band, self.alt_gain = alt_band, alt_gain
        self.caution = caution

        # state -----------------------------------------------------------
        self.x = self.rng.uniform(0.2, 0.8, size=(N, dim)) * self.box
        self.e = _normalize(self.rng.normal(size=(N, dim)))      # heading unit vec
        self.s = np.full(N, s_base, float)                       # slow register s_i
        # fast Bloch components per directed pair (i,j); dense NxN is fine for N<~1k
        self.mz = np.zeros((N, N))    # resolved align bias  in [-1,1]
        self.mx = np.zeros((N, N))    # unresolved competition
        self.my = np.zeros((N, N))
        self._mz_inf = (1.0 / T1) * 0.0   # bare bias (kept 0: symmetric a priori)

    # ------------------------------------------------------------------ #
    #  neighbour graph  (NO minimum-image: this is the non-periodic part)
    # ------------------------------------------------------------------ #
    def _neighbours(self):
        if self.boundary == 'periodic':
            d = self.x[:, None, :] - self.x[None, :, :]
            d -= self.box * np.round(d / self.box)      # minimum image
        else:
            d = self.x[:, None, :] - self.x[None, :, :]  # plain Euclidean
        dist = np.linalg.norm(d, axis=2)
        adj = (dist < self.R) & (dist > 1e-9)
        return adj, d, dist

    # ------------------------------------------------------------------ #
    #  external-drive / steering cues  -> these are the P(X(t)) channels
    # ------------------------------------------------------------------ #
    def _avoidance_vector(self, d, dist, adj):
        """Sum of steering cues (walls, obstacles, separation, altitude,
        cohesion) returned as a per-agent vector to be added to the desired
        heading.  Gated per-agent by (1 + caution * sigmoid(s_i)) so that a
        high regulatory tone increases avoidance gain -- adaptive caution."""
        N, dim = self.N, self.dim
        avoid = np.zeros((N, dim))

        # ---- walls (skip for periodic / open) ----
        if self.boundary in ('soft', 'reflect'):
            for k in range(dim):
                if self.alt_band is not None and dim == 3 and k == 2:
                    continue  # altitude handled separately as a band
                near_lo = self.x[:, k] < self.wall_margin
                near_hi = self.x[:, k] > self.box[k] - self.wall_margin
                # push inward, strength grows as you approach the wall
                f_lo = self.wall_gain * np.clip(
                    1 - self.x[:, k] / self.wall_margin, 0, 1)
                f_hi = self.wall_gain * np.clip(
                    1 - (self.box[k] - self.x[:, k]) / self.wall_margin, 0, 1)
                avoid[near_lo, k] += f_lo[near_lo]
                avoid[near_hi, k] -= f_hi[near_hi]

        # ---- altitude band for drones ----
        if dim == 3 and self.alt_band is not None:
            z_lo, z_hi = self.alt_band
            below = self.x[:, 2] < z_lo
            above = self.x[:, 2] > z_hi
            avoid[below, 2] += self.alt_gain * (z_lo - self.x[below, 2])
            avoid[above, 2] -= self.alt_gain * (self.x[above, 2] - z_hi)

        # ---- circular / spherical obstacles ----
        for c, rad in self.obstacles:
            c = np.asarray(c, float)[:dim]
            rel = self.x - c
            rdist = np.linalg.norm(rel, axis=1)
            hit = rdist < (rad + self.wall_margin)
            if np.any(hit):
                strength = self.wall_gain * np.clip(
                    1 - (rdist[hit] - rad) / self.wall_margin, 0, 1)
                avoid[hit] += strength[:, None] * _normalize(rel[hit])

        # ---- short-range separation (collision avoidance between agents) ----
        close = adj & (dist < self.r_sep)
        if np.any(close):
            # repel along -d (d_ij = x_i - x_j); weight ~ (1 - dist/r_sep)
            w = np.where(close, self.sep_gain * (1 - dist / self.r_sep), 0.0)
            avoid += np.einsum('ij,ijk->ik', w, _normalize(d))

        # ---- cohesion toward local centre of mass (open domain only) ----
        if self.boundary == 'open':
            cnt = adj.sum(1)
            has = cnt > 0
            com = np.zeros((N, dim))
            com[has] = (adj @ self.x)[has] / cnt[has, None]
            avoid[has] += self.coh_gain * (com[has] - self.x[has])

        # adaptive caution: high regulatory tone -> stronger avoidance
        gate = 1.0 + self.caution * (1.0 / (1.0 + np.exp(-self.s)))
        return avoid * gate[:, None]

    # ------------------------------------------------------------------ #
    #  one integration step
    # ------------------------------------------------------------------ #
    def step(self, dt=0.05, h_ext=None):
        N = self.N
        adj, d, dist = self._neighbours()

        # ---- fast register: longitudinal field on each channel (i,j) ----
        # consistency term  e_i . e_j  (reinforces aligning with like-headed nb.)
        cons = self.e @ self.e.T                       # NxN in [-1,1]
        hz = (self.kappa * self.s[:, None]             # slow -> fast gain (H_sp)
              + self.g_align * cons)                   # channel-channel consistency
        if h_ext is not None:                          # external drive P(X(t))
            hz = hz + h_ext
        hz = np.where(adj, hz, 0.0)
        hx = self.Gamma                                # transverse switching

        # ---- driven-dissipative Bloch update (precession + GKSL dissipation) ----
        # precession  m_dot = 2 (h x m),  h = (hx, 0, hz); the amplitude-damping
        # channels relax m_z toward a FIELD-DEPENDENT equilibrium tanh(h_z)
        # (positivity preserving, in [-1,1]) -- this is the resolved align bias.
        mz_eq = np.tanh(hz)
        mx, my, mz = self.mx, self.my, self.mz
        dmx = -2 * hz * my - mx / self.T2
        dmy = 2 * (hz * mx - hx * mz) - my / self.T2
        dmz = 2 * hx * my - (mz - mz_eq) / self.T1
        self.mx = np.where(adj, mx + dt * dmx, 0.0)
        self.my = np.where(adj, my + dt * dmy, 0.0)
        self.mz = np.clip(np.where(adj, mz + dt * dmz, 0.0), -1.0, 1.0)

        # ---- heading map: resolved weights w_ij = (m_z+1)/2 in [0,1] ----
        w = np.where(adj, 0.5 * (self.mz + 1.0), 0.0)
        D = w @ self.e                                 # weighted neighbour heading
        D = D + self._avoidance_vector(d, dist, adj)   # + steering cues
        # agents with no input keep their heading
        no_input = np.linalg.norm(D, axis=1) < 1e-9
        D[no_input] = self.e[no_input]
        target = _normalize(D)

        # heading noise (small random rotation), then bounded turn toward target
        noise = self.rng.normal(scale=self.eta, size=(N, self.dim))
        target = _normalize(target + noise)
        self.e = _rotate_toward(self.e, target, self.omega_max * dt)

        # ---- slow register: fast->slow feedback + relaxation (memory) ----
        Mi = np.where(adj, self.mz, 0.0).sum(1) / np.maximum(adj.sum(1), 1)
        s_target = np.tanh(self.lam_fb * Mi)
        self.s = self.s + dt * (-self.gamma_s * (self.s - self.s_base)
                                + self.gamma_s * s_target)

        # ---- move, then enforce boundary ----
        self.x = self.x + dt * self.v0 * self.e
        self._apply_boundary()

        return self.observables()

    def _apply_boundary(self):
        if self.boundary == 'periodic':
            self.x = np.mod(self.x, self.box)
        elif self.boundary == 'reflect':
            for k in range(self.dim):
                lo = self.x[:, k] < 0
                hi = self.x[:, k] > self.box[k]
                self.x[lo, k] = -self.x[lo, k]
                self.e[lo, k] = -self.e[lo, k]
                self.x[hi, k] = 2 * self.box[k] - self.x[hi, k]
                self.e[hi, k] = -self.e[hi, k]
            self.e = _normalize(self.e)
        else:  # 'soft' / 'open': clamp as a safety net (repulsion does the work)
            self.x = np.clip(self.x, 0.0, self.box)

    # ------------------------------------------------------------------ #
    def observables(self):
        polar = np.linalg.norm(self.e.mean(0))         # global polar order
        tone = self.s.mean()                           # mean regulatory tone
        return {"polar": float(polar), "tone": float(tone)}

    def run(self, steps=2000, dt=0.05, record_every=1, h_ext=None):
        P, S = [], []
        for t in range(steps):
            o = self.step(dt=dt, h_ext=h_ext)
            if t % record_every == 0:
                P.append(o["polar"]); S.append(o["tone"])
        return np.array(P), np.array(S)


# --------------------------------------------------------------------------- #
#  demos / figures
# --------------------------------------------------------------------------- #
def demo_bounded_arena(save="fig_bounded.png"):
    """2D bounded arena with soft walls + two obstacles; show trajectories
    coloured by the slow regulatory tone (edge agents charge up differently)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obstacles = [(np.array([18.0, 30.0]), 4.0),
                 (np.array([35.0, 16.0]), 5.0)]
    sim = SelfRegulatedSwarm(N=250, dim=2, box=(50, 50), boundary='soft',
                             obstacles=obstacles, seed=3)
    # warm up
    sim.run(steps=600, dt=0.05)
    # collect a snapshot + per-agent tone
    P, S = sim.run(steps=200, dt=0.05)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    sc = ax[0].quiver(sim.x[:, 0], sim.x[:, 1], sim.e[:, 0], sim.e[:, 1],
                      sim.s, cmap="coolwarm", clim=(-1, 1),
                      scale=40, width=0.004)
    for c, r in obstacles:
        ax[0].add_patch(plt.Circle(c, r, color="k", alpha=0.35))
    ax[0].set_xlim(0, 50); ax[0].set_ylim(0, 50); ax[0].set_aspect("equal")
    ax[0].set_title("Bounded arena (soft walls + obstacles)\n"
                    "arrows = heading, colour = slow regulatory tone $s_i$")
    fig.colorbar(sc, ax=ax[0], shrink=0.8, label=r"$s_i$")

    # distance-to-boundary vs tone  -> spatial heterogeneity of memory variable
    d_edge = np.minimum.reduce([sim.x[:, 0], sim.x[:, 1],
                                50 - sim.x[:, 0], 50 - sim.x[:, 1]])
    ax[1].scatter(d_edge, sim.s, s=12, alpha=0.6)
    ax[1].set_xlabel("distance to nearest wall")
    ax[1].set_ylabel(r"regulatory tone $s_i$")
    ax[1].set_title("Regulatory tone is spatially heterogeneous\n"
                    "(only visible once periodicity is broken)")
    fig.tight_layout(); fig.savefig(save, dpi=130)
    print("saved", save)


def demo_hysteresis(save="fig_hysteresis.png"):
    """Sweep the slow->fast feedback gain kappa up then down and record the
    steady polar order: the paper's feedback-induced hysteresis, reproduced
    here under non-periodic (soft-wall) boundary conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kappas_up = np.linspace(0.0, 3.0, 11)
    kappas = np.concatenate([kappas_up, kappas_up[::-1]])
    sim = SelfRegulatedSwarm(N=220, dim=2, box=(32, 32), boundary='soft',
                             gamma_s=0.02, eta=0.12, seed=7)
    sim.run(steps=400, dt=0.05)               # equilibrate
    branch = []
    for kp in kappas:
        sim.kappa = kp
        P, S = sim.run(steps=350, dt=0.05)    # let it settle at this kappa
        branch.append(P[-120:].mean())
    branch = np.array(branch)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    n = len(kappas_up)
    ax.plot(kappas[:n], branch[:n], "o-", label="increasing feedback")
    ax.plot(kappas[n:], branch[n:], "s--", label="decreasing feedback")
    ax.set_xlabel(r"slow$\to$fast feedback gain $\kappa$")
    ax.set_ylabel("steady polar order")
    ax.set_title("Feedback-induced hysteresis (non-periodic domain)")
    ax.legend(); fig.tight_layout(); fig.savefig(save, dpi=130)
    print("saved", save)


def demo_drone_3d(save="fig_drone3d.png"):
    """3D drone swarm confined to a box with an altitude band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    sim = SelfRegulatedSwarm(N=150, dim=3, box=(40, 40, 25), boundary='soft',
                             v0=1.0, R=5.0, alt_band=(8.0, 18.0),
                             omega_max=0.5, seed=11)
    sim.run(steps=700, dt=0.05)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(sim.x[:, 0], sim.x[:, 1], sim.x[:, 2],
              sim.e[:, 0], sim.e[:, 1], sim.e[:, 2],
              length=2.5, normalize=True, color="steelblue", alpha=0.7)
    ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_zlim(0, 25)
    ax.set_title("3D drone swarm: soft walls + altitude band [8,18]")
    fig.tight_layout(); fig.savefig(save, dpi=130)
    print("saved", save)


if __name__ == "__main__":
    demo_bounded_arena()
    demo_hysteresis()
    demo_drone_3d()
