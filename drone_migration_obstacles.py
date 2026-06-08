"""
Long-distance migration with self-healing coherence.

No waypoints. The flock has a fixed MIGRATORY HEADING (like birds flying south)
and flies indefinitely through open space. Random obstacles stream toward it.
When the flock hits an obstacle it splits and headings diverge -- coherence
(polar order) drops -- but the alignment dynamics plus the slow regulatory
memory pull it back to a common heading, so coherence restores ON ITS OWN.

The migratory heading enters as the model's external drive P(X(t)); obstacle
avoidance, separation, altitude band and cohesion are the same steering cues as
before. There are no walls in the flight direction: the camera follows the
flock's centroid so it can travel arbitrarily far.

Run:
  python3 drone_migration_obstacles.py          # live 3D animation (indefinite)
  python3 drone_migration_obstacles.py --check   # headless coherence report
"""

import itertools
import numpy as np
from swarm_nonperiodic import SelfRegulatedSwarm, _normalize


# --------------------------------------------------------------------------- #
class MigratingFlock(SelfRegulatedSwarm):
    """Open-space flock with a constant migratory heading and free flight
    (no clipping): obstacles are managed externally and stream past."""

    def __init__(self, *args, migration_dir=(0, 1, 0), migration_gain=0.6,
                 corridor=22.0, conf_gain=1.0, center_gain=0.10, **kwargs):
        kwargs.setdefault("boundary", "open")     # cohesion holds it together
        super().__init__(*args, **kwargs)
        self.m_hat = _normalize(np.array(migration_dir, float)[:self.dim]
                                .reshape(1, -1)).ravel()
        self.migration_gain = migration_gain
        self.corridor = corridor          # lateral half-width of the flyway
        self.conf_gain = conf_gain
        self.center_gain = center_gain

    def _avoidance_vector(self, d, dist, adj):
        v = super()._avoidance_vector(d, dist, adj)   # cohesion+obstacle+sep+alt
        ncount = adj.sum(1)
        # constant migratory pull, scaled to stay comparable with alignment sum
        v = v + (self.migration_gain * (1.0 + ncount))[:, None] * self.m_hat
        # soft lateral confinement to an air corridor in x. A gentle always-on
        # pull to the centre-line (x=0) funnels split halves back together so
        # the flock re-merges after a pillar; a stiff wall stops it leaving the
        # corridor. (z is handled by the altitude band.)
        if self.corridor is not None:
            x = self.x[:, 0]
            spring = -self.center_gain * x                      # gentle centring
            over = np.abs(x) - self.corridor
            wall = np.where(over > 0,
                            -np.sign(x) * self.conf_gain * (1.0 + over), 0.0)
            v[:, 0] += spring + wall
        return v

    def _apply_boundary(self):
        pass                                          # free flight in +y

    def polar(self):
        return float(np.linalg.norm(self.e.mean(0)))

    def centroid(self):
        return self.x.mean(0)

    def _adj(self):
        dd = self.x[:, None, :] - self.x[None, :, :]
        dist = np.linalg.norm(dd, axis=2)
        return (dist < self.R) & (dist > 1e-9)

    def local_coherence(self):
        """Mean local alignment: how aligned each agent is with its neighbours.
        Unlike global polar order, this drops when the flock fragments because
        split sub-groups steer differently while manoeuvring."""
        adj = self._adj()
        cnt = adj.sum(1)
        has = cnt > 0
        mean_e = (adj @ self.e)[has] / cnt[has, None]
        if not np.any(has):
            return 1.0
        return float(np.linalg.norm(mean_e, axis=1).mean())

    def largest_cluster_frac(self):
        """Fraction of agents in the largest spatially-connected group
        (connected = within sensing radius). =1 when the flock is whole, drops
        when an obstacle splits it, returns to ~1 when the halves merge."""
        adj = self._adj()
        seen = np.zeros(self.N, bool)
        best = 0
        for s in range(self.N):
            if seen[s]:
                continue
            stack, size = [s], 0
            seen[s] = True
            while stack:
                u = stack.pop()
                size += 1
                nb = np.where(adj[u] & ~seen)[0]
                seen[nb] = True
                stack.extend(nb.tolist())
            best = max(best, size)
        return best / self.N


# --------------------------------------------------------------------------- #
def stream_obstacles(sim, ahead=120.0, behind=45.0,
                     gap_range=(24.0, 34.0), rad_range=(6.0, 9.0),
                     per_row=(2, 4), rng=None):
    """Fill the flyway with a FIXED random forest of obstacle rows spanning the
    corridor, spaced (gap_range) so the flock partially reforms between rows.
    The confined flock cannot avoid the trees, so it repeatedly splits into
    sub-groups (largest-cluster fraction drops) and re-coheres in the gaps."""
    rng = rng or np.random.default_rng()
    cy = sim.centroid()[1]
    half = (sim.corridor or 16.0) - 1.0
    band = sim.alt_band or (sim.box[2] * 0.4, sim.box[2] * 0.6)
    sim.obstacles = [(p, r) for (p, r) in sim.obstacles if p[1] > cy - behind]
    front = max((p[1] for p, _ in sim.obstacles), default=cy)
    while front < cy + ahead:
        front += rng.uniform(*gap_range)
        for _ in range(int(rng.integers(per_row[0], per_row[1] + 1))):
            ox = rng.uniform(-half, half)
            oz = rng.uniform(*band) if sim.dim == 3 else 0.0
            center = np.array([ox, front, oz][:sim.dim], float)
            sim.obstacles.append((center, rng.uniform(*rad_range)))


def build():
    sim = MigratingFlock(
        N=120, dim=3, box=(0, 0, 28),       # box only carries the z scale
        boundary="open",
        v0=2.0, R=9.0,                      # sensing radius
        alt_band=(10.0, 18.0), alt_gain=3.0,
        omega_max=0.40,                     # turn rate
        eta=0.10,                           # heading noise
        coh_gain=0.35,                      # cohesion
        sep_gain=12.0, r_sep=1.8,           # collision avoidance
        wall_gain=16.0, wall_margin=8.0,    # obstacle repulsion
        gamma_s=0.025,                      # slow memory (helps re-cohere)
        migration_dir=(0, 1, 0), migration_gain=0.45,
        corridor=16.0, conf_gain=1.5, center_gain=0.12,  # flyway + mild centring
        caution=0.6, seed=5,
    )
    # start as a corridor-filling flock, mid-altitude
    sim.x = np.column_stack([
        sim.rng.uniform(-14, 14, sim.N),
        sim.rng.uniform(-14, 14, sim.N),
        sim.rng.uniform(12, 16, sim.N),
    ])
    sim.run(steps=150, dt=0.05)             # form a flock before obstacles
    return sim


# --------------------------------------------------------------------------- #
def headless_check(steps=2000, dt=0.05):
    sim = build()
    rng = np.random.default_rng(1)
    loc, clu, wid = [], [], []
    for t in range(steps):
        stream_obstacles(sim, rng=rng)
        sim.step(dt=dt)
        loc.append(sim.local_coherence())
        clu.append(sim.largest_cluster_frac())
        wid.append(sim.x[:, 0].std())
    loc, clu, wid = np.array(loc), np.array(clu), np.array(wid)
    c = sim.centroid()
    print(f"distance travelled: {c[1]:.0f} units in {steps} steps")
    print(f"local coherence   mean={loc.mean():.2f} min={loc.min():.2f}")
    print(f"largest cluster   mean={clu.mean():.2f} min={clu.min():.2f} "
          f"(1.0 = whole flock; <1 = split)")
    below = clu < 0.9
    splits = np.sum((~below[:-1]) & below[1:])
    print(f"split events (cluster <0.9): {splits}  -> each heals back toward 1.0")
    for t in range(0, steps, 200):
        print(f"  t={t:4d}-{t+200:4d}  local-coh min={loc[t:t+200].min():.2f}   "
              f"cluster min={clu[t:t+200].min():.2f} max={clu[t:t+200].max():.2f}"
              f"   width(std)={wid[t:t+200].mean():.1f}")


# --------------------------------------------------------------------------- #
def animate():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from collections import deque

    sim = build()
    rng = np.random.default_rng(1)
    TRAIL, WIN = 18, 45.0
    trails = [deque(maxlen=TRAIL) for _ in range(sim.N)]
    clu_hist = deque(maxlen=240)
    loc_hist = deque(maxlen=240)
    history = []

    fig = plt.figure(figsize=(9.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax_in = fig.add_axes([0.08, 0.70, 0.26, 0.18])     # coherence inset

    def update(_):
        stream_obstacles(sim, rng=rng)
        sim.step(dt=0.05)
        c = sim.centroid()
        frac = sim.largest_cluster_frac()
        loc = sim.local_coherence()
        for i in range(sim.N):
            trails[i].append(sim.x[i].copy())
        clu_hist.append(frac)
        loc_hist.append(loc)
        history.append(sim.x.copy())

        ax.cla()
        ax.set_xlim(c[0] - WIN, c[0] + WIN)
        ax.set_ylim(c[1] - WIN, c[1] + WIN)            # camera follows flock
        ax.set_zlim(0, sim.box[2])
        ax.set_xlabel("x"); ax.set_ylabel("y (migration)"); ax.set_zlabel("alt")
        status = "WHOLE" if frac > 0.9 else "SPLIT"
        ax.set_title(f"migrating flock  |  distance flown = {c[1]:6.0f}  |  "
                     f"largest group = {frac:.0%}  [{status}]")
        # obstacles in view
        for p, r in sim.obstacles:
            if c[1] - WIN < p[1] < c[1] + WIN:
                u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:6j]
                ax.plot_surface(p[0] + r * np.cos(u) * np.sin(v),
                                p[1] + r * np.sin(u) * np.sin(v),
                                p[2] + r * np.cos(v),
                                color="dimgray", alpha=0.45, linewidth=0)
        # trails
        for i in range(sim.N):
            if len(trails[i]) > 1:
                tr = np.array(trails[i])
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=0.5,
                        alpha=0.3, color="steelblue")
        # drones (red when whole, orange-ish when split)
        col = "crimson" if frac > 0.9 else "darkorange"
        ax.quiver(sim.x[:, 0], sim.x[:, 1], sim.x[:, 2],
                  sim.e[:, 0], sim.e[:, 1], sim.e[:, 2],
                  length=3.5, normalize=True, color=col, linewidth=1.0)

        # coherence inset: largest-cluster fraction (spatial) + local alignment
        ax_in.cla()
        ax_in.plot(clu_hist, color="darkgreen", lw=1.4, label="largest group")
        ax_in.plot(loc_hist, color="purple", lw=1.0, alpha=0.7,
                   label="local align")
        ax_in.axhline(1.0, color="gray", lw=0.6, ls=":")
        ax_in.set_ylim(0, 1.05); ax_in.set_xlim(0, 240)
        ax_in.set_title("coherence: split & recover", fontsize=9)
        ax_in.set_xticks([])
        ax_in.legend(fontsize=6, loc="lower left", framealpha=0.4)
        return ax,

    def on_close(_):
        if history:
            np.savez_compressed("drone_migration_obstacles.npz",
                                pos=np.array(history),
                                cluster=np.array(clu_hist), dt=0.05)
            print(f"saved drone_migration_obstacles.npz ({len(history)} frames)")

    fig.canvas.mpl_connect("close_event", on_close)
    _ani = FuncAnimation(fig, update, frames=itertools.count(),
                         interval=45, blit=False, cache_frame_data=False)
    plt.show()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        headless_check()
    else:
        animate()
