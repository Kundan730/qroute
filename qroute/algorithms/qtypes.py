"""Quantum-bit register: the classical data structure a QIEA rotates.

What this is
------------
Han and Kim's Quantum-Inspired Evolutionary Algorithm (IEEE Transactions on
Evolutionary Computation 6(6), 2002) does not run on a quantum computer. It
borrows one piece of notation from quantum mechanics - the qubit as a pair of
probability amplitudes - and uses it as a *probabilistic genotype*: instead of
carrying a binary string, an individual carries, for every bit position, the
probability that the bit will be observed as 1. Search then happens by nudging
those probabilities, not by crossing over strings.

A single qubit is written

    |psi> = alpha |0> + beta |1>,      |alpha|^2 + |beta|^2 = 1

and observing it yields 1 with probability ``beta**2``. A register of ``m``
qubits is stored here as two real arrays of length ``m``.

Be clear about what is and is not simulated
-------------------------------------------
* Amplitudes are **real**, not complex. Phase carries no information in this
  algorithm, so storing complex numbers would double the memory for nothing.
* The register is a **product state**. Storing ``m`` amplitude pairs describes
  exactly ``2*m`` numbers, whereas a genuine ``m``-qubit state needs ``2**m``
  amplitudes. Entanglement, interference between branches and any speed-up that
  depends on them are therefore *not* represented. What survives from the
  quantum formalism is a compact encoding of a factorised probability
  distribution over binary strings, plus a unitary (norm-preserving) update rule
  for it.
* Consequently this is a classical estimation-of-distribution algorithm whose
  distribution happens to be parameterised by angles. That is a real and useful
  thing - the rotation is a bounded, reversible, well-conditioned way to move a
  Bernoulli parameter - but it is not quantum computation, and no claim of
  quantum speed-up is made anywhere in this module.

Why the angle parameterisation is worth having
----------------------------------------------
Writing the Bernoulli parameter as ``sin^2(theta)`` rather than as a raw
probability ``p`` means every update is a rotation: the parameter can never
leave ``[0, 1]``, no clipping is needed, and equal angular steps produce small
probability changes near the ends of the range and large ones in the middle.
That is exactly the behaviour wanted from a learning rate on a probability, and
it comes for free from the geometry.
"""

from __future__ import annotations

import numpy as np

# Magnitudes of the rotation angle, expressed as multiples of the base step
# ``delta_theta``. Han and Kim tabulate them as absolute angles (0.005*pi to
# 0.05*pi); with the default ``delta_theta = 0.01*pi`` these multipliers
# reproduce those values exactly, while letting the whole table be scaled by a
# single parameter or an annealing schedule.
_MULT_AGREE_ONE = 2.5      # 0.025*pi - both strings say 1, reinforce it
_MULT_AGREE_ZERO = 2.5     # 0.025*pi - both say 0 (see the note on tables below)
_MULT_DISAGREE_WORSE = 5.0    # 0.05*pi  - move toward the better string's bit
_MULT_DISAGREE_BETTER = 1.0   # 0.01*pi  - keep the current bit, gently


class QubitRegister:
    """``m`` independent qubits stored as real amplitude pairs.

    Parameters
    ----------
    size:
        Number of qubits ``m``.
    table:
        Which rotation lookup table to use.

        ``"han_kim"``
            The 2002 paper's table literally: agreeing 0-bits get **no**
            rotation while agreeing 1-bits are reinforced. That asymmetry is
            deliberate in the paper because its benchmark is the 0/1 knapsack,
            where a 1-bit means "item taken" and only taken items carry
            information.
        ``"symmetric"``
            Identical except that agreeing 0-bits are reinforced by the same
            amount as agreeing 1-bits. Use this whenever 0 and 1 are equally
            meaningful - notably when a bit-string is read as a *number*, as in
            :class:`~qroute.algorithms.qiea.QuantumRotationKeys`, since under
            ``"han_kim"`` a register drifts toward all-ones and therefore biases
            every decoded value upward. This is the default here because that
            failure mode is easy to hit and hard to notice.

    Notes
    -----
    Amplitudes start at ``1/sqrt(2)`` each, the equal superposition: before any
    evidence arrives every binary string of length ``m`` is equally likely.
    """

    __slots__ = ("alpha", "beta", "table")

    def __init__(self, size: int, table: str = "symmetric",
                 alpha: np.ndarray | None = None, beta: np.ndarray | None = None):
        if size <= 0:
            raise ValueError("a register needs at least one qubit")
        if table not in ("han_kim", "symmetric"):
            raise ValueError(f"unknown rotation table {table!r}")
        self.table = table
        if alpha is None or beta is None:
            root = 1.0 / np.sqrt(2.0)
            self.alpha = np.full(size, root, dtype=np.float64)
            self.beta = np.full(size, root, dtype=np.float64)
        else:
            self.alpha = np.array(alpha, dtype=np.float64, copy=True)
            self.beta = np.array(beta, dtype=np.float64, copy=True)
            if self.alpha.shape != (size,) or self.beta.shape != (size,):
                raise ValueError("amplitude arrays do not match the register size")
            self._renormalise()

    # ------------------------------------------------------------------ basics
    @property
    def size(self) -> int:
        return int(self.alpha.shape[0])

    def __len__(self) -> int:
        return self.size

    def copy(self) -> "QubitRegister":
        return QubitRegister(self.size, self.table, self.alpha, self.beta)

    def probabilities(self) -> np.ndarray:
        """Probability of observing 1 at each position, i.e. ``beta**2``."""
        return self.beta * self.beta

    def entropy(self, per_qubit: bool = False) -> np.ndarray | float:
        """Binary Shannon entropy in bits.

        Returns the mean over the register (a scalar) unless ``per_qubit``.
        A value near 1 means the register is still undecided everywhere; near 0
        means it has converged to a single string and the population has stopped
        exploring. This is the QIEA analogue of swarm diversity and is what the
        convergence history records.
        """
        p = np.clip(self.probabilities(), 1e-15, 1.0 - 1e-15)
        h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        return h if per_qubit else float(h.mean())

    # ------------------------------------------------------------- observation
    def observe(self, rng: np.random.Generator) -> np.ndarray:
        """Collapse the register once, returning a ``uint8`` vector of bits.

        This does not modify the register: unlike a physical measurement, the
        simulated state survives so the same distribution can be sampled again.
        Keeping the state is the whole point - the algorithm learns by moving
        the distribution, not by destroying it.
        """
        return (rng.random(self.size) < self.probabilities()).astype(np.uint8)

    # ---------------------------------------------------------------- rotation
    def rotate(self, x: np.ndarray, best_x: np.ndarray, better: bool,
               delta_theta: float = 0.01 * np.pi,
               rng: np.random.Generator | None = None) -> None:
        """Apply the quantum rotation gate in place.

        The gate is the ordinary 2x2 rotation

            [alpha']   [cos t   -sin t] [alpha]
            [beta' ] = [sin t    cos t] [beta ]

        applied independently to every qubit, with ``t = s * mult * delta_theta``.
        It is orthogonal, so the norm is preserved exactly and the state stays a
        legal qubit for any angle - that is why no clipping appears here.

        Parameters
        ----------
        x:
            The bit-string just observed from this register.
        best_x:
            The reference string, normally the best solution seen by this
            individual (or, after a migration, by its group or the population).
        better:
            ``True`` when ``x`` scored better than ``best_x``. The caller owns
            the sense of "better" (this project minimises cost), so no objective
            values are passed in.
        delta_theta:
            Base step size. The lookup table scales it per row.
        rng:
            Only consulted for the degenerate rows where a qubit has already
            collapsed to exactly ``|0>`` or ``|1>`` and the rotation direction is
            genuinely arbitrary. Pass one for reproducibility; without it the
            tie is broken deterministically toward positive angles. In practice
            :meth:`h_epsilon` keeps these rows from ever firing.

        The lookup table
        ----------------
        Every row answers one question: which bit value should this qubit be
        pushed toward, and how hard?

        ===== ===== ========= =============== ==================================
        x_i   b_i   x better  target bit      magnitude (multiples of delta)
        ===== ===== ========= =============== ==================================
        0     0     no        0               2.5   (0 with table="han_kim")
        0     0     yes       0               2.5   (0 with table="han_kim")
        0     1     no        1  (from b)     5.0   b is better and says 1
        0     1     yes       0  (from x)     1.0   x is better and says 0
        1     0     no        0  (from b)     5.0   b is better and says 0
        1     0     yes       1  (from x)     1.0   x is better and says 1
        1     1     no        1               2.5   both agree on 1
        1     1     yes       1               2.5   both agree on 1
        ===== ===== ========= =============== ==================================

        Two things are worth stating plainly. First, published versions of this
        table disagree with each other on the sign column, and several printed
        variants are internally inconsistent (they list a sign that rotates
        *away* from the intended bit in one quadrant). Rather than copy a
        particular printing, the magnitudes are taken from the paper and the
        sign is *derived*, below, from the requirement that the step must
        increase the probability of the target bit. That gives the same answers
        as the correct rows of the published table and is checkable.

        Second, the four-quadrant sign rule falls straight out of calculus.
        Differentiating at ``t = 0``:

            d(beta^2)/dt  = +2 * alpha * beta
            d(alpha^2)/dt = -2 * alpha * beta

        so to raise ``P(1) = beta^2`` take ``s = sign(alpha*beta)``, and to raise
        ``P(0) = alpha^2`` take ``s = -sign(alpha*beta)``. The four sign cells of
        the paper's table (alpha*beta > 0, alpha*beta < 0, alpha = 0, beta = 0)
        are exactly this expression together with its two degenerate cases.
        """
        x = np.asarray(x, dtype=np.uint8)
        b = np.asarray(best_x, dtype=np.uint8)
        if x.shape != (self.size,) or b.shape != (self.size,):
            raise ValueError("bit-strings do not match the register size")

        agree = x == b
        # The bit to move toward: whichever string is better owns the target.
        # Where the strings agree both choices coincide, so this one expression
        # covers all eight rows.
        target = x if better else b

        mult = np.empty(self.size, dtype=np.float64)
        agree_one = agree & (x == 1)
        agree_zero = agree & (x == 0)
        mult[agree_one] = _MULT_AGREE_ONE
        mult[agree_zero] = _MULT_AGREE_ZERO if self.table == "symmetric" else 0.0
        disagree = ~agree
        mult[disagree] = _MULT_DISAGREE_BETTER if better else _MULT_DISAGREE_WORSE

        # ---- four-quadrant sign rule -------------------------------------
        ab = self.alpha * self.beta
        s = np.sign(ab)
        s = np.where(target == 1, s, -s)

        # Degenerate rows: a fully collapsed qubit has ab == 0. If it already
        # sits on the target bit no rotation is needed; if it sits on the
        # opposite one, either direction moves it back, so the choice is free.
        collapsed = ab == 0.0
        if collapsed.any():
            on_target = ((target == 1) & (self.alpha == 0.0)) | \
                        ((target == 0) & (self.beta == 0.0))
            free = collapsed & ~on_target
            s = np.where(collapsed, 0.0, s)
            if free.any():
                if rng is None:
                    s = np.where(free, 1.0, s)
                else:
                    s = np.where(free, np.where(rng.random(self.size) < 0.5, 1.0, -1.0), s)

        theta = s * mult * float(delta_theta)
        ct = np.cos(theta)
        st = np.sin(theta)
        new_alpha = ct * self.alpha - st * self.beta
        new_beta = st * self.alpha + ct * self.beta
        self.alpha = new_alpha
        self.beta = new_beta
        # Analytically the rotation is norm-preserving; this only sweeps up
        # floating-point drift accumulated over millions of gates.
        self._renormalise()

    # ------------------------------------------------------------ H-epsilon
    def h_epsilon(self, eps: float = 0.01) -> None:
        """Clamp every observation probability into ``[eps, 1 - eps]``.

        Han and Kim call this the H-epsilon gate, and it is the only thing
        standing between the algorithm and premature convergence, because QIEA
        has no mutation operator.

        The failure it prevents is absorbing. A qubit driven onto a pole has
        ``P(1) = 1`` exactly, so observation can never return 0 again; and the
        gate's own degenerate row assigns a zero angle to a qubit that already
        sits on its target bit, so rotation cannot bring it back either. That
        degree of freedom is then lost for the rest of the run. Even short of
        the pole the effect bites: at ``P(1) = 0.999`` a population of 20 needs
        about fifty generations to see the other bit once.

        Clamping costs a small, bounded amount of exploitation - the register
        can never be more than ``1 - eps`` certain - and buys back ergodicity:
        every binary string keeps non-zero probability forever.

        The clamp preserves the sign of each amplitude, so it does not silently
        flip which quadrant a qubit is in and therefore does not disturb the
        rotation sign rule.
        """
        eps = float(eps)
        if not 0.0 <= eps < 0.5:
            raise ValueError("eps must lie in [0, 0.5)")
        p = np.clip(self.probabilities(), eps, 1.0 - eps)
        # np.sign(0) is 0, which would zero an amplitude outright; treat an
        # exactly-zero amplitude as positive so the clamp can lift it away.
        sa = np.where(self.alpha < 0.0, -1.0, 1.0)
        sb = np.where(self.beta < 0.0, -1.0, 1.0)
        self.beta = sb * np.sqrt(p)
        self.alpha = sa * np.sqrt(1.0 - p)

    # ------------------------------------------------------------- internals
    def _renormalise(self) -> None:
        norm = np.sqrt(self.alpha * self.alpha + self.beta * self.beta)
        # A zero norm cannot arise from a rotation of a normalised state, but
        # guard anyway so a caller-supplied array cannot produce NaNs.
        bad = norm <= 0.0
        if bad.any():
            root = 1.0 / np.sqrt(2.0)
            self.alpha = np.where(bad, root, self.alpha)
            self.beta = np.where(bad, root, self.beta)
            norm = np.where(bad, 1.0, norm)
        self.alpha = self.alpha / norm
        self.beta = self.beta / norm

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"QubitRegister(size={self.size}, table={self.table!r}, "
                f"entropy={self.entropy():.3f} bits)")
