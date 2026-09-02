"""Reference searches used to calibrate what the metaheuristics contribute.

``RandomRestart`` repeatedly samples a random customer ordering, splits it
optimally and applies the same local search the metaheuristics use, keeping the
best result. It has no memory, no population and no learning of any kind.

It is the most important baseline in the project, and the least flattering. A
memetic algorithm whose global search adds nothing would score exactly the same
as this, because everything else in the pipeline is shared. Reporting it makes
the contribution of the swarm rule measurable rather than assumed.
"""

from __future__ import annotations

import numpy as np

from qroute.algorithms.base import Optimizer
from qroute.algorithms.decoder import Decoder
from qroute.core.types import Solution


class RandomRestart(Optimizer):
    """Multi-start local search: the control against which search rules are judged."""

    name = "random-restart"

    def __init__(self, instance, stop=None, seed=None, callback=None,
                 batch: int = 10, neighbours: int = 15, local_search: bool = True,
                 decoder: Decoder | None = None, **kw):
        super().__init__(instance, stop, seed, callback, batch=batch,
                         neighbours=neighbours, local_search=local_search, **kw)
        self.batch = int(batch)
        self.decoder = decoder or Decoder(instance, neighbours=neighbours,
                                          use_local_search=local_search)

    def _run(self) -> int:
        n = self.instance.n_customers
        it = 0
        while not self.should_stop(it):
            it += 1
            costs = []
            for _ in range(self.batch):
                keys = self.rng.random(n)
                routes, cost, _ = self.decoder.decode(keys)
                costs.append(cost)
                self.offer(Solution([list(r) for r in routes], cost))
            self.evaluations += self.batch
            self.record(it, self._best.cost, float(np.mean(costs)), 0.0, True)
        return it
