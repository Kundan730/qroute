/**
 * The method page: what the algorithm actually does, stated so a reader can
 * check it against the source.
 *
 * The tone here is set deliberately. "Quantum-behaved" names a sampling rule
 * borrowed from the analogy of a particle in a delta potential well; it does
 * not mean quantum hardware, quantum speed-up, or anything running on a qubit,
 * and saying so plainly is more useful to a panel than the alternative. Every
 * equation on this page corresponds to a specific line of
 * `qroute/algorithms/qpso.py`, and the parameter-study result quoted at the end
 * is the one the study actually produced, including the part that does not
 * flatter the method.
 */

import { Eq, Op, T, V } from '../components/method/Equation';
import { SamplingDemo } from '../components/method/SamplingDemo';
import { Panel } from '../components/ui';

const PROSE: React.CSSProperties = {
  maxWidth: 780,
  color: 'var(--text-dim)',
  fontSize: 13.5,
  lineHeight: 1.68,
};

export function MethodPage() {
  return (
    <div className="page-scroll">
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: 12 }}>
        {/* ------------------------------------------------------ left column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Panel title="What this method is, and what it is not">
            <div style={PROSE}>
              <p>
                The optimiser is <strong style={{ color: 'var(--text)' }}>quantum-behaved
                particle swarm optimisation</strong> (QPSO), introduced by Sun,
                Feng and Xu in 2004. The word quantum describes where the update
                rule comes from — the probability density of a particle bound in
                a delta potential well — and nothing else. This runs on an
                ordinary CPU, it uses no quantum hardware, and it claims no
                quantum speed-up. What it does claim is a specific and testable
                property of its sampling distribution, set out below.
              </p>
              <p>
                The routing model itself is classical and exact where it can be.
                Shortest paths between stops are computed with Dijkstra's
                algorithm, in polynomial time, with no metaheuristic involved.
                The hard part — deciding which vehicle serves which customers and
                in what order — is the part the swarm searches.
              </p>
            </div>
          </Panel>

          <Panel title="The update rule">
            <div style={PROSE}>
              <p>
                In classical PSO each particle carries a position and a velocity,
                and moves on a damped oscillation around a weighted average of
                its own best position and the swarm's best:
              </p>
            </div>
            <Eq
              note={
                <>
                  χ is the constriction coefficient (0.7298 in the standard
                  parameterisation) and <V>r</V><sub>1</sub>, <V>r</V><sub>2</sub>{' '}
                  are independent uniform draws. Because <V>v</V> is bounded,
                  so is the region reachable in one step.
                </>
              }
            >
              <V sub="id">v</V>
              <Op>←</Op>
              <T>χ</T>
              <Op>[</Op>
              <V sub="id">v</V>
              <Op>+</Op>
              <V sub="1">c</V>
              <V sub="1">r</V>
              <Op>(</Op>
              <V sub="id">p</V>
              <Op>−</Op>
              <V sub="id">x</V>
              <Op>)</Op>
              <Op>+</Op>
              <V sub="2">c</V>
              <V sub="2">r</V>
              <Op>(</Op>
              <V sub="d">g</V>
              <Op>−</Op>
              <V sub="id">x</V>
              <Op>)</Op>
              <Op>]</Op>
              <span style={{ margin: '0 1.4em' }} />
              <V sub="id">x</V>
              <Op>←</Op>
              <V sub="id">x</V>
              <Op>+</Op>
              <V sub="id">v</V>
            </Eq>

            <div style={PROSE}>
              <p>
                QPSO removes the velocity entirely. Each particle is treated as
                a quantum particle bound in a delta well centred on a{' '}
                <em>local attractor</em> <V sub="id">p</V>, a random convex
                combination of its personal best and the swarm best:
              </p>
            </div>
            <Eq
              note={
                <>
                  φ is drawn independently per particle and per dimension, so the
                  attractor wanders between the two bests rather than sitting at
                  their midpoint.
                </>
              }
            >
              <T>φ</T>
              <Op>∼</Op>
              <T>U(0, 1)</T>
              <span style={{ margin: '0 1.6em' }} />
              <V sub="id">p</V>
              <Op>=</Op>
              <T>φ</T>
              <Op>·</Op>
              <V sub="id">pbest</V>
              <Op>+</Op>
              <Op>(1 −</Op>
              <T>φ</T>
              <Op>)</Op>
              <Op>·</Op>
              <V sub="d">gbest</V>
            </Eq>

            <div style={PROSE}>
              <p>
                The characteristic length of the well is set from the swarm's{' '}
                <em>mean best position</em>, which is the average of every
                particle's personal best. The next position is then drawn by
                inverting the cumulative distribution of the double-exponential
                density:
              </p>
            </div>
            <Eq
              note={
                <>
                  <V>u</V> is uniform on (0, 1) and the sign is a fair coin. As{' '}
                  <V>u</V> → 0 the logarithm diverges, so the support of this
                  distribution is the whole real line: at every iteration the
                  particle has non-zero probability of appearing anywhere in the
                  search space. That is the concrete sense in which QPSO
                  explores more globally than PSO, and it is what the
                  illustration to the right measures.
                </>
              }
            >
              <V>mbest</V>
              <Op>=</Op>
              <Op>(1/</Op>
              <V>M</V>
              <Op>)</Op>
              <Op>Σ</Op>
              <V sub="i">pbest</V>
              <span style={{ margin: '0 1.6em' }} />
              <V sub="id">x</V>
              <Op>←</Op>
              <V sub="id">p</V>
              <Op>±</Op>
              <T>β</T>
              <Op>·</Op>
              <Op>|</Op>
              <V sub="d">mbest</V>
              <Op>−</Op>
              <V sub="id">x</V>
              <Op>|</Op>
              <Op>·</Op>
              <T>ln</T>
              <Op>(1/</Op>
              <V>u</V>
              <Op>)</Op>
            </Eq>

            <div style={PROSE}>
              <p>
                β is the contraction–expansion coefficient and the algorithm's
                one critical parameter. Sun et al.'s stability analysis shows the
                swarm converges for β below roughly 1.78. The implementation
                follows the standard schedule, decreasing β linearly from 1.0 to
                0.5 across the run; because the sampling width also carries the
                factor <Op>|</Op>
                <V sub="d">mbest</V>
                <Op>−</Op>
                <V sub="id">x</V>
                <Op>|</Op>, exploration narrows automatically as the personal
                bests cluster, with no separate schedule needed.
              </p>
            </div>
          </Panel>

          <Panel title="From continuous positions to vehicle routes">
            <div style={PROSE}>
              <p>
                The update rule above lives in a continuous space, and vehicle
                routing is a permutation problem. The bridge is a{' '}
                <strong style={{ color: 'var(--text)' }}>random-key</strong>{' '}
                encoding: a particle's position is a vector of one real number
                per customer, decoded by sorting the keys into a giant tour and
                then splitting that tour optimally into routes.
              </p>
              <p>
                The split is not a heuristic. Given a fixed customer order, the
                cheapest way to cut it into capacity-feasible routes is a
                shortest path in an auxiliary directed acyclic graph, and is
                solved exactly by dynamic programming in <V>O</V>(<V>n</V>
                <sup>2</sup>) time. Local search then refines each decoded
                solution, and the improvement is written back into the
                particle's keys so the swarm accumulates structural progress
                instead of rediscovering it.
              </p>
              <p>
                Every baseline in the benchmark — PSO, the genetic algorithm,
                simulated annealing, ant colony — shares that same decoder,
                split and local search. Only the rule for generating new
                candidate orderings differs. This is what makes the comparison
                meaningful: it isolates the search rule rather than measuring
                which implementation had the better local search bolted on.
              </p>
            </div>
          </Panel>
        </div>

        {/* ----------------------------------------------------- right column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Panel title="Where a quantum particle can go">
            <SamplingDemo />
          </Panel>

          <Panel title="The objective being minimised">
            <div style={PROSE}>
              <p>
                The problem statement asks for travel time, distance and
                congestion to be reduced together, so the objective is a
                weighted sum with a term for fleet size:
              </p>
            </div>
            <Eq
              note={
                <>
                  Benchmark runs use <V sub="dist">w</V> = 1 and the rest zero,
                  which reproduces the classical CVRP objective and makes the
                  reported gaps directly comparable with published best-known
                  solutions. The live traffic demonstration sets non-zero time
                  and congestion weights instead.
                </>
              }
            >
              <T>min</T>
              <Op>&nbsp;</Op>
              <V sub="time">w</V>
              <Op>Σ</Op>
              <V sub="ij">t</V>
              <Op>+</Op>
              <V sub="dist">w</V>
              <Op>Σ</Op>
              <V sub="ij">d</V>
              <Op>+</Op>
              <V sub="cong">w</V>
              <Op>Σ</Op>
              <T>c</T>
              <Op>(</Op>
              <V sub="ij">·</V>
              <Op>)</Op>
              <Op>+</Op>
              <V sub="veh">w</V>
              <Op>·</Op>
              <V>K</V>
            </Eq>
            <div style={PROSE}>
              <p>
                Capacity, time-window, route-duration, fleet-size and edge-load
                constraints are handled by an adaptive penalty rather than by
                rejecting infeasible candidates, because on tightly constrained
                instances a search confined to the feasible region explores far
                more slowly. Reported feasibility is computed from the raw
                violation amounts, not from the penalised cost, so a solution
                cannot be made to look feasible by tuning a weight.
              </p>
            </div>
          </Panel>

          <Panel title="What the parameter study found">
            <div style={PROSE}>
              <p>
                A sweep over swarm size, contraction schedule, mutation operator
                and local-search policy was run on A-n45-k7, A-n80-k10,
                X-n101-k25 and R101, three seeds each, with an equal twelve
                second budget per configuration.
              </p>
              <p>
                <strong style={{ color: 'var(--text)' }}>
                  Mean gap to the best-known solution across every configuration
                  tried fell in a narrow band, roughly 1.6 % to 2.1 %, with a
                  standard deviation across runs near 1.0.
                </strong>{' '}
                In other words the differences between reasonable parameter
                settings were smaller than the run-to-run noise, and no setting
                was significantly better than another.
              </p>
              <p>
                That is worth stating plainly rather than hiding, because it says
                something true about this class of algorithm: once an optimal
                split and a good local search are in place, they do most of the
                work, and the swarm rule mainly decides which orderings get
                refined. The defaults shipped are the best observed mean, not a
                claim of tuned superiority. The comparison that does carry
                information is against the same pipeline driven by other search
                rules, which is what the benchmark page reports, and against the
                random-restart control, which is what separates a real search
                from a lucky decoder.
              </p>
            </div>
          </Panel>

          <Panel title="References">
            <ol
              style={{
                ...PROSE,
                paddingLeft: 20,
                margin: 0,
                fontSize: 12.5,
                lineHeight: 1.75,
              }}
            >
              <li>
                Sun, J., Feng, B. and Xu, W. (2004). Particle swarm optimization
                with particles having quantum behavior. <em>Proceedings of the
                IEEE Congress on Evolutionary Computation</em>, 325–331.
              </li>
              <li>
                Sun, J., Fang, W., Palade, V., Wu, X. and Xu, W. (2011).
                Quantum-behaved particle swarm optimization with Gaussian
                distributed local attractor point. <em>Applied Mathematics and
                Computation</em> 218(7), 3763–3775.
              </li>
              <li>
                Xi, M., Sun, J. and Xu, W. (2008). An improved quantum-behaved
                particle swarm optimization algorithm with weighted mean best
                position. <em>Applied Mathematics and Computation</em> 205(2),
                751–759.
              </li>
              <li>
                Clerc, M. and Kennedy, J. (2002). The particle swarm — explosion,
                stability, and convergence in a multidimensional complex space.{' '}
                <em>IEEE Transactions on Evolutionary Computation</em> 6(1),
                58–73.
              </li>
              <li>
                Prins, C. (2004). A simple and effective evolutionary algorithm
                for the vehicle routing problem. <em>Computers &amp; Operations
                Research</em> 31(12), 1985–2002. (The optimal split used by the
                decoder.)
              </li>
              <li>
                Bean, J. C. (1994). Genetic algorithms and random keys for
                sequencing and optimization. <em>ORSA Journal on Computing</em>{' '}
                6(2), 154–160.
              </li>
              <li>
                Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T. and
                Subramanian, A. (2017). New benchmark instances for the
                capacitated vehicle routing problem. <em>European Journal of
                Operational Research</em> 257(3), 845–858.
              </li>
              <li>
                Solomon, M. M. (1987). Algorithms for the vehicle routing and
                scheduling problems with time window constraints.{' '}
                <em>Operations Research</em> 35(2), 254–265.
              </li>
              <li>
                Bureau of Public Roads (1964). <em>Traffic Assignment Manual</em>.
                U.S. Department of Commerce. (The volume–delay function used by
                the traffic simulator.)
              </li>
              <li>
                Transportation Research Board (2016).{' '}
                <em>Highway Capacity Manual</em>, 6th edition. (The incident
                residual-capacity table used when a lane is blocked.)
              </li>
              <li>
                Demšar, J. (2006). Statistical comparisons of classifiers over
                multiple data sets. <em>Journal of Machine Learning Research</em>{' '}
                7, 1–30. (The Friedman and Holm procedure used on the benchmark
                page.)
              </li>
            </ol>
          </Panel>
        </div>
      </div>
    </div>
  );
}
