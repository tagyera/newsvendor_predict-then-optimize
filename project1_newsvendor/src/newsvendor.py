import numpy as np

import matplotlib.pyplot as plt

# functions


def newsvendor_cost(q, d, c_o, c_u):
    # cost of ordering q when the demand turns out to be d

    q = np.asarray(q, dtype=float)
    d = np.asarray(d, dtype=float)
    overage = np.maximum(q - d, 0.0)  # leftover
    underage = np.maximum(d - q, 0.0)  # unmet demand
    return c_o * overage + c_u * underage


def critical_ratio(c_o, c_u):

    return c_u / (c_o + c_u)


def optimal_order(demand_samples, c_o, c_u):

    cr = critical_ratio(c_o, c_u)

    return np.quantile(demand_samples, cr, method="inverted_cdf")


# ---------- demo ----------

if __name__ == "__main__":
    c_o, c_u = 1.0, 3.0  # bakery: pays £1, sells £4, no salvage value
    cr = critical_ratio(c_o, c_u)
    print(f"c_o={c_o}, c_u={c_u}, critical ratio={cr:.2f}\n")

    # 1) Cost for single outcomes
    print("--- Single-day costs ---")
    print("order 10, demand 7  ->", newsvendor_cost(10, 7, c_o, c_u))
    print("order 10, demand 14 ->", newsvendor_cost(10, 14, c_o, c_u))
    print("order 10, demand 10 ->", newsvendor_cost(10, 10, c_o, c_u))

    # 2) Ten days of observed demand

    rng = np.random.default_rng(42)

    demand = np.array([12, 15, 9, 20, 14, 11, 18, 13, 16, 10])
    # demand = rng.lognormal(mean=3, sigma=0.6, size=1000)
    # print(demand.mean(), np.quantile(demand, 0.75), np.quantile(demand, 0.95))

    print("\n--- Ten days of demand ---")
    print("demand:", demand.tolist())
    print("mean  :", demand.mean())
    print("median:", np.median(demand))

    q_mean = demand.mean()
    q_star = optimal_order(demand, c_o, c_u)
    print(f"75th-percentile order (q*): {q_star}")

    # 3) Compare 'order the mean' vs 'order the CR-quantile'
    cost_mean = newsvendor_cost(q_mean, demand, c_o, c_u)
    cost_star = newsvendor_cost(q_star, demand, c_o, c_u)
    print("\n--- Average cost per day ---")
    print(f"order the mean ({q_mean:.1f}): {cost_mean.mean():.2f}")
    print(f"order q* ({q_star}):          {cost_star.mean():.2f}")

    # 4) Brute-force check: try every order quantity, find the cheapest
    grid = np.arange(0, 26)
    avg_costs = [newsvendor_cost(q, demand, c_o, c_u).mean() for q in grid]
    best = grid[int(np.argmin(avg_costs))]
    print(f"\nBrute-force best q over grid: {best} (avg cost {min(avg_costs):.2f})")

    # 5) Symmetric costs: CR = 0.5, optimal order is the median
    cr_sym = critical_ratio(1.0, 1.0)
    q_sym = optimal_order(demand, 1.0, 1.0)
    print(
        f"\nSymmetric costs: CR={cr_sym}, optimal order={q_sym} (median={np.median(demand)})"
    )

    # 6) Oracle and regret
    oracle_cost = newsvendor_cost(demand, demand, c_o, c_u)  # order exactly d each day
    regret = cost_mean - oracle_cost
    print(f"\nOracle avg cost: {oracle_cost.mean():.2f}")
    print(f"Regret of ordering the mean: {regret.mean():.2f} per day")

    grid = np.arange(0, 26)
    avg = [newsvendor_cost(q, demand, 1, 3).mean() for q in grid]
    plt.plot(grid, avg)
    plt.xlabel("order q")
    plt.ylabel("avg cost")
    plt.savefig("../figures/cost_vs_q_1.png", dpi=150)
