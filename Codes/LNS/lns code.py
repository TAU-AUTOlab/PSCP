"""
Large Neighborhood Search (LNS) Framework for the Partial Set Covering Problem (PSCP)
=====================================================================================
"""

import os
import pandas as pd
import time
import random
import heapq
from docplex.mp.model import Model


# ==============================================================================
# 1. DATA LOADING & STRUCTURAL PARSING
# ==============================================================================

def loading(file_path):
    """
    Parses a specialized Excel spreadsheet containing a PSCP problem instance.

    Parameters:
        file_path (str): Relative or absolute system path to the instance file.

    Returns:
        tuple: (a, num_customers, num_sites)
            - a (list of list of int): Adjacency list mapping each 0-indexed customer
                                       to a list of 1-indexed eligible facility sites.
            - num_customers (int): Total number of demand points / customers.
            - num_sites (int): Total number of potential facility locations.
    """
    df = pd.read_excel(file_path, header=None, engine='openpyxl')
    df = df.dropna(how="all").reset_index(drop=True)
    data = df.values.tolist()

    num_customers = int(data[0][0])
    num_sites = int(data[0][1])
    a = []
    i = 1

    # Process structural dimensions row by row
    while len(a) < num_customers:
        while i < len(data):
            row = [x for x in data[i] if pd.notna(x)]
            if len(row) == 1:
                k = int(row[0])  # Degree of the current customer
                i += 1
                break
            i += 1
        values = []
        while len(values) < k and i < len(data):
            row = [x for x in data[i] if pd.notna(x)]
            values.extend(int(x) for x in row)
            i += 1
        a.append(values[:k])
    return a, num_customers, num_sites


def build_site_to_customers(a, num_sites):
    """
    Inverts the customer adjacency matrix to compile a mapping from sites to customers.

    Parameters:
        a (list of list of int): Customer adjacency list.
        num_sites (int): Total number of potential facility locations.

    Returns:
        dict: Keys are 1-indexed site IDs, values are sets of 0-indexed customer IDs.
    """
    site_to_customers = {j: set() for j in range(1, num_sites + 1)}
    for i, sites in enumerate(a):
        for s in sites:
            site_to_customers[s].add(i)
    return site_to_customers


# ==============================================================================
# 2. SITE SELECTION SCORES & CONSTRUCTIVE HEURISTIC SEEDING
# ==============================================================================

def compute_customer_degrees(a):
    """Calculates the covering degree (total overlapping sites) for each customer."""
    return [len(sites) for sites in a]


def compute_ssf(site_to_customers, customer_degrees):
    """
    Computes the Site Shortage Factor (SSF). 

    A higher score indicates the site covers rare/vulnerable customer vertices.
    $$\text{SSF}_j = \sum_{i \in \mathcal{C}_j} \frac{1}{d_i}$$
    """
    ssf = {}
    for m, custs in site_to_customers.items():
        val = 0.0
        for i in custs:
            d = customer_degrees[i]
            if d > 0:
                val += 1.0 / d
        ssf[m] = val
    return ssf


def forward_greedy_seed(a, num_customers, num_sites, required_coverage):
    """
    Standard deterministic constructive heuristic using set-covering greedy scores.
    Maintains a multi-criteria priority key: (Immediate Gain, SSF Tie-Breaker, Site ID).
    """
    site_to_customers = build_site_to_customers(a, num_sites)
    deg = compute_customer_degrees(a)
    ssf = compute_ssf(site_to_customers, deg)

    uncovered = set(range(num_customers))
    chosen = set()
    sites = list(range(1, num_sites + 1))

    while len(uncovered) > 0 and (num_customers - len(uncovered)) < required_coverage:
        best_site = None
        best_key = None  # (gain, ssf, -site_id)
        for m in sites:
            if m in chosen:
                continue
            gain = 0.0
            for i in site_to_customers[m]:
                if i in uncovered:
                    d = deg[i]
                    if d > 0:
                        gain += 1.0 / d
            key = (gain, ssf[m], -m)
            if (best_key is None) or (key > best_key):
                best_key = key
                best_site = m

        if best_site is None:
            break

        chosen.add(best_site)
        for i in site_to_customers[best_site]:
            if i in uncovered:
                uncovered.remove(i)

        if (num_customers - len(uncovered)) >= required_coverage:
            break

    covered_count = num_customers - len(uncovered)
    return sorted(chosen), covered_count


def fast_forward_greedy_seed(a, num_customers, num_sites, required_coverage):
    """
    High-performance constructive greedy framework utilizing incremental score updates 
    and a lazy evaluation heap. Drastically scales down performance loops on massive vertices.
    """
    site_to_customers = {j: [] for j in range(1, num_sites + 1)}
    for i, sites in enumerate(a):
        for s in sites:
            site_to_customers[s].append(i)

    deg = [len(row) for row in a]
    inv_deg = [(1.0 / d if d > 0 else 0.0) for d in deg]

    ssf = [0.0] * (num_sites + 1)
    for s in range(1, num_sites + 1):
        val = 0.0
        for i in site_to_customers[s]:
            val += inv_deg[i]
        ssf[s] = val

    covered = [False] * num_customers
    uncovered_cnt = num_customers
    gain = [0.0] * (num_sites + 1)
    for s in range(1, num_sites + 1):
        g = 0.0
        for i in site_to_customers[s]:
            if not covered[i]:
                g += inv_deg[i]
        gain[s] = g

    # Min-heap handles tracking maximum gain using inverse elements
    heap = []
    ver = [0] * (num_sites + 1)
    for s in range(1, num_sites + 1):
        heapq.heappush(heap, (-gain[s], -ssf[s], s, ver[s]))

    chosen = []
    target = required_coverage

    while (num_customers - (uncovered_cnt)) < target:
        while True:
            if not heap:
                covered_count = num_customers - uncovered_cnt
                return sorted(chosen), covered_count
            ng, nssf, s, v = heapq.heappop(heap)
            if v == ver[s]:
                break  # Record is fresh, not invalidated by lazy score shifts

        if -ng <= 1e-15:
            covered_count = num_customers - uncovered_cnt
            return sorted(chosen), covered_count

        chosen.append(s)

        newly = []
        for i in site_to_customers[s]:
            if not covered[i]:
                covered[i] = True
                newly.append(i)
                uncovered_cnt -= 1

        if (num_customers - (uncovered_cnt)) >= target:
            break

        # Adjust score parameters dynamically for intersecting network neighborhoods
        for i in newly:
            w = inv_deg[i]
            if w == 0.0:
                continue
            for t in a[i]:
                if t == s:
                    continue
                gain[t] -= w
                ver[t] += 1
                heapq.heappush(heap, (-gain[t], -ssf[t], t, ver[t]))

        gain[s] = float("-inf")
        ver[s] += 1

    covered_count = num_customers - uncovered_cnt
    return sorted(chosen), covered_count


# ==============================================================================
# 3. SOLUTION VALIDATION METRICS
# ==============================================================================

def compute_cover_counts(solution, site_to_customers, num_customers):
    """Computes reference matrix matching each customer index to its current overlapping facility coverage count."""
    counts = [0] * num_customers
    for s in solution:
        for i in site_to_customers[s]:
            counts[i] += 1
    return counts


# ==============================================================================
# 4. BASELINE DESTROY & REPAIR LOGIC
# ==============================================================================

def greedy_destroy_min_loss_baseline(best_solution, site_to_customers, num_customers, destroy_k,
                                     require_feasible_after=False, required_coverage=None):
    """
    Standard sequential lookahead destroy algorithm.
    Eradicates locations causing the minimum immediate drop in overall network coverage metric.
    """
    solution = set(best_solution)
    if destroy_k <= 0 or len(solution) <= 1:
        return sorted(solution), []

    counts = compute_cover_counts(solution, site_to_customers, num_customers)

    def total_covered_now():
        return sum(1 for c in counts if c > 0)

    removed = []
    for _ in range(min(destroy_k, len(solution) - 1)):
        losses = []
        total_now = total_covered_now()
        for s in solution:
            uniq_loss = 0
            for i in site_to_customers[s]:
                if counts[i] == 1:
                    uniq_loss += 1
            if require_feasible_after and required_coverage is not None:
                if total_now - uniq_loss < required_coverage:
                    continue
            losses.append((uniq_loss, len(site_to_customers[s]), s))

        if not losses:
            break

        _, _, s_star = min(losses)

        for i in site_to_customers[s_star]:
            counts[i] -= 1
        solution.remove(s_star)
        removed.append(s_star)

    return sorted(solution), removed


def site_fixing_greedy_delta_baseline(a, num_sites, required_coverage, max_iterations=None, fixed_open_sites=None):
    """
    Standard two-branch construction heuristic.
    Evaluates inclusion/exclusion constraints iteratively via exploratory local search passes.
    """
    site_to_customers = {j: set() for j in range(1, num_sites + 1)}
    for i, sites in enumerate(a):
        for s in sites:
            site_to_customers[s].add(i)

    all_sites = list(range(1, num_sites + 1))
    current_solution = set(fixed_open_sites or [])
    covered_customers = set()
    for s in current_solution:
        covered_customers.update(site_to_customers[s])

    best_solution = None
    best_count = float('inf')
    fixed_sites = {s: 1 for s in current_solution}
    iterations = 0
    while len(fixed_sites) < num_sites:
        if max_iterations is not None and iterations >= max_iterations:
            break
        site = random.choice([s for s in all_sites if s not in fixed_sites])
        iterations += 1

        selected1 = current_solution | {site}
        covered1 = covered_customers | site_to_customers[site]
        sol1 = selected1.copy()
        cov1 = covered1.copy()

        remaining_sites = [s for s in all_sites if s not in sol1 and s not in fixed_sites]
        while len(cov1) < required_coverage:
            best_site = max(remaining_sites, key=lambda s: len(site_to_customers[s] - cov1), default=None)
            if best_site is None:
                break
            sol1.add(best_site)
            cov1.update(site_to_customers[best_site])
            remaining_sites.remove(best_site)

        selected2 = current_solution.copy()
        covered2 = covered_customers.copy()
        remaining_sites2 = [s for s in all_sites if s != site and s not in fixed_sites]
        while len(covered2) < required_coverage:
            best_site = max(remaining_sites2, key=lambda s: len(site_to_customers[s] - covered2), default=None)
            if best_site is None:
                break
            selected2.add(best_site)
            covered2.update(site_to_customers[best_site])
            remaining_sites2.remove(best_site)

        if len(sol1) <= len(selected2):
            current_solution = sol1
            covered_customers = cov1
            fixed_sites[site] = 1
        else:
            current_solution = selected2
            covered_customers = covered2
            fixed_sites[site] = 0

        if len(current_solution) < best_count:
            best_solution = current_solution.copy()
            best_count = len(current_solution)

    return best_count, sorted(best_solution), len(covered_customers), round(time.time(), 2)


def milp_repair_baseline(a, num_customers, num_sites, required_coverage, fixed_sites, timelimit=1):
    """Standard Exact Repair: Instantiates an internal CPLEX model instance from scratch on every call."""
    model = Model(name="MILP_Repair", log_output=False)
    x = model.binary_var_list(num_customers, name="x")
    y = model.binary_var_list(num_sites, name="y")

    for i in range(num_customers):
        model.add_constraint(model.sum(y[j - 1] for j in a[i]) >= x[i])
    for j in fixed_sites:
        model.add_constraint(y[j - 1] == 1)
    model.add_constraint(model.sum(x) >= required_coverage)
    model.minimize(model.sum(y))

    model.parameters.timelimit = timelimit
    solution = model.solve()
    if not solution:
        return None, 0

    selected_sites = [j + 1 for j in range(num_sites) if y[j].solution_value > 0.5]
    covered_customers = sum(1 for i in range(num_customers) if x[i].solution_value > 0.5)
    return selected_sites, covered_customers


# ==============================================================================
# 5. HIGH-PERFORMANCE OPTIMIZED DESTROY & REPAIR MODULES
# ==============================================================================

def greedy_destroy_min_loss_fast(best_solution, site_to_customers, num_customers, destroy_k,
                                 require_feasible_after=False, required_coverage=None):
    """Optimized greedy destroy layer utilizing localized updates instead of full matrix re-evaluations."""
    solution = set(best_solution)
    if destroy_k <= 0 or len(solution) <= 1:
        return sorted(solution), []

    counts = compute_cover_counts(solution, site_to_customers, num_customers)
    total_covered = sum(1 for v in counts if v > 0)

    removed = []
    for _ in range(min(destroy_k, len(solution) - 1)):
        best = None
        best_tuple = None
        sc = site_to_customers
        local_counts = counts
        local_total = total_covered

        for s in solution:
            uniq_loss = 0
            for i in sc[s]:
                if local_counts[i] == 1:
                    uniq_loss += 1

            if require_feasible_after and required_coverage is not None:
                if local_total - uniq_loss < required_coverage:
                    continue

            t = (uniq_loss, len(sc[s]), s)
            if best_tuple is None or t < best_tuple:
                best_tuple = t
                best = s

        if best is None:
            break

        for i in sc[best]:
            if local_counts[i] == 1:
                total_covered -= 1
            local_counts[i] -= 1

        solution.remove(best)
        removed.append(best)

    return sorted(solution), removed


def greedy_destroy_min_loss_lazy(best_solution, a, site_to_customers, num_customers, destroy_k,
                                 require_feasible_after=False, required_coverage=None):
    """
    Advanced lazy tracking destroy operator. Maintains an internal state priority heap
    and executes network updates exclusively across intersecting neighborhood boundaries.
    """
    S = set(best_solution)
    if destroy_k <= 0 or len(S) <= 1:
        return sorted(S), []

    counts = [0] * num_customers
    for s in S:
        for i in site_to_customers[s]:
            counts[i] += 1

    total_cov = sum(1 for c in counts if c > 0)

    uniq_loss = {}
    for s in S:
        ul = 0
        for i in site_to_customers[s]:
            if counts[i] == 1:
                ul += 1
        uniq_loss[s] = ul

    heap = []
    ver = {s: 0 for s in S}

    def push(s):
        heapq.heappush(heap, (uniq_loss[s], len(site_to_customers[s]), s, ver[s]))

    for s in S:
        push(s)

    removed = []
    steps = min(destroy_k, len(S) - 1)

    while steps > 0 and S:
        while heap:
            ul, covsz, s, v = heapq.heappop(heap)
            if s in S and v == ver[s] and ul == uniq_loss[s]:
                break
        else:
            break

        if require_feasible_after and required_coverage is not None:
            if total_cov - uniq_loss[s] < required_coverage:
                ver[s] += 1
                push(s)
                continue

        S.remove(s)
        removed.append(s)
        steps -= 1

        for i in site_to_customers[s]:
            c_before = counts[i]
            if c_before == 0:
                continue
            counts[i] -= 1
            c_after = counts[i]

            if c_before == 1 and c_after == 0:
                total_cov -= 1

            if c_before == 2 and c_after == 1:
                for t in a[i]:
                    if t != s and t in S:
                        uniq_loss[t] = uniq_loss.get(t, 0) + 1
                        ver[t] = ver.get(t, 0) + 1
                        heapq.heappush(heap, (uniq_loss[t], len(site_to_customers[t]), t, ver[t]))
                        break

    return sorted(S), removed


def site_fixing_greedy_delta_fast(a, num_sites, required_coverage, max_iterations=None, fixed_open_sites=None):
    """Accelerated dual-branch constructive heuristic replacing heavy Python set ops with local array tracking loops."""
    site_to_customers = {j: set() for j in range(1, num_sites + 1)}
    for i, sites in enumerate(a):
        for s in sites:
            site_to_customers[s].add(i)

    all_sites = list(range(1, num_sites + 1))
    current_solution = set(fixed_open_sites or [])
    covered_customers = set()
    for s in current_solution:
        covered_customers.update(site_to_customers[s])

    best_solution = None
    best_count = float('inf')
    fixed_sites = {s: 1 for s in current_solution}
    iterations = 0

    while len(fixed_sites) < num_sites:
        if max_iterations is not None and iterations >= max_iterations:
            break
        pick_pool = [s for s in all_sites if s not in fixed_sites]
        if not pick_pool:
            break
        site = random.choice(pick_pool)
        iterations += 1

        # BRANCH A: FORCE SITE INCLUSION
        S_A = current_solution | {site}
        C_A = covered_customers.union(site_to_customers[site])
        remaining_A = [s for s in all_sites if s not in S_A and s not in fixed_sites]
        while len(C_A) < required_coverage and remaining_A:
            best_s = None
            best_gain = -1
            sc = site_to_customers
            Cref = C_A
            for cand in remaining_A:
                gain = 0
                for i in sc[cand]:
                    if i not in Cref:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_s = cand
            if best_s is None or best_gain <= 0:
                break
            S_A.add(best_s)
            C_A.update(site_to_customers[best_s])
            remaining_A.remove(best_s)
        A_ok = (len(C_A) >= required_coverage)
        size_A = len(S_A) if A_ok else float('inf')

        # BRANCH B: FORCE SITE EXCLUSION
        S_B = set(current_solution)
        C_B = set(covered_customers)
        remaining_B = [s for s in all_sites if s != site and s not in fixed_sites and s not in S_B]
        while len(C_B) < required_coverage and remaining_B:
            best_s = None
            best_gain = -1
            sc = site_to_customers
            Cref = C_B
            for cand in remaining_B:
                gain = 0
                for i in sc[cand]:
                    if i not in Cref:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_s = cand
            if best_s is None or best_gain <= 0:
                break
            S_B.add(best_s)
            C_B.update(site_to_customers[best_s])
            remaining_B.remove(best_s)
        B_ok = (len(C_B) >= required_coverage)
        size_B = len(S_B) if B_ok else float('inf')

        if A_ok and (size_A <= size_B or not B_ok):
            current_solution = S_A
            covered_customers = C_A
            fixed_sites[site] = 1
        else:
            current_solution = S_B
            covered_customers = C_B
            fixed_sites[site] = 0

        cur_len = len(current_solution)
        if cur_len < best_count:
            best_count = cur_len
            best_solution = set(current_solution)

    if best_solution is None:
        best_solution = set(current_solution)
    return best_count, sorted(best_solution), len(covered_customers), round(time.time(), 2)


class MilpRepairer:
    """
    Persistent, Reusable Optimization Interface to DOCPLEX CPLEX Engine.
    Keeps mathematical objects cached and updates upper/lower bounds dynamically
    to execute high-speed subproblem repairs without compilation overheads.
    """

    def __init__(self, a, num_customers, num_sites, required_coverage,
                 timelimit=1.0, threads=None, log_output=False):
        self.num_customers = num_customers
        self.num_sites = num_sites
        self.required_coverage = required_coverage

        m = Model(name="MILP_Repair", log_output=log_output)
        x = m.binary_var_list(num_customers, name="x")
        y = m.binary_var_list(num_sites, name="y")

        for i in range(num_customers):
            m.add_constraint(m.sum(y[j - 1] for j in a[i]) >= x[i])
        m.add_constraint(m.sum(x) >= required_coverage)
        m.minimize(m.sum(y))

        if threads is not None:
            try:
                m.parameters.threads = threads
            except Exception:
                pass
        m.parameters.timelimit = timelimit
        m.parameters.mip.display = 0

        self.model = m
        self.x = x
        self.y = y

    def solve_with_fixed(self, fixed_sites, timelimit=None):
        """Injects neighborhood restrictions as variable bounds prior to mathematical resolution."""
        y = self.y
        m = self.model
        if timelimit is not None:
            m.parameters.timelimit = timelimit

        to_restore = []
        for j in fixed_sites:
            var = y[j - 1]
            lb, ub = var.lb, var.ub
            if lb != 1 or ub != 1:
                to_restore.append((var, lb, ub))
                var.lb = 1
                var.ub = 1

        sol = m.solve()
        if not sol:
            for var, lb, ub in to_restore:
                var.lb = lb
                var.ub = ub
            return None, 0

        selected_sites = [j + 1 for j in range(self.num_sites) if y[j].solution_value > 0.5]
        covered_customers = sum(1 for i in range(self.num_customers) if self.x[i].solution_value > 0.5)

        # Restore bounds to keep memory clean for subsequent LNS metaheuristic updates
        for var, lb, ub in to_restore:
            var.lb = lb
            var.ub = ub

        return selected_sites, covered_customers


def milp_repair_fast(a, num_customers, num_sites, required_coverage, fixed_sites, timelimit=1, _cache={}):
    """Wrapper function accessing the persistent object cache for fast MILP subproblem fixes."""
    key = (id(a), num_customers, num_sites, required_coverage)
    rep = _cache.get(key)
    if rep is None:
        rep = MilpRepairer(a, num_customers, num_sites, required_coverage, timelimit=timelimit, threads=None,
                           log_output=False)
        _cache[key] = rep
    return rep.solve_with_fixed(fixed_sites, timelimit=timelimit)


# ==============================================================================
# 6. SOLUTION CLEANUP / POST-PROCESSING
# ==============================================================================

def drop_cleanup(solution, site_to_customers, required_coverage):
    """
    Executes a fast greedy cleanup pass over the solution array.
    Drops any location whose coverage overlap is entirely redundant.
    """
    S = set(solution)
    covered_counts = {}
    for s in S:
        for i in site_to_customers[s]:
            covered_counts[i] = covered_counts.get(i, 0) + 1

    for s in list(S):
        can_drop = True
        for i in site_to_customers[s]:
            if covered_counts.get(i, 0) == 1:
                can_drop = False
                break
        if not can_drop:
            continue

        S.remove(s)
        for i in site_to_customers[s]:
            covered_counts[i] -= 1
            if covered_counts[i] == 0:
                del covered_counts[i]

        if len(covered_counts) < required_coverage:
            S.add(s)
            for i in site_to_customers[s]:
                covered_counts[i] = covered_counts.get(i, 0) + 1

    return sorted(S)


# ==============================================================================
# 7. MAIN ENFORCED LNS EXECUTION PIPELINE
# ==============================================================================

def _now(): return time.time()


def _remaining(deadline): return deadline - _now()


def lns_with_dual_repair(a, num_customers, num_sites, required_coverage,
                         time_limit=60, destroy_fraction=0.2, destroy_mode="mixed",
                         greedy_keep_feasible=False, milp_time_limit=1, verbose=False,
                         speed_opt=False, milp_or_greedy=0.5, safety_buffer=0.15, worst_solutions=False, percentage=0):
    """
    Main Metaheuristic Pipeline for LNS.
    Coordinates randomized neighborhood destruction and dual-engine repairs.
    """
    t0 = _now()
    deadline = t0 + time_limit
    site_to_customers = build_site_to_customers(a, num_sites)

    # Dynamic algorithmic linking based on system flags
    destroy_fn = greedy_destroy_min_loss_fast if speed_opt else greedy_destroy_min_loss_baseline
    greedy_fix_fn = site_fixing_greedy_delta_fast if speed_opt else site_fixing_greedy_delta_baseline
    milp_fn = (lambda *args, **kw: milp_repair_fast(*args, **kw)) if speed_opt else (
        lambda *args, **kw: milp_repair_baseline(*args, **kw))

    # Constructive initialization seeding pass
    init_sites, init_cov = fast_forward_greedy_seed(a, num_customers, num_sites, required_coverage)
    the_best_solution = init_sites[:]
    the_best_count = len(the_best_solution)
    best_solution = init_sites[:]
    best_count = len(best_solution)
    time_to_best = _now() - t0
    if verbose:
        print(f"initial: |S|={best_count}, covered={init_cov}/{num_customers}")

    forced_random_mode = False

    # Primary Search Loop
    while True:
        if _remaining(deadline) <= safety_buffer:
            break

        prev_solution_set = set(best_solution)
        destroy_k = max(1, int(len(best_solution) * destroy_fraction))

        # Destruction strategy configuration
        if forced_random_mode:
            use_greedy = False
        else:
            if destroy_mode == "greedy":
                use_greedy = True
            elif destroy_mode == "random":
                use_greedy = False
            else:
                use_greedy = (random.random() < 0.5)

        if use_greedy and len(best_solution) > 1:
            fixed_part, removed = destroy_fn(
                best_solution, site_to_customers, num_customers, destroy_k,
                require_feasible_after=greedy_keep_feasible,
                required_coverage=required_coverage
            )
            if not removed:
                removed = random.sample(best_solution, min(destroy_k, len(best_solution) - 1))
                fixed_part = [s for s in best_solution if s not in removed]
        else:
            removed = random.sample(best_solution, min(destroy_k, len(best_solution) - 1))
            fixed_part = [s for s in best_solution if s not in removed]

        if _remaining(deadline) <= safety_buffer:
            break

        # Stochastic choice between MILP and constructive Greedy subproblem solvers
        repair_method = "milp" if (random.random() < milp_or_greedy) else "greedy"
        repaired_solution = None
        covered = -1

        if repair_method == "greedy":
            _, repaired_solution, covered, _ = greedy_fix_fn(
                a, num_sites, required_coverage, fixed_open_sites=fixed_part
            )
            if _remaining(deadline) <= safety_buffer:
                break
        else:
            milp_cap = min(milp_time_limit, max(0.0, _remaining(deadline) - safety_buffer))
            if milp_cap <= 0.0:
                break
            rs, covered = milp_fn(
                a, num_customers, num_sites, required_coverage, fixed_part, timelimit=milp_cap
            )
            repaired_solution = rs
            if _remaining(deadline) <= safety_buffer:
                break

        improved = False
        changed = False

        if repaired_solution is not None:
            repaired_solution = drop_cleanup(repaired_solution, site_to_customers, required_coverage)
            changed = (set(repaired_solution) != prev_solution_set)

            # Dominance criterion validation evaluation
            if covered >= required_coverage and len(repaired_solution) <= the_best_count:
                if len(repaired_solution) < the_best_count:
                    time_to_best = _now() - t0
                the_best_solution = repaired_solution
                the_best_count = len(repaired_solution)
                best_solution = repaired_solution
                best_count = len(repaired_solution)
                improved = True
                forced_random_mode = False
                if verbose:
                    print(f"{'greedy-destroy' if use_greedy else 'random-destroy'} + {repair_method}  "
                          f"{_now() - t0:.2f}  best={best_count}")
            # Non-improving exploration step activation (Ablation configuration support)
            elif covered >= required_coverage and len(repaired_solution) > the_best_count and worst_solutions == True:
                if random.random() < percentage:
                    best_solution = repaired_solution
                    best_count = len(repaired_solution)
        else:
            changed = False

        # Trap handler to mitigate stagnation / local minima traps
        if use_greedy and not changed and not improved:
            forced_random_mode = True

        if _remaining(deadline) <= safety_buffer:
            break

    return the_best_count, sorted(the_best_solution), time_to_best


# ==============================================================================
# 8. BATCH EXECUTION MONITOR & EXPORT LOGISTICS
# ==============================================================================

if __name__ == "__main__":
    # initial settings
    SPEED_OPT = False
    milp_or_greedy = 0.5
    worst_solutions = False
    percentage = 0

    # fill the relevant filepaths
    filepaths = [
        "cleanedscpcyc06.xlsx", "cleanedscpcyc06_6Cycle.xlsx", "cleanedscpcyc06_8Cycle.xlsx",
        "cleanedscpcyc07.xlsx", "cleanedscpcyc07_6Cycle.xlsx", "cleanedscpcyc07_8Cycle.xlsx",
        "cleanedscpcyc08.xlsx", "cleanedscpcyc08_6Cycle.xlsx", "cleanedscpcyc08_8Cycle.xlsx",
        "cleanedscpcyc09.xlsx", "cleanedscpcyc09_6Cycle.xlsx", "cleanedscpcyc09_8Cycle.xlsx",
        "cleanedscpcyc10.xlsx", "cleanedscpcyc10_6Cycle.xlsx", "cleanedscpcyc10_8Cycle.xlsx",
        "cleanedscpcyc11.xlsx", "cleanedscpcyc11_6Cycle.xlsx", "cleanedscpcyc11_8Cycle.xlsx",
        "cleanedscpcyc12_4Cycle.xlsx", "cleanedscpcyc12_6Cycle.xlsx", "cleanedscpcyc12_8Cycle.xlsx",
    ]

    runs_per_instance = 10  # choose number of runs per instance
    coverages = [0.9, 0.95]  # choose coverages
    milp_time_limit = 1  # initial settings
    TIME_LIMIT_SECONDS = 100  # time limit
    VERBOSE = False  # change to TRUE if you want to follow the algorithm progress
    RESULTS_XLSX = "lns_algorithm_solution_results.xlsx"

    rows = []
    for filepath in filepaths:
        for coverage in coverages:
            if not os.path.exists(filepath):
                print(f"⚠️ Missing file skipped: {filepath}")
                continue

            a, num_customers, num_sites = loading(filepath)
            required_coverage = int(num_customers * coverage)

            worst_solutions = False
            percentage = 0
            SPEED_OPT = False

            # Configure algorithmic parameters across instance sizes dynamically
            if filepath in ("cleanedscpcyc06.xlsx", "cleanedscpcyc06_6Cycle.xlsx", "cleanedscpcyc06_8Cycle.xlsx"):
                destroy_fraction = 0.4
                milp_or_greedy = 0.5
                milp_time_limit = 1
            elif filepath in ("cleanedscpcyc07.xlsx", "cleanedscpcyc07_6Cycle.xlsx", "cleanedscpcyc07_8Cycle.xlsx"):
                destroy_fraction = 0.4
                milp_or_greedy = 0.5
                milp_time_limit = 1
                worst_solutions = True
                percentage = 0.3
            elif filepath in ("cleanedscpcyc08.xlsx", "cleanedscpcyc08_6Cycle.xlsx", "cleanedscpcyc08_8Cycle.xlsx"):
                destroy_fraction = 0.25
                milp_or_greedy = 0.6
                milp_time_limit = 1.5
                worst_solutions = True
                percentage = 0.1
            elif filepath in ("cleanedscpcyc09.xlsx", "cleanedscpcyc09_6Cycle.xlsx", "cleanedscpcyc09_8Cycle.xlsx"):
                destroy_fraction = 0.25
                milp_or_greedy = 0.8
                milp_time_limit = 1.5
            elif filepath in ("cleanedscpcyc10.xlsx", "cleanedscpcyc10_6Cycle.xlsx", "cleanedscpcyc10_8Cycle.xlsx"):
                destroy_fraction = 0.2
                milp_or_greedy = 0.95
                milp_time_limit = 2
                SPEED_OPT = True
            elif filepath in ("cleanedscpcyc11.xlsx", "cleanedscpcyc11_6Cycle.xlsx", "cleanedscpcyc11_8Cycle.xlsx"):
                destroy_fraction = 0.15
                milp_or_greedy = 1
                milp_time_limit = 5
                SPEED_OPT = True
            else:
                destroy_fraction = 0.1
                milp_or_greedy = 1
                milp_time_limit = 5
                SPEED_OPT = True

            print(f"\n🚀 Processing: {filepath} | Coverage: {coverage} | Speed Opt: {SPEED_OPT}")

            for run_idx in range(1, runs_per_instance + 1):
                print(f"  -> Run {run_idx}/10")
                best_val, best_sites, t_best = lns_with_dual_repair(
                    a, num_customers, num_sites, required_coverage,
                    time_limit=TIME_LIMIT_SECONDS,
                    destroy_fraction=destroy_fraction,
                    milp_time_limit=milp_time_limit,
                    verbose=VERBOSE,
                    speed_opt=SPEED_OPT,
                    milp_or_greedy=milp_or_greedy,
                    worst_solutions=worst_solutions,
                    percentage=percentage
                )
                print("  🔧 LNS Result:", best_val, "sites")

                solution_string = ", ".join(map(str, best_sites))

                rows.append({
                    "instance": filepath,
                    "coverage": coverage,
                    "run_number": run_idx,
                    "best_value": best_val,
                    "time_to_best_sec": round(t_best, 3),
                    "solution_sites": solution_string
                })

    if rows:
        df = pd.DataFrame(rows)
        output_cols = ["instance", "coverage", "run_number", "best_value", "time_to_best_sec", "solution_sites"]
        df = df[output_cols]
        df.to_excel(RESULTS_XLSX, index=False)
        print(f"\n✨ Completed! All solution sets written to: {RESULTS_XLSX}")
    else:
        print("\n❌ No experiments were logged. Ensure files match the required names.")