#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <unordered_map>
#include <sstream>
#include <filesystem>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>

using namespace std;
namespace fs = std::filesystem;


// Fast count for 0/1 selection vectors (avoids vector<bool> proxies)
static inline int count_selected_vec(const std::vector<uint8_t>& sel) {
    int c = 0;
    for (uint8_t x : sel) c += (x != 0);
    return c;
}

// Global verbosity flag
static bool G_QUIET = false;

// Global toggle: enable/disable N1 1-swap moves (frozen when false)
static bool G_ENABLE_N1_SWAP = true;

// Sensitivity parameters (overridable via PCSP_RL_CONST / PCSP_RL_IMP)
static int G_RL_CONST = 8;   // intensification run length after improvement
static int G_RL_IMP   = 128; // unselected-supplier candidate list size in 1-swap

// Ablation toggles (set to false via PCSP_DISABLE_DROP / _SWAP / _N2)
static bool G_ENABLE_N1_DROP = true; // (i)   zero-loss drop moves
static bool G_ENABLE_N2      = true; // (iii) 2-to-1 replacement moves

// Problem parameters
int N, M;  // Number of customers and suppliers
vector<vector<bool>> C;  // Coverage matrix: C[n][m] = true if supplier m covers customer n
double TARGET_COVERAGE = 0.9;  // Default 90% coverage target (now configurable)
int COVERABLE_N = 0;                 // number of customers covered by at least one supplier

// Adaptive GA params (tunable)
static const double MIN_MUTATION_RATE = 0.02;
static const double MAX_MUTATION_RATE = 0.25;
static const int DIVERSE_MATE_SAMPLE = 10; // candidates to sample for distant mate selection

// Adjacency for fast greedy
static vector<vector<int>> customers_of_supplier; // for each supplier m, list of customers it covers
static vector<int> degree_customer;               // for each customer n, how many suppliers cover it

static vector<vector<int>> suppliers_of_customer; // for each customer n, list of suppliers covering it

// Run-time statistics for instrumentation
struct RunStats {
    long long ls_calls = 0;
    long long n1_drop_ops = 0;
    long long n1_swap_ops = 0;
    long long n2_det_ops = 0;
    long long n2_rand_ops = 0;
    long long repair_added = 0;
    long long prune_removed = 0;
    long long shake_dropped = 0;
    long long vns_iters = 0;
    long long vns_accepts = 0;
};

// Solution representation
struct Solution {
    // 0/1 selection vector (uint8_t is much faster than vector<bool>)
    std::vector<uint8_t> selected;  // selected[m] = 1 if supplier m is selected
    int num_selected;               // Number of selected suppliers
    int covered;                    // Number of covered customers
    double fitness;                 // For comparison

    Solution() : num_selected(0), covered(0), fitness(0) {
        selected.assign(M, 0);
    }

    // Evaluate using sparse adjacency (customers_of_supplier) instead of scanning the full matrix
    void evaluate() {
        num_selected = count_selected_vec(selected);
        // thread_local reuse to reduce allocations
        static thread_local std::vector<int> coverage_count;
        if ((int)coverage_count.size() != N) coverage_count.assign(N, 0);
        else std::fill(coverage_count.begin(), coverage_count.end(), 0);

        covered = 0;
        for (int m = 0; m < M; ++m) {
            if (!selected[m]) continue;
            for (int n : customers_of_supplier[m]) {
                if (coverage_count[n]++ == 0) ++covered;
            }
        }

        int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
        fitness = (covered >= target_required) ?
                1.0 / (1.0 + num_selected) :   // Feasible: minimize suppliers
                (double)covered / N;           // Infeasible: maximize coverage
    }

    bool is_feasible() const {
        int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
        return covered >= target_required;
    }

    // Jaccard distance between two solutions (1 - |A∩B|/|A∪B|)
    double jaccard_distance(const Solution& other) const {
        int intersection = 0;
        int union_size = 0;

        for (int m = 0; m < M; ++m) {
            const bool a = (selected[m] != 0);
            const bool b = (other.selected[m] != 0);
            if (a && b) ++intersection;
            if (a || b) ++union_size;
        }

        if (union_size == 0) return 0.0;  // Both solutions are empty

        double jaccard_sim = (double)intersection / union_size;
        return 1.0 - jaccard_sim;
    }

    // Jaccard distance between customer coverage sets (built via adjacency)
    double customer_coverage_distance(const Solution& other) const {
        static thread_local std::vector<uint8_t> cov1, cov2;
        if ((int)cov1.size() != N) { cov1.assign(N, 0); cov2.assign(N, 0); }
        else { std::fill(cov1.begin(), cov1.end(), 0); std::fill(cov2.begin(), cov2.end(), 0); }

        for (int m = 0; m < M; ++m) {
            if (selected[m]) {
                for (int n : customers_of_supplier[m]) cov1[n] = 1;
            }
            if (other.selected[m]) {
                for (int n : customers_of_supplier[m]) cov2[n] = 1;
            }
        }

        int intersection = 0, union_size = 0;
        for (int n = 0; n < N; ++n) {
            if (cov1[n] && cov2[n]) ++intersection;
            if (cov1[n] || cov2[n]) ++union_size;
        }

        return union_size > 0 ? 1.0 - (double)intersection / union_size : 0.0;
    }

    // Hybrid distance combining supplier and customer coverage distances
    double hybrid_distance(const Solution& other, double alpha = 0.7) const {
        double supplier_dist = jaccard_distance(other);
        double customer_dist = customer_coverage_distance(other);
        return alpha * supplier_dist + (1.0 - alpha) * customer_dist;
    }

    void print() const {
        std::cout << "Suppliers: " << num_selected
                  << " | Coverage: " << covered << "/" << N << " ("
                  << std::fixed << std::setprecision(1) << (100.0 * covered / N) << "%)"
                  << " | Fitness: " << std::setprecision(6) << fitness
                  << (is_feasible() ? " | FEASIBLE" : "")
                  << "\n";
}
};

// Forward declarations (defaults only here) (defaults only here)
static void local_search_improve(Solution& s, int time_limit_ms = 200, RunStats* stats = nullptr);
static void fast_prune_to_minimal(Solution& s, RunStats* stats = nullptr);
static void fast_repair_to_target(Solution& s, RunStats* stats = nullptr, const std::vector<char>* avoid_add = nullptr);
static int shake_drop_random(Solution& s, int r, std::mt19937& rng, std::vector<int>* dropped_ms = nullptr);
static bool ejection_chain2(Solution& s, std::mt19937& rng, int time_limit_ms, RunStats* stats = nullptr);
// N2 consolidation (feasible 2->1) pass
static bool n2_consolidate_pass(Solution& s, std::mt19937& rng, int time_limit_ms = 60, RunStats* stats = nullptr);
static bool some_function_name_placeholder(Solution& s, std::mt19937& rng, int time_limit_ms, RunStats* stats);

static thread_local std::mt19937* G_SOLVER_RNG = nullptr;

// Helper: fast repair to target coverage using adjacency and counters
static void fast_repair_to_target(Solution& s, RunStats* stats, const std::vector<char>* avoid_add) {
    const int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
    // Build coverage counters from current selection
    vector<int> coverage_count(N, 0);
    int covered = 0;
    for (int m = 0; m < M; ++m) if (s.selected[m]) {
        for (int n : customers_of_supplier[m]) {
            if (coverage_count[n]++ == 0) ++covered;
        }
    }
    s.covered = covered;
    s.num_selected = (int)count_selected_vec(s.selected);

    auto choose_and_add = [&](bool respect_avoid) -> bool {
        const int TOPK = 8;
        struct Cand { int gain; double w; int m; };
        std::vector<Cand> cand; cand.reserve(M);
        int best_gain = 0;
        for (int m = 0; m < M; ++m) if (!s.selected[m]) {
            if (respect_avoid && avoid_add && m < (int)avoid_add->size() && (*avoid_add)[m]) continue;
            int gain = 0;
            double w = 0.0;
            for (int n : customers_of_supplier[m]) {
                if (coverage_count[n] == 0) {
                    ++gain;
                    int d = degree_customer[n] > 0 ? degree_customer[n] : 1;
                    w += 1.0 / (double)d;
                }
            }
            if (gain <= 0) continue;
            cand.push_back({gain, w, m});
            if (gain > best_gain) best_gain = gain;
        }
        if (best_gain <= 0 || cand.empty()) return false;

        int K = std::min(TOPK, (int)cand.size());
        std::nth_element(cand.begin(), cand.begin() + K, cand.end(), [](const Cand& a, const Cand& b){
            return a.gain > b.gain;
        });
        cand.resize(K);
        int cutoff = std::max(1, best_gain - 1);
        std::vector<Cand> rcl; rcl.reserve(cand.size());
        for (const auto& p : cand) if (p.gain >= cutoff) rcl.push_back(p);
        if (rcl.empty()) rcl = cand;

        int best_m = rcl[0].m;
        double best_w = rcl[0].w;
        int ties = 1;
        for (int i = 1; i < (int)rcl.size(); ++i) {
            if (rcl[i].w > best_w + 1e-12) {
                best_w = rcl[i].w;
                best_m = rcl[i].m;
                ties = 1;
            } else if (std::abs(rcl[i].w - best_w) <= 1e-12) {
                ++ties;
                if (G_SOLVER_RNG) {
                    std::uniform_int_distribution<int> d(1, ties);
                    if (d(*G_SOLVER_RNG) == 1) best_m = rcl[i].m;
                }
            }
        }
        s.selected[best_m] = true;
        ++s.num_selected;
        for (int n : customers_of_supplier[best_m]) {
            if (coverage_count[n]++ == 0) ++covered;
        }
        s.covered = covered;
        if (stats) ++stats->repair_added;
        return true;
    };

    while (covered < target_required) {
        // First try: avoid immediate re-adding recently dropped suppliers.
        if (choose_and_add(true)) continue;
        // Fallback: if avoidance blocks feasibility, allow any supplier.
        if (!choose_and_add(false)) break;
    }
    s.fitness = (covered >= target_required) ? 1.0 / (1.0 + s.num_selected) : (double)covered / N;
}

// Helper: fast prune to minimal using adjacency and counters
static void fast_prune_to_minimal(Solution& s, RunStats* stats) {
    const int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
    // Build coverage counters from current selection
    vector<int> coverage_count(N, 0);
    int covered = 0;
    for (int m = 0; m < M; ++m) if (s.selected[m]) {
        for (int n : customers_of_supplier[m]) {
            if (coverage_count[n]++ == 0) ++covered;
        }
    }
    // Try removals in passes
    bool improved = true;
    while (improved) {
        improved = false;
        for (int m = 0; m < M; ++m) if (s.selected[m]) {
            int would_uncover = 0;
            for (int n : customers_of_supplier[m]) if (coverage_count[n] == 1) ++would_uncover;
            if (covered - would_uncover >= target_required) {
                s.selected[m] = false;
                for (int n : customers_of_supplier[m]) {
                    if (--coverage_count[n] == 0) --covered;
                }
                improved = true;
                if (stats) ++stats->prune_removed;
            }
        }
    }
    s.num_selected = (int)count_selected_vec(s.selected);
    s.covered = covered;
    s.fitness = (covered >= target_required) ? 1.0 / (1.0 + s.num_selected) : (double)covered / N;
}

// Read input file
void read_input(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }
    // Enlarge internal buffer to reduce I/O calls
    static vector<char> file_buf(1 << 20); // 1MB buffer
    fin.rdbuf()->pubsetbuf(file_buf.data(), (streamsize)file_buf.size());

    // Fast integer reader using streambuf
    auto* sb = fin.rdbuf();
    auto fast_read_int = [&](int& out) -> bool {
        int c = sb->sgetc();
        // Skip whitespace
        while (c != EOF && (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f')) {
            c = sb->snextc();
        }
        if (c == EOF) return false;
        int val = 0;
        // Read digits (inputs are non-negative)
        while (c != EOF && (c >= '0' && c <= '9')) {
            val = val * 10 + (c - '0');
            c = sb->snextc();
        }
        out = val;
        return true;
    };

    // Read N and M
    if (!fast_read_int(N) || !fast_read_int(M)) { cerr << "Error: Failed to read N and M" << endl; exit(1); }

    // Skip supplier costs (M numbers)
    int skip;
    for (int i = 0; i < M; ++i) {
        if (!fast_read_int(skip)) { cerr << "Error: Failed to read costs" << endl; exit(1); }
    }

    // Initialize C and adjacency with the correct dimensions
    C.resize(N, vector<bool>(M, false));
    customers_of_supplier.assign(M, {});
    degree_customer.assign(N, 0);
    suppliers_of_customer.assign(N, {});

    // Read coverage matrix
    for (int n = 0; n < N; ++n) {
        int k, m;
        if (!fast_read_int(k)) { cerr << "Error: Failed to read supplier count k for customer " << n+1 << "\n"; exit(1); }
        while (k--) {
            if (!fast_read_int(m)) { cerr << "Error: Failed to read supplier id for customer " << n+1 << "\n"; exit(1); }
            if (m < 1 || m > M) { cerr << "Error: Supplier id out of range (" << m << ") at customer " << n+1 << "; expected 1.." << M << "\n"; exit(1); }
            int midx = m-1;
            if (!C[n][midx]) {
                C[n][midx] = true;  // Convert to 0-based
                customers_of_supplier[midx].push_back(n);
                degree_customer[n]++;
                suppliers_of_customer[n].push_back(midx);
            }
        }
        // Progress printing is intentionally minimized for performance on large files.
    }
    // Compute COVERABLE_N (customers with at least one covering supplier)
    COVERABLE_N = 0;
    for (int n = 0; n < N; ++n) if (degree_customer[n] > 0) ++COVERABLE_N;
    {
        const bool ENABLE_PREPROCESS_DUP = false;
        if (ENABLE_PREPROCESS_DUP) {
            // Normalize customer lists (sort unique)
            for (int m = 0; m < M; ++m) {
                auto& v = customers_of_supplier[m];
                sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());
            }
            // Detect duplicates via string signature
            unordered_map<string,int> rep;
            vector<char> is_dup(M, 0);
            int dup_removed = 0;
            for (int m = 0; m < M; ++m) {
                const auto &lst = customers_of_supplier[m];
                if (lst.empty()) continue;
                ostringstream os; bool first = true;
                for (int n : lst) { if (!first) os << ','; os << n; first = false; }
                string key = os.str();
                auto it = rep.find(key);
                if (it == rep.end()) rep.emplace(move(key), m);
                else { is_dup[m] = 1; ++dup_removed; }
            }
            // Clear duplicate suppliers' coverage
            for (int m = 0; m < M; ++m) if (is_dup[m]) customers_of_supplier[m].clear();
            // Rebuild suppliers_of_customer and degree_customer from scratch
            suppliers_of_customer.assign(N, {});
            degree_customer.assign(N, 0);
            for (int m = 0; m < M; ++m) for (int n : customers_of_supplier[m]) { suppliers_of_customer[n].push_back(m); ++degree_customer[n]; }
            // Recompute C sparse flags for consistency
            for (int n = 0; n < N; ++n) fill(C[n].begin(), C[n].end(), false);
            for (int m = 0; m < M; ++m) for (int n : customers_of_supplier[m]) C[n][m] = true;
            // Recompute COVERABLE_N after duplicates removal
            COVERABLE_N = 0; for (int n = 0; n < N; ++n) if (degree_customer[n] > 0) ++COVERABLE_N;
            // Log
            int activeM = 0; for (int m = 0; m < M; ++m) if (!customers_of_supplier[m].empty()) ++activeM;
            static bool dup_logged_once = false;
            if (!dup_logged_once) {
                dup_logged_once = true;
            }
        } else {
        }
    }
    // Preprocessing: column dominance (subset) removal for small columns
    {
        const bool ENABLE_PREPROCESS_DOM = false;
        if (ENABLE_PREPROCESS_DOM) {
            // Ensure each supplier's list is sorted
            for (int m = 0; m < M; ++m) {
                auto& v = customers_of_supplier[m];
                sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());
            }

            const int DOM_CHECK_LIMIT = 64; // only attempt subset checks for small columns
            int removed_dom = 0;

            // Helper to check if set A (list of customers) is subset of set B (both sorted)
            auto is_subset = [](const vector<int>& A, const vector<int>& B) -> bool {
                if (A.empty()) return true;
                if (B.size() < A.size()) return false;
                size_t i = 0, j = 0;
                while (i < A.size() && j < B.size()) {
                    if (A[i] == B[j]) { ++i; ++j; }
                    else if (A[i] > B[j]) { ++j; }
                    else { return false; }
                }
                return i == A.size();
            };

            // Build quick access to suppliers covering a customer
            // (we already have suppliers_of_customer)

            vector<char> removed(M, 0);
            // Iterate columns by increasing cardinality up to limit
            vector<pair<int,int>> order; order.reserve(M);
            for (int m = 0; m < M; ++m) if (!customers_of_supplier[m].empty()) order.emplace_back((int)customers_of_supplier[m].size(), m);
            sort(order.begin(), order.end());

            for (auto [sz, j] : order) {
                if (removed[j] || sz == 0 || sz > DOM_CHECK_LIMIT) continue;
                const auto& Sj = customers_of_supplier[j];
                if (Sj.empty()) continue;
                // candidates: suppliers that cover 1-2 pivot customers from j, with size >= sz
                int p1 = Sj[0];
                int p2 = (sz >= 2 ? Sj[1] : -1);
                const auto& cand1 = suppliers_of_customer[p1];
                vector<int> cand;
                cand.reserve(cand1.size());
                if (p2 >= 0) {
                    // intersection of cand1 and suppliers_of_customer[p2]
                    const auto& cand2 = suppliers_of_customer[p2];
                    size_t a = 0, b = 0;
                    while (a < cand1.size() && b < cand2.size()) {
                        if (cand1[a] == cand2[b]) { cand.push_back(cand1[a]); ++a; ++b; }
                        else if (cand1[a] < cand2[b]) ++a; else ++b;
                    }
                } else {
                    cand = cand1;
                }
                bool dominated = false;
                for (int i : cand) {
                    if (i == j || removed[i]) continue;
                    const auto& Si = customers_of_supplier[i];
                    if ((int)Si.size() < sz) continue;
                    if (is_subset(Sj, Si)) { dominated = true; break; }
                }
                if (dominated) {
                    // Remove dominated column j
                    for (int n : Sj) C[n][j] = false;
                    removed[j] = 1; ++removed_dom;
                }
            }

            if (removed_dom > 0) {
                // Rebuild adjacency after dominance removals
                customers_of_supplier.assign(M, {});
                degree_customer.assign(N, 0);
                suppliers_of_customer.assign(N, {});
                for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m) if (C[n][m]) {
                    customers_of_supplier[m].push_back(n);
                    suppliers_of_customer[n].push_back(m);
                    ++degree_customer[n];
                }
                COVERABLE_N = 0; for (int n = 0; n < N; ++n) if (degree_customer[n] > 0) ++COVERABLE_N;
            }
        } else {
        }
    }
}



// Greedy solution generator (respects a hard deadline)
Solution greedy_solution(int seed, const chrono::steady_clock::time_point& deadline) {
    mt19937 rng(seed);
    Solution s;
    vector<bool> covered(N, false);
    
    while (true) {
        // Respect global greedy deadline
        if (chrono::steady_clock::now() >= deadline) {
            break;
        }
        int best_m = -1;
        int best_gain = -1;
        
        // Try all suppliers in random order
        vector<int> suppliers(M);
        iota(suppliers.begin(), suppliers.end(), 0);
        shuffle(suppliers.begin(), suppliers.end(), rng);
        
        for (int m : suppliers) {
            if (s.selected[m]) continue;
            
            int gain = 0;
            for (int n = 0; n < N; ++n) {
                if (!covered[n] && C[n][m]) gain++;
            }
            
            if (gain > best_gain) {
                best_gain = gain;
                best_m = m;
            }
        }
        
        if (best_m == -1) break;  // No improvement possible
        
        // Add best supplier
        s.selected[best_m] = true;
        s.num_selected++;
        for (int n = 0; n < N; ++n) {
            if (C[n][best_m]) covered[n] = true;
        }
        
        // Update coverage
        s.covered = count(covered.begin(), covered.end(), true);
        
        // Stop if we've reached the target coverage
        int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
        if (s.covered >= target_required) break;
    }
    
    s.evaluate();
    return s;
}

// Find two most distant solutions in population using hybrid distance
pair<Solution, Solution> find_most_distant(const vector<Solution>& population) {
    double max_dist = -1.0;
    int best_i = 0, best_j = 1;
    
    for (int i = 0; i < population.size(); ++i) {
        for (int j = i + 1; j < population.size(); ++j) {
            double dist = population[i].hybrid_distance(population[j]);
            if (dist > max_dist) {
                max_dist = dist;
                best_i = i;
                best_j = j;
            }
        }
    }
    
    return {population[best_i], population[best_j]};
}

// Calculate coverage of a single supplier
int calculate_supplier_coverage(int m) {
    int count = 0;
    for (int n = 0; n < N; ++n) {
        if (C[n][m]) count++;
    }
    return count;
}

// Calculate marginal coverage of a supplier given current coverage
int calculate_marginal_coverage(int m, const vector<bool>& covered) {
    int marginal = 0;
    for (int n = 0; n < N; ++n) {
        if (C[n][m] && !covered[n]) {
            marginal++;
        }
    }
    return marginal;
}

// Compute average Jaccard diversity across population (sampled for speed)
static double average_jaccard_diversity(const vector<Solution>& pop, int sample_pairs = 200) {
    int P = (int)pop.size();
    if (P < 2) return 0.0;
    mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(0, P - 1);
    double sum = 0.0; int cnt = 0;
    int max_pairs = min(sample_pairs, P * (P - 1) / 2);
    for (int k = 0; k < max_pairs; ++k) {
        int i = dist(rng), j = dist(rng);
        if (i == j) continue;
        sum += pop[i].jaccard_distance(pop[j]);
        ++cnt;
    }
    return cnt ? (sum / cnt) : 0.0;
}

// Select second parent to maximize Jaccard distance from p1 among a random sample
static const Solution& select_diverse_mate(const vector<Solution>& pop, const Solution& p1, mt19937& rng, int sample_size) {
    uniform_int_distribution<int> dist(0, (int)pop.size() - 1);
    double best_d = -1.0; int best_idx = dist(rng);
    for (int t = 0; t < sample_size; ++t) {
        int idx = dist(rng);
        if (&pop[idx] == &p1) continue;
        double d = p1.jaccard_distance(pop[idx]);
        if (d > best_d) { best_d = d; best_idx = idx; }
    }
    return pop[best_idx];
}

// k-tournament selection (returns reference to best among k random individuals)
static const Solution& tournament_pick(const vector<Solution>& pop, int k, mt19937& rng) {
    uniform_int_distribution<int> dist(0, (int)pop.size() - 1);
    int best_idx = dist(rng);
    for (int i = 1; i < k; ++i) {
        int idx = dist(rng);
        if (pop[idx].fitness > pop[best_idx].fitness) best_idx = idx;
    }
    return pop[best_idx];
}

Solution union_prune_crossover(const Solution& p1, const Solution& p2) {
    Solution child;
    child.selected.assign(M, false);
    
    // Count suppliers in each parent
    int p1_count = 0, p2_count = 0, union_count = 0;
    vector<int> p1_suppliers, p2_suppliers, union_suppliers;
    
    for (int m = 0; m < M; ++m) {
        if (p1.selected[m]) {
            p1_count++;
            p1_suppliers.push_back(m);
        }
        if (p2.selected[m]) {
            p2_count++;
            p2_suppliers.push_back(m);
        }
        if (p1.selected[m] || p2.selected[m]) {
            child.selected[m] = true;
            union_count++;
            union_suppliers.push_back(m);
        }
    }
    
    child.evaluate();
    
    // 2. Calculate coverage for each customer
    vector<int> coverage_count(N, 0);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            if (child.selected[m] && C[n][m]) {
                coverage_count[n]++;
            }
        }
    }
    
    // 3. Prune suppliers one by one
    bool improved = true;
    while (improved) {
        improved = false;
        int best_to_remove = -1;
        int min_marginal = N + 1;
        
        // Find supplier with minimum marginal contribution
        for (int m = 0; m < M; ++m) {
            if (!child.selected[m]) continue;
            
            // Calculate how many customers would become uncovered
            int would_uncover = 0;
            for (int n = 0; n < N; ++n) {
                if (C[n][m] && coverage_count[n] == 1) {
                    would_uncover++;
                }
            }
            
            // Check if we can remove this supplier
            int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
            if (child.covered - would_uncover >= target_required) {
                if (would_uncover < min_marginal) {
                    min_marginal = would_uncover;
                    best_to_remove = m;
                    improved = true;
                }
            }
        }
        
        // Remove the best candidate if found
        if (improved) {
            child.selected[best_to_remove] = false;
            
            // Update coverage counts
            for (int n = 0; n < N; ++n) {
                if (C[n][best_to_remove]) {
                    coverage_count[n]--;
                    if (coverage_count[n] == 0) {
                        child.covered--;
                    }
                }
            }
        }
    }
    
    child.evaluate();
    return child;
}

// Intersection-Repair Crossover
Solution intersection_repair_crossover(const Solution& p1, const Solution& p2) {
    Solution child;
    child.selected.assign(M, false);
    
    // 1. Start with intersection of parents
    for (int m = 0; m < M; ++m) {
        if (p1.selected[m] && p2.selected[m]) {
            child.selected[m] = true;
        }
    }
    child.evaluate();
    
    // 2. Calculate coverage for each customer
    vector<int> coverage_count(N, 0);
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            if (child.selected[m] && C[n][m]) {
                coverage_count[n]++;
            }
        }
    }
    
    // 3. Add suppliers until feasible
    while (!child.is_feasible()) {
        int best_supplier = -1;
        int max_marginal = -1;
        
        // Find best supplier to add (from either parent)
        for (int m = 0; m < M; ++m) {
            if (!child.selected[m] && (p1.selected[m] || p2.selected[m])) {
                int marginal = 0;
                for (int n = 0; n < N; ++n) {
                    if (C[n][m] && coverage_count[n] == 0) {
                        marginal++;
                    }
                }
                
                if (marginal > max_marginal) {
                    max_marginal = marginal;
                    best_supplier = m;
                }
            }
        }
        
        if (best_supplier == -1) break;  // No more suppliers to add
        
        // Add the best supplier
        child.selected[best_supplier] = true;
        
        // Update coverage counts
        for (int n = 0; n < N; ++n) {
            if (C[n][best_supplier]) {
                if (coverage_count[n] == 0) {
                    child.covered++;
                }
                coverage_count[n]++;
            }
        }
    }
    
    // 4. Try to remove redundant suppliers
    for (int m = 0; m < M; ++m) {
        if (child.selected[m]) {
            // Check if we can remove this supplier
            bool can_remove = true;
            for (int n = 0; n < N; ++n) {
                if (C[n][m] && coverage_count[n] == 1) {
                    can_remove = false;
                    break;
                }
            }
            
            if (can_remove) {
                child.selected[m] = false;
                // Update coverage counts
                for (int n = 0; n < N; ++n) {
                    if (C[n][m]) {
                        coverage_count[n]--;
                    }
                }
            }
        }
    }
    
    child.evaluate();
    return child;
}

// Local Search: 1-drop and 1-swap with marginal gain caching; time-bounded
static bool some_function_name_placeholder(Solution& s, std::mt19937& rng, int time_limit_ms, RunStats* stats = nullptr) {
    const int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
    Solution t = s;
    auto t0 = chrono::steady_clock::now();
    auto elapsed_ms = [&](){ return (int)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t0).count(); };
    if (stats) ++stats->ls_calls;
    // Build coverage counters
    vector<int> cov(N, 0);
    for (int m = 0; m < M; ++m) if (s.selected[m]) for (int n : customers_of_supplier[m]) ++cov[n];
    s.covered = 0; for (int n = 0; n < N; ++n) if (cov[n] > 0) ++s.covered;
    s.num_selected = (int)count_selected_vec(s.selected);

    // Marginal caches
    vector<int> drop_loss(M, 0);  // customers that would be uncovered if removing m
    vector<int> add_gain(M, 0);   // customers newly covered if adding m
    for (int m = 0; m < M; ++m) {
        for (int n : customers_of_supplier[m]) {
            if (s.selected[m]) { if (cov[n] == 1) ++drop_loss[m]; }
            else { if (cov[n] == 0) ++add_gain[m]; }
        }
    }

    bool improved = true;
    while (improved && elapsed_ms() < time_limit_ms) {
        improved = false;

        // 1-drop: remove any zero-loss supplier
        if (G_ENABLE_N1_DROP)
        for (int m = 0; m < M && elapsed_ms() < time_limit_ms; ++m) if (s.selected[m] && drop_loss[m] == 0) {
            s.selected[m] = false; --s.num_selected;
            for (int n : customers_of_supplier[m]) {
                int before = cov[n];
                if (--cov[n] < 0) cov[n] = 0;
                if (before == 2) { // n becomes critical for neighbors
                    for (int q : suppliers_of_customer[n]) if (s.selected[q]) ++drop_loss[q];
                }
                if (before == 1) { // n just became uncovered
                    for (int q : suppliers_of_customer[n]) if (!s.selected[q]) ++add_gain[q];
                }
            }
            // refresh caches for m
            drop_loss[m] = 0; add_gain[m] = 0; for (int n : customers_of_supplier[m]) if (cov[n] == 0) ++add_gain[m];
            improved = true;
            if (stats) ++stats->n1_drop_ops;
        }
        if (improved) continue;

        if (G_ENABLE_N1_SWAP) { // 1-swap: choose (a selected, b not selected) with best gain-loss
            int best_a = -1, best_b = -1, best_delta = 0;
            const int K_SEL = 64;
            const int K_UNSEL = G_RL_IMP;
            vector<pair<int,int>> sel_pairs; sel_pairs.reserve(s.num_selected);
            vector<pair<int,int>> unsel_pairs; unsel_pairs.reserve(M - s.num_selected);
            for (int a = 0; a < M; ++a) if (s.selected[a]) sel_pairs.emplace_back(drop_loss[a], a);
            for (int b = 0; b < M; ++b) if (!s.selected[b]) unsel_pairs.emplace_back(add_gain[b], b);
            if (!sel_pairs.empty()) {
                int k = min((int)sel_pairs.size(), K_SEL);
                if ((int)sel_pairs.size() > k) {
                    nth_element(sel_pairs.begin(), sel_pairs.begin() + k, sel_pairs.end(), [](auto& A, auto& B){ return A.first < B.first; });
                    sel_pairs.resize(k);
                }
            }
            if (!unsel_pairs.empty()) {
                int k = min((int)unsel_pairs.size(), K_UNSEL);
                if ((int)unsel_pairs.size() > k) {
                    nth_element(unsel_pairs.begin(), unsel_pairs.begin() + k, unsel_pairs.end(), [](auto& A, auto& B){ return A.first > B.first; });
                    unsel_pairs.resize(k);
                }
            }
            for (const auto& [loss_raw, a] : sel_pairs) {
                if (elapsed_ms() >= time_limit_ms) break;
                int loss = loss_raw; if (loss < 0) loss = 0;
                for (const auto& [gain, b] : unsel_pairs) {
                    int delta = gain - loss;
                    if (s.covered - loss + gain < target_required) continue;
                    if (delta > best_delta) { best_delta = delta; best_a = a; best_b = b; }
                }
            }
            if (best_delta > 0 && best_a != -1) {
                int a = best_a, b = best_b;
                // Apply swap
                s.selected[a] = false; s.selected[b] = true;
                for (int n : customers_of_supplier[a]) {
                    int before = cov[n];
                    if (--cov[n] < 0) cov[n] = 0;
                    if (before == 2) {
                        for (int q : suppliers_of_customer[n]) if (s.selected[q]) ++drop_loss[q];
                    }
                    if (before == 1) {
                        for (int q : suppliers_of_customer[n]) if (!s.selected[q] && add_gain[q] > 0) --add_gain[q];
                    }
                }
                for (int n : customers_of_supplier[b]) {
                    int before = cov[n];
                    if (cov[n]++ == 0) ++s.covered;
                    if (before == 0) {
                        for (int q : suppliers_of_customer[n]) if (!s.selected[q] && add_gain[q] > 0) --add_gain[q];
                    } else if (before == 1) {
                        for (int q : suppliers_of_customer[n]) if (s.selected[q] && drop_loss[q] > 0) --drop_loss[q];
                    }
                }
                // refresh a and b caches
                drop_loss[a] = 0; add_gain[a] = 0; for (int n : customers_of_supplier[a]) if (cov[n] == 1) ++drop_loss[a];
                add_gain[b] = 0; drop_loss[b] = 0; for (int n : customers_of_supplier[b]) if (cov[n] == 0) ++add_gain[b];
                improved = true;
                if (stats) ++stats->n1_swap_ops;
            }
        }
        if (improved) continue;

        if (G_ENABLE_N2) { // N2: 2->1 replacement (net -1 supplier) with feasibility check
        // Build small candidate sets
        const int K_SEL2 = 48;
        const int K_UNSEL2 = 96;
        vector<pair<int,int>> sel_list; sel_list.reserve(s.num_selected);
        for (int m = 0; m < M; ++m) if (s.selected[m]) sel_list.emplace_back(drop_loss[m], m);
        if (!sel_list.empty()) {
            nth_element(sel_list.begin(), sel_list.begin() + min((int)sel_list.size(), K_SEL2), sel_list.end(), [](auto& A, auto& B){return A.first < B.first;});
        }
        int take_sel2 = min((int)sel_list.size(), K_SEL2);
        vector<int> sel_cands2; sel_cands2.reserve(take_sel2);
        for (int i = 0; i < take_sel2; ++i) sel_cands2.push_back(sel_list[i].second);
        vector<pair<int,int>> unsel_list; unsel_list.reserve(M - s.num_selected);
        for (int m = 0; m < M; ++m) if (!s.selected[m]) unsel_list.emplace_back(add_gain[m], m);
        if (!unsel_list.empty()) {
            nth_element(unsel_list.begin(), unsel_list.begin() + min((int)unsel_list.size(), K_UNSEL2), unsel_list.end(), [](auto& A, auto& B){return A.first > B.first;});
        }
        int take_unsel2 = min((int)unsel_list.size(), K_UNSEL2);
        vector<int> unsel_cands2; unsel_cands2.reserve(take_unsel2);
        for (int i = 0; i < take_unsel2; ++i) unsel_cands2.push_back(unsel_list[i].second);

        int best_a1=-1, best_a2=-1, best_b2=-1; int best_delta2 = 0;
        for (int bi : unsel_cands2) {
            if (elapsed_ms()>=time_limit_ms) break;
            const auto& Bc = customers_of_supplier[bi];
            for (int i = 0; i < (int)sel_cands2.size(); ++i) {
                int a1 = sel_cands2[i];
                for (int j = i+1; j < (int)sel_cands2.size(); ++j) {
                    int a2 = sel_cands2[j];
                    // Compute exact coverage change on union of affected customers
                    int uncovered_loss = 0, new_cover = 0;
                    // mark presence quickly by checking cov deltas
                    // Iterate over customers of a1, a2, and b
                    // Use a small visited mark to avoid double counting
                    static thread_local vector<int> touched; touched.clear();
                    static thread_local vector<char> seen; if ((int)seen.size() < N) seen.assign(N, 0);
                    auto touch = [&](const vector<int>& arr){ for (int n : arr) if (!seen[n]) { seen[n] = 1; touched.push_back(n); } };
                    touch(customers_of_supplier[a1]);
                    touch(customers_of_supplier[a2]);
                    touch(Bc);
                    for (int n : touched) {
                        int before = cov[n];
                        int after = before - (C[n][a1] ? 1 : 0) - (C[n][a2] ? 1 : 0) + (C[n][bi] ? 1 : 0);
                        if (before > 0 && after == 0) ++uncovered_loss;
                        if (before == 0 && after > 0) ++new_cover;
                        seen[n] = 0; // reset mark for reuse
                    }
                    int new_total = s.covered - uncovered_loss + new_cover;
                    if (new_total < target_required) continue; // infeasible
                    int gain = add_gain[bi];
                    // approximate loss by uncovered_loss to align with feasibility
                    int loss = uncovered_loss; // conservative
                    int delta = gain - loss - 1; // -1 for supplier count reduction
                    if (delta > best_delta2) { best_delta2 = delta; best_a1 = a1; best_a2 = a2; best_b2 = bi; }
                    if (elapsed_ms()>=time_limit_ms) break;
                }
                if (elapsed_ms()>=time_limit_ms) break;
            }
            if (elapsed_ms()>=time_limit_ms) break;
        }
        if (best_a1 != -1 && elapsed_ms() < time_limit_ms) {
            int a1 = best_a1, a2 = best_a2, b = best_b2;
            // Apply: remove a1,a2; add b
            s.selected[a1] = false; s.selected[a2] = false; s.selected[b] = true;
            s.num_selected -= 1; // net -1
            // Update cov and covered exactly
            static thread_local vector<int> touched2; touched2.clear();
            static thread_local vector<char> seen2; if ((int)seen2.size() < N) seen2.assign(N, 0);
            auto touch2 = [&](const vector<int>& arr){ for (int n : arr) if (!seen2[n]) { seen2[n] = 1; touched2.push_back(n); } };
            touch2(customers_of_supplier[a1]);
            touch2(customers_of_supplier[a2]);
            touch2(customers_of_supplier[b]);
            for (int n : touched2) {
                int before = cov[n];
                int after = before - (C[n][a1] ? 1 : 0) - (C[n][a2] ? 1 : 0) + (C[n][b] ? 1 : 0);
                if (before > 0 && after == 0) --s.covered;
                if (before == 0 && after > 0) ++s.covered;
                cov[n] = after < 0 ? 0 : after;
            }
            // Recompute local caches for a1,a2,b
            drop_loss[a1] = 0; add_gain[a1] = 0; for (int n : customers_of_supplier[a1]) if (cov[n] == 1) ++drop_loss[a1];
            drop_loss[a2] = 0; add_gain[a2] = 0; for (int n : customers_of_supplier[a2]) if (cov[n] == 1) ++drop_loss[a2];
            add_gain[b] = 0; drop_loss[b] = 0; for (int n : customers_of_supplier[b]) if (cov[n] == 0) ++add_gain[b];
            improved = true;
            if (stats) ++stats->n2_det_ops;
        }
        } // G_ENABLE_N2
    }
    s.fitness = (s.covered >= target_required) ? 1.0 / (1.0 + s.num_selected) : (double)s.covered / N;
    return true;
}

// Shake: drop r suppliers (biased towards high drop_loss) to diversify
static int shake_drop_random(Solution& s, int r, std::mt19937& rng, std::vector<int>* dropped_ms) {
    if (r <= 0) return 0;
    // Build coverage counters
    vector<int> cov(N, 0);
    for (int m = 0; m < M; ++m) if (s.selected[m]) for (int n : customers_of_supplier[m]) ++cov[n];
    // Compute drop_loss for selected suppliers
    vector<int> drop_loss(M, 0);
    vector<int> sel_list; sel_list.reserve(s.num_selected);
    for (int m = 0; m < M; ++m) if (s.selected[m]) sel_list.push_back(m);
    if (sel_list.empty()) return 0;
    for (int m : sel_list) {
        int loss = 0;
        for (int n : customers_of_supplier[m]) if (cov[n] == 1) ++loss;
        drop_loss[m] = loss;
    }
    // Sort by drop_loss descending (more critical first)
    sort(sel_list.begin(), sel_list.end(), [&](int a, int b){ return drop_loss[a] > drop_loss[b];});
    int to_drop = min(r, (int)sel_list.size());
    // Prefer from top window but randomize within it to avoid determinism
    int window = min((int)sel_list.size(), max(10, to_drop * 5));
    uniform_int_distribution<int> pick(0, window - 1);
    int dropped = 0;
    for (int k = 0; k < to_drop; ++k) {
        int idx = (int)sel_list.size() <= window ? k : pick(rng);
        if (idx >= (int)sel_list.size()) idx = (int)sel_list.size() - 1;
        int m = sel_list[idx];
        if (!s.selected[m]) continue;
        s.selected[m] = false;
        --s.num_selected;
        ++dropped;
        if (dropped_ms) dropped_ms->push_back(m);
        // Update cov and drop_loss incrementally for remaining selected
        for (int n : customers_of_supplier[m]) {
            int before = cov[n];
            if (--cov[n] < 0) cov[n] = 0;
            if (before == 2) { // n becomes critical for neighbors
                for (int q : suppliers_of_customer[n]) if (s.selected[q]) ++drop_loss[q];
            }
        }
    }
    // s.covered will be recomputed by repair
    return dropped;
}

// Coverage-boosting 1-1 swap: increase coverage (size-neutral), unlocking later pruning
static void coverage_boost_swap(Solution& s, int time_limit_ms, RunStats* stats) {
    if (!G_ENABLE_N1_SWAP) { return; }
    const int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));

    // Build coverage counters
    vector<int> cov(N, 0);
    for (int m = 0; m < M; ++m) if (s.selected[m]) for (int n : customers_of_supplier[m]) ++cov[n];
    vector<char> need(N, 0); // customers with cov<=1 are leverage points
    for (int n = 0; n < N; ++n) if (cov[n] <= 1) need[n] = 1;

    // Precompute add_gain and drop_loss
    vector<int> add_gain(M, 0), drop_loss(M, 0);
    for (int m = 0; m < M; ++m) {
        if (s.selected[m]) {
            for (int n : customers_of_supplier[m]) if (cov[n] == 1) ++drop_loss[m];
        } else {
            for (int n : customers_of_supplier[m]) if (cov[n] == 0) ++add_gain[m];
        }
    }

    // Build candidate lists
    const int K_SEL = 64;
    const int K_UNSEL = G_RL_IMP;
    vector<int> sel_cands; sel_cands.reserve(min(K_SEL, s.num_selected));
    vector<pair<int,int>> sel_pairs; sel_pairs.reserve(s.num_selected);
    for (int m = 0; m < M; ++m) if (s.selected[m]) sel_pairs.emplace_back(drop_loss[m], m);
    nth_element(sel_pairs.begin(), sel_pairs.begin() + min((int)sel_pairs.size(), K_SEL), sel_pairs.end(), [](auto& A, auto& B){return A.first > B.first;});
    int take_sel = min((int)sel_pairs.size(), K_SEL);
    for (int i = 0; i < take_sel; ++i) sel_cands.push_back(sel_pairs[i].second);

    vector<pair<int,int>> unsel_pairs; unsel_pairs.reserve(M - s.num_selected);
    for (int m = 0; m < M; ++m) if (!s.selected[m]) {
        // relevance: must cover at least one customer with cov<=1 to have a chance to improve
        bool relevant = false;
        for (int n : customers_of_supplier[m]) if (need[n]) { relevant = true; break; }
        if (!relevant) continue;
        unsel_pairs.emplace_back(add_gain[m], m);
    }
    if (!unsel_pairs.empty()) {
        nth_element(unsel_pairs.begin(), unsel_pairs.begin() + min((int)unsel_pairs.size(), K_UNSEL), unsel_pairs.end(), [](auto& A, auto& B){return A.first > B.first;});
    }
    int take_unsel = min((int)unsel_pairs.size(), K_UNSEL);
    vector<int> unsel_cands; unsel_cands.reserve(take_unsel);
    for (int i = 0; i < take_unsel; ++i) unsel_cands.push_back(unsel_pairs[i].second);

    auto t0 = chrono::steady_clock::now();
    auto elapsed_ms = [&](){ return (int)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t0).count(); };

    bool improved = true;
    while (improved && elapsed_ms() < time_limit_ms) {
        improved = false;
        int best_a = -1, best_b = -1, best_delta = 0;
        // Search only among candidate lists
        for (int a : sel_cands) {
            int loss = drop_loss[a]; if (loss < 0) loss = 0;
            for (int b : unsel_cands) {
                int gain = add_gain[b];
                int delta = gain - loss;
                if (delta > best_delta) { best_delta = delta; best_a = a; best_b = b; }
            }
            if (elapsed_ms() >= time_limit_ms) break;
        }
        if (best_delta > 0 && best_a != -1) {
            int a = best_a, b = best_b;
            // Apply swap
            s.selected[a] = false; s.selected[b] = true;
            // Update cov, gains/losses, covered
            for (int n : customers_of_supplier[a]) {
                int before = cov[n];
                if (--cov[n] == 0) --s.covered; // n became uncovered
                if (before == 2) {
                    for (int q : suppliers_of_customer[n]) if (s.selected[q]) ++drop_loss[q];
                }
                if (before == 1) {
                    for (int q : suppliers_of_customer[n]) if (!s.selected[q] && add_gain[q] > 0) --add_gain[q];
                }
            }
            for (int n : customers_of_supplier[b]) {
                int before = cov[n];
                if (cov[n]++ == 0) ++s.covered;
                if (before == 0) {
                    for (int q : suppliers_of_customer[n]) if (!s.selected[q] && add_gain[q] > 0) --add_gain[q];
                } else if (before == 1) {
                    for (int q : suppliers_of_customer[n]) if (s.selected[q] && drop_loss[q] > 0) --drop_loss[q];
                }
            }
            // refresh a and b caches
            drop_loss[a] = 0; add_gain[a] = 0; for (int n : customers_of_supplier[a]) if (cov[n] == 1) ++drop_loss[a];
            add_gain[b] = 0; drop_loss[b] = 0; for (int n : customers_of_supplier[b]) if (cov[n] == 0) ++add_gain[b];
            improved = true;
            if (stats) ++stats->n1_swap_ops;
        }
    }
    s.covered = 0; for (int n = 0; n < N; ++n) if (cov[n] > 0) ++s.covered;
}

// Forward greedy: start from empty, add suppliers with high uniqueness-weighted gain
static Solution forward_greedy_seed(int seed, const chrono::steady_clock::time_point* deadline = nullptr) {
    mt19937 rng(seed);
    Solution s; s.selected.assign(M, false);
    vector<char> covered(N, false);
    int covered_cnt = 0;
    const int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));

    // Helper: uniqueness-weighted marginal gain (sum 1/deg for uncovered customers)
    auto weighted_gain = [&](int m)->double{
        double g = 0.0;
        for (int n : customers_of_supplier[m]) if (!covered[n]) g += 1.0 / max(1, degree_customer[n]);
        return g;
    };

    // Pure deterministic greedy - no randomization
    while (covered_cnt < target_required) {
        if (deadline && chrono::steady_clock::now() >= *deadline) break;
        // Find best supplier by weighted gain
        int best_m = -1;
        double best_gain = 0.0;
        
        for (int m = 0; m < M; ++m) if (!s.selected[m]) {
            if (deadline && (m % 128) == 0 && chrono::steady_clock::now() >= *deadline) break;
            double g = weighted_gain(m);
            if (g > best_gain) {
                best_gain = g;
                best_m = m;
            }
        }
        
        if (best_m == -1) break; // no improvement possible
        
        s.selected[best_m] = true; s.num_selected++;
        for (int n : customers_of_supplier[best_m]) if (!covered[n]) { covered[n] = 1; ++covered_cnt; }
    }
    
    s.covered = covered_cnt;
    // Prune redundancies aggressively
    fast_prune_to_minimal(s);
    // Tighten a bit more with very small LS budget
    local_search_improve(s, 20);
    s.evaluate();
    return s;
}

// Forward declaration for batch helper
pair<int,int> pcsp_run(const string& input_file, int max_time_seconds, bool quiet, double coverage_pct = 90.0);

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <instance.txt> <runtime_seconds> [coverage_percent]" << endl;
        cerr << "Example: " << argv[0] << " scp41.txt 60 90" << endl;
        cerr << "Default coverage: 90%" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    int time_s = max(1, atoi(argv[2]));
    double coverage_pct = 90.0;  // Default 90%
    
    if (argc >= 4) {
        coverage_pct = atof(argv[3]);
        if (coverage_pct < 50.0 || coverage_pct > 100.0) {
            cerr << "Error: Coverage percentage must be between 50 and 100" << endl;
            return 1;
        }
    }
    
    G_QUIET = true; // Default to quiet mode for clean output
    auto [sz, acc] = pcsp_run(input_file, time_s, true, coverage_pct);
    (void)sz; (void)acc; // details printed inside
    return 0;
}

// Helper to run a single solve and return best size and accepts. Returns pair(best_size, accepts).
pair<int,int> pcsp_run(const string& input_file, int max_time_seconds, bool quiet, double coverage_pct) {
    G_QUIET = quiet;
    TARGET_COVERAGE = coverage_pct / 100.0;  // Convert percentage to decimal
    // Disable CSV logging when quiet
    // Run the existing main body but capture outputs
    const int MAX_TIME_SECONDS = max_time_seconds;
    auto start_time = chrono::steady_clock::now();
    auto deadline = start_time + chrono::seconds(MAX_TIME_SECONDS);
    read_input(input_file);
    vector<Solution> population; Solution best_solution;
    // No SSF precomputation needed.
    // Compute instance statistics for optional AUTO param mode and logging
    long long total_deg = 0;
    for (int n = 0; n < N; ++n) total_deg += degree_customer[n];
    long long total_cov = 0;
    for (int m = 0; m < M; ++m) total_cov += (long long)customers_of_supplier[m].size();
    double avg_suppliers_per_customer = N > 0 ? (double)total_deg / (double)N : 0.0; // k_cust
    double avg_customers_per_supplier = M > 0 ? (double)total_cov / (double)M : 0.0; // k_supp
    double density = (N > 0 && M > 0) ? (double)total_deg / (double)(N * (long long)M) : 0.0;
    double n_over_m = (M > 0) ? (double)N / (double)M : 0.0;
    // Environment overrides for Taguchi/meta-optimization
    // PCSP_SHAKE_S controls r_max = num_selected / S (default S=15)
    // PCSP_LS_MS controls base local search time per iter (default 240)
    // PCSP_N2_MS controls time for n2_consolidate_pass calls (default 300/200 previously)
    int ENV_SHAKE_S = 15;
    int ENV_LS_MS   = 240;
    int ENV_N2_MS   = 300;
    int ENV_RESTART_STAG_ITERS = 800;
    int ENV_RESTART_MAX = 2;
    int ENV_RL_CONST = 8;
    int ENV_RL_IMP   = 128;
    const char* ev_shake = std::getenv("PCSP_SHAKE_S");
    const char* ev_ls    = std::getenv("PCSP_LS_MS");
    const char* ev_n2    = std::getenv("PCSP_N2_MS");
    const char* ev_rst_it = std::getenv("PCSP_RESTART_STAG_ITERS");
    const char* ev_rst_mx = std::getenv("PCSP_RESTART_MAX");
    const char* ev_rl_c  = std::getenv("PCSP_RL_CONST");
    const char* ev_rl_i  = std::getenv("PCSP_RL_IMP");
    if (ev_shake) { int t = atoi(ev_shake); if (t > 0) ENV_SHAKE_S = t; }
    if (ev_ls)    { int t = atoi(ev_ls);    if (t > 0) ENV_LS_MS   = t; }
    if (ev_n2)    { int t = atoi(ev_n2);    if (t > 0) ENV_N2_MS   = t; }
    if (ev_rst_it) { int t = atoi(ev_rst_it); if (t >= 50 && t <= 500000) ENV_RESTART_STAG_ITERS = t; }
    if (ev_rst_mx) { int t = atoi(ev_rst_mx); if (t >= 0 && t <= 1000) ENV_RESTART_MAX = t; }
    if (ev_rl_c)  { int t = atoi(ev_rl_c);  if (t >= 1 && t <= 1000) ENV_RL_CONST = t; }
    if (ev_rl_i)  { int t = atoi(ev_rl_i);  if (t >= 1 && t <= 2048) ENV_RL_IMP   = t; }
    G_RL_CONST = ENV_RL_CONST;
    G_RL_IMP   = ENV_RL_IMP;
    // Ablation toggles: reset defaults then apply env overrides
    G_ENABLE_N1_DROP = true; G_ENABLE_N1_SWAP = true; G_ENABLE_N2 = true;
    if (std::getenv("PCSP_DISABLE_DROP")) G_ENABLE_N1_DROP = false;
    if (std::getenv("PCSP_DISABLE_SWAP")) G_ENABLE_N1_SWAP = false;
    if (std::getenv("PCSP_DISABLE_N2"))   G_ENABLE_N2      = false;
    // Optional AUTO mode selecting parameters from instance stats unless explicitly overridden
    bool have_env_shake = (ev_shake != nullptr);
    bool have_env_ls    = (ev_ls    != nullptr);
    bool have_env_n2    = (ev_n2    != nullptr);
    const char* pmode = std::getenv("PCSP_PARAM_MODE");
    // Explicit modes: 'pop' uses population (instance) statistics; 'fixed' uses static defaults unless env overrides.
    bool POP_MODE   = (pmode && strcmp(pmode, "POP") == 0);
    bool FIXED_MODE = (pmode && strcmp(pmode, "FIXED") == 0);
    (void)FIXED_MODE; // currently implied by not POP_MODE
    if (POP_MODE) {
        // Category-based tuning using density, redundancy (k_cust), and size (M)
        double k_cust = avg_suppliers_per_customer;
        double dens   = density;
        auto clamp_int = [](int v, int lo, int hi){ return std::max(lo, std::min(hi, v)); };
        auto clamp_d   = [](double v, double lo, double hi){ return std::max(lo, std::min(hi, v)); };

        // Updated density categories to match Methodology.md
        bool very_sparse = (dens < 0.01);   // <1%
        bool sparseCat   = (!very_sparse && dens < 0.03);  // <3%
        bool denseCat    = (dens >= 0.08);  // ≥8%
        bool mediumCat   = (!very_sparse && !sparseCat && !denseCat);
        bool high_red    = (k_cust >= 6.0); // ≥6 suppliers per customer
        bool sz_small    = (M <= 200);      // ≤200 suppliers
        bool sz_large    = (M > 1000);      // >1000 suppliers

        // shake_s: 10-22 range (stronger shaking for dense/high-redundancy)
        if (!have_env_shake) {
            int S = 15; // default
            if (denseCat || high_red) S = 10;  // stronger shaking for dense/high-redundancy
            else if (very_sparse)     S = 20;  // weaker shaking for very sparse
            else if (sparseCat)       S = 18;  // weaker shaking for sparse
            else /*medium*/           S = 15;
            ENV_SHAKE_S = clamp_int(S, 10, 22);
        }
        // ls_ms: 280-480 range (longer search for sparse instances)
        if (!have_env_ls) {
            int LS = 240; // default
            if (very_sparse)      LS = 450;  // much longer for very sparse
            else if (sparseCat)   LS = 360;  // longer for sparse
            else if (mediumCat)   LS = 300;  // slightly longer for medium
            else /*dense*/        LS = 280;  // shorter for dense
            if (sz_large) LS += 40; // size adjustment for large instances
            ENV_LS_MS = clamp_int(LS, 280, 480);
        }
        // n2_ms: 260-450 range (more consolidation for dense instances)
        if (!have_env_n2) {
            int N2 = 300; // default
            if (denseCat)         N2 = 420; // more consolidation for dense
            else if (very_sparse) N2 = 400; // help very sparse instances
            else if (sparseCat)   N2 = 350; // moderate increase for sparse
            else /*medium*/       N2 = 320; // slight increase for medium
            ENV_N2_MS = clamp_int(N2, 260, 450);
        }
    }
    // Seeding loop (same as main)
    // Seed-phase event logging: ensure we write and flush progress early so that
    // hard-killed runs still leave recoverable BEST/LAST_IMP information.
    auto append_seed_event = [&](const std::string& stage, long long elapsed_ms, const std::string& event, int new_best_size, int covered, const std::string& note){
        try {
            const char* env_events = std::getenv("PCSP_EVENTS_PATH");
            if (!env_events || !*env_events) return;
            const char* events_path = env_events;
            bool write_header = !fs::exists(events_path) || fs::file_size(events_path) == 0;
            std::ofstream ev(events_path, std::ios::app);
            if (!ev) return;
            if (write_header) {
                ev << "ts_epoch,instance,stage,elapsed_ms,event,new_best_size,covered,coverage_pct,note\n";
            }
            std::string inst_name = input_file;
            try { inst_name = fs::path(input_file).filename().string(); } catch (...) {}
            auto ts_epoch = (long long)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            double coverage_pct = COVERABLE_N > 0 ? (100.0 * (double)covered / (double)COVERABLE_N) : 0.0;
            ev.setf(std::ios::fixed); ev << std::setprecision(1);
            ev << ts_epoch << "," << inst_name << "," << stage << "," << elapsed_ms << "," << event << ",";
            if (new_best_size >= 0) ev << new_best_size;
            ev << "," << covered << "," << coverage_pct << "," << note << "\n";
            ev.unsetf(std::ios::floatfield);
            ev.flush();
        } catch (...) { /* swallow */ }
    };

    // Create the events file/header immediately (even before we have a feasible seed)
    {
        long long elapsed_ms = (long long)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
        append_seed_event("Init", elapsed_ms, "start", -1, 0, "");
    }

    int REQUIRED_FEASIBLE_SEEDS = 25;
    const char* ev_fs = std::getenv("PCSP_FEASIBLE_SEEDS");
    if (ev_fs) {
        int t = atoi(ev_fs);
        if (t >= 5 && t <= 500) REQUIRED_FEASIBLE_SEEDS = t;
    } else if (POP_MODE) {
        // Category-based feasible seeds
        double k_cust = avg_suppliers_per_customer; (void)k_cust;
        double dens   = density;
        auto clamp_int = [](int v, int lo, int hi){ return std::max(lo, std::min(hi, v)); };
        bool very_sparse = (dens < 0.010);
        bool denseCat    = (dens >= 0.080);
        int FS = 25;
        if (denseCat)         FS = 20;
        else if (very_sparse) FS = 35; // bump seeds for very-sparse
        REQUIRED_FEASIBLE_SEEDS = clamp_int(FS, 5, 500);
    }
    int seed_counter = 0;
    int greedy_improves = 0;
    while ((int)population.size() < REQUIRED_FEASIBLE_SEEDS) {
        if (chrono::steady_clock::now() >= deadline) break;
        Solution s;
        // Seed using forward greedy (then repair/prune/LS as needed)
        s = forward_greedy_seed(seed_counter++, &deadline);
        if (!s.is_feasible()) { fast_repair_to_target(s); fast_prune_to_minimal(s); local_search_improve(s, 20); }
        if (s.is_feasible()) {
            bool improved = population.empty() || s.fitness > best_solution.fitness;
            if (improved) {
                best_solution = s; ++greedy_improves;
                long long elapsed_ms = (long long)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
                append_seed_event("Greedy", elapsed_ms, "improvement", best_solution.num_selected, best_solution.covered, "Seed best after seeding");
            }
            population.push_back(s);
        }
    }
    // If time budget is extremely small and we didn't manage to seed any feasible solution,
    // attempt a time-bounded greedy seed as a last resort.
    if (population.empty()) {
        Solution s = forward_greedy_seed(seed_counter++, &deadline);
        if (!s.is_feasible()) { fast_repair_to_target(s); }
        fast_prune_to_minimal(s);
        if (s.is_feasible()) {
            best_solution = s;
            population.push_back(s);
            long long elapsed_ms = (long long)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
            append_seed_event("Greedy", elapsed_ms, "improvement", best_solution.num_selected, best_solution.covered, "Seed best after seeding");
        }
    }
    if (population.size() > 0) {
        int per_solution_ms = max(20, 2000 / max(1, (int)population.size()));
        local_search_improve(best_solution, per_solution_ms);
    }
    // Ensure we do not start VNS from an infeasible incumbent
    if (!best_solution.is_feasible()) {
        fast_repair_to_target(best_solution);
        fast_prune_to_minimal(best_solution);
        local_search_improve(best_solution, 80);
        best_solution.evaluate();
    }
    // Replace verbose console banners with summarized output at the end
    // VNS
    unsigned int rng_seed = static_cast<unsigned int>(chrono::steady_clock::now().time_since_epoch().count());
    const char* ev_seed = std::getenv("PCSP_SEED");
    if (ev_seed) {
        long long t = atoll(ev_seed);
        if (t > 0) rng_seed = static_cast<unsigned int>(t);
    }
    mt19937 rng(rng_seed);
    G_SOLVER_RNG = &rng;
    int r = 1; int r_max = max(3, best_solution.num_selected / ENV_SHAKE_S); int iter = 0; RunStats stats; int last_accept_iter=-1; int intensify_iters=0;
    // Simplified v2 behavior (no env toggles): Reactive OFF, MiniLS OFF, Endgame ON with fixed params
    const int  V2_TIME_PRESSURE_PCT = 90;  // percent 0..100 to trigger endgame escalation
    const int  V2_STAG_ITERS        = 80;  // stagnation iterations threshold before endgame escalation
    const int  V2_ENDGAME_PULSES    = 1;   // number of escalation pulses allowed
    int no_improve_iters = 0;              // non-improving iteration counter
    int endgame_pulses_used = 0;           // pulses used so far
    // Event logging helper: append per-improvement and pulse events
    auto append_stage_event = [&](const std::string& stage, long long elapsed_ms, const std::string& event, int new_best_size, int covered, const std::string& note){
        try {
            const char* env_events = std::getenv("PCSP_EVENTS_PATH");
            const char* events_path = env_events ? env_events : "stage_events.csv";
            bool write_header = !fs::exists(events_path) || fs::file_size(events_path) == 0;
            std::ofstream ev(events_path, std::ios::app);
            if (!ev) return;
            if (write_header) {
                ev << "ts_epoch,instance,stage,elapsed_ms,event,new_best_size,covered,coverage_pct,note\n";
            }
            std::string inst_name = input_file;
            try { inst_name = fs::path(input_file).filename().string(); } catch (...) {}
            auto ts_epoch = (long long)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            double coverage_pct = COVERABLE_N > 0 ? (100.0 * (double)covered / (double)COVERABLE_N) : 0.0;
            ev.setf(std::ios::fixed); ev << std::setprecision(1);
            ev << ts_epoch << "," << inst_name << "," << stage << "," << elapsed_ms << "," << event << ",";
            if (new_best_size >= 0) ev << new_best_size; ev << "," << covered << "," << coverage_pct << "," << note << "\n";
            ev.unsetf(std::ios::floatfield);
            ev.flush();
        } catch (...) { /* swallow */ }
    };
    // Totals
    long long total_evals = 0; // proxy: count VNS iterations + seed eval
    long long total_improves = 0;
    struct ConsoleEvent { std::string stage; long long ms; int new_best; int covered; double cov_pct; std::string note; };
    std::vector<ConsoleEvent> improvements_console;
    long long last_improve_ms = 0;
    auto note_with_since = [&](const std::string& note, long long elapsed_ms) -> std::string {
        long long since_ms = elapsed_ms - last_improve_ms;
        std::string out = note;
        if (!out.empty()) out += ";";
        out += "since_last_improve_ms=" + std::to_string(since_ms);
        return out;
    };
    // Log Greedy seed as an improvement event
    {
        long long elapsed_ms = (long long)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
        append_stage_event("Greedy", elapsed_ms, "improvement", best_solution.num_selected, best_solution.covered, note_with_since("Seed best after seeding", elapsed_ms));
        double covp = COVERABLE_N > 0 ? (100.0 * (double)best_solution.covered / (double)COVERABLE_N) : 0.0;
        improvements_console.push_back({"Greedy", elapsed_ms, best_solution.num_selected, best_solution.covered, covp, "Seed best after seeding"});
        ++total_improves;
        ++total_evals;
        last_improve_ms = elapsed_ms;
    }
    ofstream csv; // not opened in quiet mode
    Solution base_solution = best_solution;
    int restarts_used = 0;
    while (true) {
        auto current_time = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::seconds>(current_time - start_time).count();
        if (elapsed >= MAX_TIME_SECONDS) break;
        Solution T = base_solution;
        bool use_alt_elite = false; // simplify: always start from best in quiet solve
        (void)use_alt_elite;
        RunStats before = stats;
        std::vector<int> dropped_ms;
        int dropped = shake_drop_random(T, r, rng, &dropped_ms); stats.shake_dropped += dropped;
        std::vector<char> avoid_add;
        if (!dropped_ms.empty()) {
            avoid_add.assign(M, 0);
            for (int m : dropped_ms) if (m >= 0 && m < M) avoid_add[m] = 1;
        }
        // R2: endgame multi-repair pulse (try several alternative repairs from the same shaken base)
        // to reduce path-dependence and help escape hard local basins.
        Solution T_shaken = T;
        auto is_better_candidate = [&](const Solution& a, const Solution& b) -> bool {
            int target_required = (int)ceil(COVERABLE_N * TARGET_COVERAGE);
            bool af = (a.covered >= target_required);
            bool bf = (b.covered >= target_required);
            if (af != bf) return af; // prefer feasible
            if (a.num_selected != b.num_selected) return a.num_selected < b.num_selected;
            return a.covered > b.covered;
        };

        int repair_tries = 1;
        {
            double frac = (double)elapsed / (double)MAX_TIME_SECONDS;
            if ((int)(frac * 100.0) >= V2_TIME_PRESSURE_PCT) repair_tries = 3;
        }

        Solution T_best = T_shaken;
        bool have_best = false;
        for (int tr = 0; tr < repair_tries; ++tr) {
            Solution cand = T_shaken;
            std::mt19937 local_rng(rng());
            std::mt19937* prev_rng = G_SOLVER_RNG;
            G_SOLVER_RNG = &local_rng;
            fast_repair_to_target(cand, &stats, avoid_add.empty() ? nullptr : &avoid_add);
            fast_prune_to_minimal(cand, &stats);
            local_search_improve(cand, 60, &stats);
            cand.evaluate();
            G_SOLVER_RNG = prev_rng;
            if (!have_best || is_better_candidate(cand, T_best)) { T_best = std::move(cand); have_best = true; }
        }
        if (have_best) T = std::move(T_best);

        int cb_ms = intensify_iters > 0 ? 160 : 100;
        int ej_ms = intensify_iters > 0 ? 70  : 40;
        int ls_ms = intensify_iters > 0 ? max(ENV_LS_MS, (int)(ENV_LS_MS * 4 / 3)) : ENV_LS_MS;
        coverage_boost_swap(T, cb_ms, &stats);
        fast_prune_to_minimal(T, &stats);
        if (iter % 10 == 0) { n2_consolidate_pass(T, rng, ENV_N2_MS, &stats); fast_prune_to_minimal(T, &stats); }
        ejection_chain2(T, rng, ej_ms, &stats);
        local_search_improve(T, ls_ms, &stats);
        if (iter % 50 == 0) { local_search_improve(T, 90, &stats); }
        if (iter % 100 == 0) { n2_consolidate_pass(T, rng, ENV_N2_MS, &stats); fast_prune_to_minimal(T, &stats); }
        n2_consolidate_pass(T, rng, ENV_N2_MS, &stats); fast_prune_to_minimal(T, &stats); local_search_improve(T, 80, &stats);
        if (T.covered < (int)ceil(COVERABLE_N * TARGET_COVERAGE)) { fast_repair_to_target(T, &stats); fast_prune_to_minimal(T, &stats); }
        T.evaluate();
        bool T_feasible = (T.covered >= (int)ceil(COVERABLE_N * TARGET_COVERAGE));
        // Count one evaluation per VNS loop cycle as a proxy
        ++total_evals;
        bool better;
        if (best_solution.is_feasible()) {
            // Normal: accept only feasible improvements
            better = T_feasible && ((T.num_selected < best_solution.num_selected) || (T.num_selected == best_solution.num_selected && T.covered > best_solution.covered));
        } else {
            // Bootstrap: prefer any feasible T; otherwise improve coverage (and tie-break by fewer suppliers)
            better = T_feasible || (T.covered > best_solution.covered) || (T.covered == best_solution.covered && T.num_selected < best_solution.num_selected);
        }
        if (better) {
            best_solution = T; ++stats.vns_accepts; r = 1; r_max = max(3, best_solution.num_selected / ENV_SHAKE_S); last_accept_iter = iter; intensify_iters = G_RL_CONST;
            no_improve_iters = 0;
            base_solution = best_solution;
            local_search_improve(best_solution, 360, &stats);
            ejection_chain2(best_solution, rng, 60, &stats);
            fast_prune_to_minimal(best_solution, &stats);
            n2_consolidate_pass(best_solution, rng, ENV_N2_MS, &stats);
            fast_prune_to_minimal(best_solution, &stats);
            local_search_improve(best_solution, 120, &stats);
            if (best_solution.covered < (int)ceil(COVERABLE_N * TARGET_COVERAGE)) { fast_repair_to_target(best_solution, &stats); fast_prune_to_minimal(best_solution, &stats); }
            best_solution.evaluate();
            // Log improvement event with stage label (Endgame if within endgame window and after pulse; otherwise VNS)
            auto now = chrono::steady_clock::now();
            long long elapsed_ms = chrono::duration_cast<chrono::milliseconds>(now - start_time).count();
            double frac = (double)chrono::duration_cast<chrono::seconds>(now - start_time).count() / (double)MAX_TIME_SECONDS;
            std::string stage_label = (endgame_pulses_used > 0 && (int)(frac * 100.0) >= V2_TIME_PRESSURE_PCT) ? "Endgame" : "VNS";
            append_stage_event(stage_label, elapsed_ms, "improvement", best_solution.num_selected, best_solution.covered, note_with_since("", elapsed_ms));
            double covp = COVERABLE_N > 0 ? (100.0 * (double)best_solution.covered / (double)COVERABLE_N) : 0.0;
            improvements_console.push_back({stage_label, elapsed_ms, best_solution.num_selected, best_solution.covered, covp, ""});
            ++total_improves;
            last_improve_ms = elapsed_ms;
            // Suppress per-iteration accept print in favor of summarized output
        } else {
            // neighborhood growth policy
            // Reactive OFF: gentler shake growth and quicker reset on stagnation
            r = min(r + ((iter % 3) == 0 ? 1 : 0), r_max);
            if (last_accept_iter >= 0 && iter - last_accept_iter > 400) r = 1;
            // Track stagnation
            ++no_improve_iters;
            if (ENV_RESTART_MAX > 0 && restarts_used < ENV_RESTART_MAX && no_improve_iters >= ENV_RESTART_STAG_ITERS) {
                // Hard time-guard: do not start an expensive restart too close to the end of the budget
                {
                    auto now_chk = chrono::steady_clock::now();
                    auto elapsed_chk = chrono::duration_cast<chrono::seconds>(now_chk - start_time).count();
                    if (elapsed_chk >= MAX_TIME_SECONDS) break;
                    if (MAX_TIME_SECONDS - elapsed_chk < 5) {
                        // Not enough time left to reseed safely
                        no_improve_iters = 0;
                        goto skip_restart_v3;
                    }
                }
                Solution s;
                // Restart from a fresh forward greedy seed
                s = forward_greedy_seed(seed_counter++);
                if (!s.is_feasible()) { fast_repair_to_target(s); }
                fast_prune_to_minimal(s);
                local_search_improve(s, 80);
                if (!s.is_feasible()) { fast_repair_to_target(s); fast_prune_to_minimal(s); }
                base_solution = s;
                r = 1;
                r_max = max(3, base_solution.num_selected / ENV_SHAKE_S);
                last_accept_iter = iter;
                intensify_iters = 0;
                no_improve_iters = 0;
                ++restarts_used;
                long long elapsed_ms = (long long)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
                append_stage_event("VNS", elapsed_ms, "restart", base_solution.num_selected, base_solution.covered, note_with_since("", elapsed_ms));
                // Hard time-guard: if restart work pushed us over the budget, stop now.
                {
                    auto now_chk = chrono::steady_clock::now();
                    auto elapsed_chk = chrono::duration_cast<chrono::seconds>(now_chk - start_time).count();
                    if (elapsed_chk >= MAX_TIME_SECONDS) break;
                }
            }
            skip_restart_v3:
            // Endgame escalation (fixed params): near time limit and stagnating -> pulse to max neighborhood
            if (V2_ENDGAME_PULSES > 0) {
                double frac = (double)elapsed / (double)MAX_TIME_SECONDS;
                if ((int)(frac * 100.0) >= V2_TIME_PRESSURE_PCT && no_improve_iters >= V2_STAG_ITERS && endgame_pulses_used < V2_ENDGAME_PULSES) {
                    r = r_max;
                    ++endgame_pulses_used;
                    no_improve_iters = 0;
                    long long elapsed_ms = (long long)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
                    append_stage_event("Endgame", elapsed_ms, "pulse", -1, best_solution.covered, note_with_since("Escalate r to r_max", elapsed_ms));
                }
            }
        }
        ++iter; ++stats.vns_iters; if (intensify_iters > 0) --intensify_iters;
    }
    ++iter; ++stats.vns_iters; if (intensify_iters > 0) --intensify_iters;

    best_solution.evaluate();
    // Compute elapsed time regardless of verbosity
    auto end_time = chrono::steady_clock::now();
    auto elapsed_s = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();

    // Summarized console output per the agreed format
    std::string inst_name = input_file; try { inst_name = fs::path(input_file).filename().string(); } catch (...) {}
    // Summary table
    cout.setf(std::ios::fixed); cout << setprecision(1);
    cout << "# Summary (per run)\n";
    cout << "| Instance | Actual runtime (s) | Best result | Total evals | Total improves |\n";
    cout << "|---|---:|---:|---:|---:|\n";
    cout << "| " << inst_name << " | " << (double)elapsed_s << " | " << best_solution.num_selected << " | "
         << total_evals << " | " << total_improves << " |\n\n";
    // Improvements table
    cout << "# Improvements (each improvement as a separate line)\n";
    cout << "| Stage | Elapsed (ms) | New Best | Covered | Coverage (%) | Note |\n";
    cout << "|---|---:|---:|---:|---:|---|\n";
    for (const auto& ev : improvements_console) {
        cout << "| " << ev.stage << " | " << ev.ms << " | " << ev.new_best << " | "
             << ev.covered << " | " << setprecision(1) << ev.cov_pct << " | " << ev.note << " |\n";
    }
    cout << "\nTotal Improvements: " << total_improves << "\n\n";
    // Suppliers list
    cout << "Suppliers (1-based): ";
    bool first = true;
    for (int m = 0; m < M; ++m) {
        if (best_solution.selected[m]) {
            if (!first) {
                cout << " ";
            }
            cout << (m + 1);
            first = false;
        }
    }
    cout << "\n";
    cout.unsetf(std::ios::floatfield);

    // Append a compact CSV summary for batch analysis
    try {
        const char* env_csv = std::getenv("PCSP_CSV_PATH");
        const char* csv_path = env_csv ? env_csv : "vns_progress_60s.csv";
        bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
        std::ofstream out(csv_path, std::ios::app);
        if (out) {
            // Determine if we include optional columns (controlled by env flags)
            bool include_feasible = (std::getenv("PCSP_FEASIBLE_SEEDS") != nullptr);
            bool include_stats    = (std::getenv("PCSP_LOG_INSTANCE_STATS") != nullptr);
            if (write_header) {
                // Include coverage fields for tie-breaking (covered and coverable_n)
                out << "ts_epoch,instance,run_seconds,run_seconds_actual,shake_s,ls_ms,n2_ms,best_size,covered,coverable_n,accepts,total_evals,total_improves";
                if (include_feasible) out << ",feasible_seeds";
                if (include_stats)    out << ",avg_suppliers_per_customer,avg_customers_per_supplier,density,n_over_m";
                out << ",selected_set\n";
            }
            std::string inst_name = input_file;
            try { inst_name = fs::path(input_file).filename().string(); } catch (...) {}
            auto ts_epoch = (long long)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            double actual_seconds = (double)elapsed_s;
            out << ts_epoch << ","
                << inst_name << ","
                << MAX_TIME_SECONDS << ","
                << actual_seconds << ","
                << ENV_SHAKE_S << "," << ENV_LS_MS << "," << ENV_N2_MS << ","
                << best_solution.num_selected << ","
                << best_solution.covered << "," << COVERABLE_N << ","
                << stats.vns_accepts << ","
                << total_evals << "," << total_improves;
            if (include_feasible) out << "," << REQUIRED_FEASIBLE_SEEDS;
            if (include_stats) {
                out.setf(std::ios::fixed); out << std::setprecision(4);
                out << "," << avg_suppliers_per_customer
                    << "," << avg_customers_per_supplier
                    << "," << density
                    << "," << n_over_m;
                out.unsetf(std::ios::floatfield);
            }
            out << ",\"";
            bool first_sel = true;
            for (int m = 0; m < M; ++m) {
                if (best_solution.selected[m]) {
                    if (!first_sel) out << " ";
                    out << (m + 1);
                    first_sel = false;
                }
            }
            out << "\"\n";
        }
    } catch (...) {
        // Best-effort logging; ignore any errors writing CSV
    }

    return {best_solution.num_selected, (int)stats.vns_accepts};
}

// Depth-2 ejection chain: drop a low-loss supplier, repair, then prune; accept if net -1 and feasible
static bool ejection_chain2(Solution& s, std::mt19937& rng, int time_limit_ms, RunStats* stats) {
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_ms = [&](){ return (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count(); };
    const int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));

    // Build coverage counts
    std::vector<int> cov(N, 0);
    for (int m = 0; m < M; ++m) if (s.selected[m]) for (int n : customers_of_supplier[m]) ++cov[n];

    // Rank selected by loss (how many cov==1 customers they uniquely cover)
    std::vector<std::pair<int,int>> sel_by_loss; sel_by_loss.reserve(s.num_selected);
    for (int m = 0; m < M; ++m) if (s.selected[m]) {
        int loss = 0; for (int n : customers_of_supplier[m]) if (cov[n] == 1) ++loss; sel_by_loss.emplace_back(loss, m);
    }
    std::sort(sel_by_loss.begin(), sel_by_loss.end());

    const int MAX_TRIES = 16;
    int tries = 0;
    for (auto [loss, a] : sel_by_loss) {
        if (elapsed_ms() >= time_limit_ms || ++tries > MAX_TRIES) break;
        Solution t = s;
        t.selected[a] = false; --t.num_selected;
        // Update covered for customers of a
        for (int n : customers_of_supplier[a]) if (--cov[n] == 0) --t.covered;

        fast_repair_to_target(t, stats, nullptr);
        fast_prune_to_minimal(t, stats);
        if (t.num_selected < s.num_selected && t.covered >= target_required) { s = std::move(t); return true; }

        // revert cov for next candidate
        for (int n : customers_of_supplier[a]) ++cov[n];
    }
    return false;
}

static void local_search_improve(Solution& s, int time_limit_ms, RunStats* stats) {
    if (G_SOLVER_RNG) {
        (void)some_function_name_placeholder(s, *G_SOLVER_RNG, time_limit_ms, stats);
        return;
    }
    std::mt19937 tmp_rng(1);
    (void)some_function_name_placeholder(s, tmp_rng, time_limit_ms, stats);
}

// Lightweight N2 (2->1) consolidation: sample candidate pairs and an add, apply if feasible net -1
static bool n2_consolidate_pass(Solution& s, std::mt19937& rng, int time_limit_ms, RunStats* stats) {
    if (!G_ENABLE_N2) return false;
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed_ms = [&](){ return (int)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count(); };
    const int target_required = static_cast<int>(ceil(COVERABLE_N * TARGET_COVERAGE));
    if (s.num_selected < 2) return false;
    // Build coverage counters
    std::vector<int> cov(N, 0);
    for (int m = 0; m < M; ++m) if (s.selected[m]) for (int n : customers_of_supplier[m]) ++cov[n];
    // Build candidate lists: selected by drop-loss, unselected by add-gain
    std::vector<int> drop_loss(M, 0), add_gain(M, 0);
    for (int m = 0; m < M; ++m) {
        if (s.selected[m]) {
            for (int n : customers_of_supplier[m]) if (cov[n] == 1) ++drop_loss[m];
        } else {
            for (int n : customers_of_supplier[m]) if (cov[n] == 0) ++add_gain[m];
        }
    }
    std::vector<std::pair<int,int>> sel_list; sel_list.reserve(s.num_selected);
    for (int m = 0; m < M; ++m) if (s.selected[m]) sel_list.emplace_back(drop_loss[m], m);
    if (sel_list.empty()) return false;
    std::nth_element(sel_list.begin(), sel_list.begin() + std::min((int)sel_list.size(), 160), sel_list.end(), [](auto& A, auto& B){return A.first < B.first;});
    int take_sel = std::min((int)sel_list.size(), 160);
    std::vector<int> sel_cands; sel_cands.reserve(take_sel);
    for (int i = 0; i < take_sel; ++i) sel_cands.push_back(sel_list[i].second);
    std::vector<std::pair<int,int>> unsel_list; unsel_list.reserve(M - s.num_selected);
    for (int m = 0; m < M; ++m) if (!s.selected[m]) unsel_list.emplace_back(add_gain[m], m);
    if (!unsel_list.empty()) std::nth_element(unsel_list.begin(), unsel_list.begin() + std::min((int)unsel_list.size(), 320), unsel_list.end(), [](auto& A, auto& B){return A.first > B.first;});
    int take_unsel = std::min((int)unsel_list.size(), 320);
    std::vector<int> unsel_cands; unsel_cands.reserve(take_unsel);
    for (int i = 0; i < take_unsel; ++i) unsel_cands.push_back(unsel_list[i].second);

    if (unsel_cands.empty() || sel_cands.size() < 2) return false;
    // Random-sampled tries
    std::uniform_int_distribution<int> ds(0, (int)sel_cands.size()-1);
    std::uniform_int_distribution<int> du(0, (int)unsel_cands.size()-1);
    int tries = 0, MAX_TRIES = 600;
    while (elapsed_ms() < time_limit_ms && tries++ < MAX_TRIES) {
        int a1 = sel_cands[ds(rng)], a2 = sel_cands[ds(rng)]; if (a1 == a2) continue;
        int b = unsel_cands[du(rng)];
        // Compute uncovered loss/new cover on the fly over the union neighborhoods (no C[][] lookups)
        static thread_local std::vector<int> touched; touched.clear();
        static thread_local std::vector<char> seen;
        static thread_local std::vector<uint8_t> mask; // bit0=a1, bit1=a2, bit2=b
        if ((int)seen.size() < N) { seen.assign(N, 0); mask.assign(N, 0); }
        auto touch = [&](const std::vector<int>& arr, uint8_t bit){
            for (int n : arr) {
                mask[n] |= bit;
                if (!seen[n]) { seen[n] = 1; touched.push_back(n); }
            }
        };
        touch(customers_of_supplier[a1], 1);
        touch(customers_of_supplier[a2], 2);
        touch(customers_of_supplier[b],  4);
        int uncovered_loss = 0, new_cover = 0;
        for (int n : touched) {
            int before = cov[n];
            int after  = before - ((mask[n] & 1) ? 1 : 0) - ((mask[n] & 2) ? 1 : 0) + ((mask[n] & 4) ? 1 : 0);
            if (before > 0 && after == 0) ++uncovered_loss;
            if (before == 0 && after > 0) ++new_cover;
            seen[n] = 0;
            mask[n] = 0;
        }
        int new_total = s.covered - uncovered_loss + new_cover;
        if (new_total < target_required) continue;
        // Apply move
        s.selected[a1] = false; s.selected[a2] = false; s.selected[b] = true; s.num_selected -= 1;
        // Apply delta on the touched neighborhood (recompute after from membership mask)
        // Note: we rebuild membership from adjacency again to avoid storing extra state.
        static thread_local std::vector<char> seen2;
        static thread_local std::vector<uint8_t> mask2;
        if ((int)seen2.size() < N) { seen2.assign(N, 0); mask2.assign(N, 0); }
        std::vector<int> touched2; touched2.reserve(touched.size());
        auto touch2 = [&](const std::vector<int>& arr, uint8_t bit){
            for (int n : arr) {
                mask2[n] |= bit;
                if (!seen2[n]) { seen2[n] = 1; touched2.push_back(n); }
            }
        };
        touch2(customers_of_supplier[a1], 1);
        touch2(customers_of_supplier[a2], 2);
        touch2(customers_of_supplier[b],  4);
        for (int n : touched2) {
            int before = cov[n];
            int after  = before - ((mask2[n] & 1) ? 1 : 0) - ((mask2[n] & 2) ? 1 : 0) + ((mask2[n] & 4) ? 1 : 0);
            if (before > 0 && after == 0) --s.covered;
            if (before == 0 && after > 0) ++s.covered;
            cov[n] = std::max(0, after);
            seen2[n] = 0;
            mask2[n] = 0;
        }
        if (stats) ++stats->n2_rand_ops;
        return true;
    }
    return false;
}
