/*
Cell Resampling (Task 2)

Implements the cell resampling algorithm from arXiv:2109.07851 (Section 2)
for eliminating negative event weights in NLO Monte Carlo samples.

Using a 2D k-d tree:
The tree-based approach discussed in arXiv:2303.15246 (Section 2.2).
The paper uses vantage-point trees; we use a k-d tree which provides
equivalent O(log N) average query time in our 2D phase space.

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <unordered_set>
#include <limits>


struct Event {
    double pt;
    double y;
    double weight;
};

//for per cell recor record
struct CellRecord {
    int size;       //number of events in this cell
    double radius;  //max distance from seed to any cell member
};

//Distance metric
/*
//Scaled distance in (pt, y) space. The rapidity scaling factor compensates
//For the different numerical ranges of pt (~0-300 GeV) and y (~-5 to 5).

Without scaling, a rapidity difference of dy=1 (physically significant particles going in noticeably different directions) would be treated the

same as dpt=1 GeV (negligible compared to the pt range). The algorithm

would then group events that differ widely in rapidity, distorting the

y distribution after weight redistribution.

The factor 100 is roughly (pt_range / y_range)^2 ~ (200/10)^2 = 400,
This makes 100 a reasonable choice
*/

static constexpr double Y_SCALE = 100.0;

static double scaled_distance(double pt1, double y1, double pt2, double y2) {
    double dp = pt1 - pt2;
    double dy = y1 - y2;
    return std::sqrt(dp * dp + Y_SCALE * dy * dy);
}

// 2D k-d tree for nearest-neighbor search

/*
Standard k-d tree partitioning the (pt, y) plane by alternating splits.
Supports nearest-neighbor queries with an exclusion set, which is needed
for incrementally growing cells
We exclude events already in the current cell so that each query returns the next-closest event.

Build: O(N log N). Single NN query: O(log N) average, O(N) worst case.
For the full resampling with S seeds and average cell size k:
Total NN queries = S * k, this gives us O(S * k * log N) average.
*/
class KDTree {
public:
    explicit KDTree(const std::vector<Event>& events) : events_(events) {
        if (events.empty()) return;
        std::vector<int> indices(events.size());
        std::iota(indices.begin(), indices.end(), 0);
        root_ = build(indices, 0, static_cast<int>(indices.size()) - 1, 0);
    }

    ~KDTree() { free_tree(root_); }

    //Finds the nearest event to (qpt, qy) whose index is NOT in `exclude`.
    //Returns the event index, or -1 if every event is excluded.
    int find_nearest(double qpt, double qy,
                     const std::unordered_set<int>& exclude) const {
        double best_dist = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        search(root_, qpt, qy, best_dist, best_idx, exclude);
        return best_idx;
    }

private:
    struct Node {
        int idx;
        int axis;               
        Node* left  = nullptr;
        Node* right = nullptr;
    };

    Node* root_ = nullptr;
    const std::vector<Event>& events_;

    //recursively building a balanced tree
    //using median splitting(kind of binary search only)
    Node* build(std::vector<int>& indices, int lo, int hi, int depth) {
        if (lo > hi) return nullptr;

        int axis = depth % 2;
        int mid  = lo + (hi - lo) / 2;
        
        std::nth_element(
            indices.begin() + lo,
            indices.begin() + mid,
            indices.begin() + hi + 1,
            [&](int a, int b) {
                return (axis == 0) ? events_[a].pt < events_[b].pt
                                   : events_[a].y  < events_[b].y;
            }
        );
        auto* node  = new Node;
        node->idx   = indices[mid];
        node->axis  = axis;
        node->left  = build(indices, lo, mid - 1, depth + 1);
        node->right = build(indices, mid + 1, hi, depth + 1);
        return node;
    }

    //Standard k-d tree NN search with branch pruning.
    //Excluded indices are skipped as candidates but the tree structure
    //(split planes) is still used for pruning — this is correct because
    //the split plane position is a property of the tree, not of whether
    //the node is a valid candidate.
    void search(Node* node, double qpt, double qy,
                double& best_dist, int& best_idx,
                const std::unordered_set<int>& exclude) const {
        if (!node) return;

        //Consider this node as a candidate unless excluded
        if (exclude.count(node->idx) == 0) {
            double d = scaled_distance(qpt, qy,
                                       events_[node->idx].pt,
                                       events_[node->idx].y);
            if (d < best_dist) {
                best_dist = d;
                best_idx  = node->idx;
            }
        }

        //decides which side of the splitting plane does the query fall on
        double diff = (node->axis == 0)
            ? (qpt - events_[node->idx].pt)
            : (qy  - events_[node->idx].y);

        Node* near = (diff <= 0) ? node->left  : node->right;
        Node* far  = (diff <= 0) ? node->right : node->left;

        search(near, qpt, qy, best_dist, best_idx, exclude);

        //Minimum possible scaled distance to any point on the far side.
        //For axis 0, pt split, then just |dpt|, since dy can be zero.
        //For axis 1 (y split), then  sqrt(Y_SCALE) * |dy|, since dpt can be zero.
        double plane_dist = (node->axis == 0)
            ? std::abs(diff)
            : std::sqrt(Y_SCALE) * std::abs(diff);

        if (plane_dist < best_dist)
            search(far, qpt, qy, best_dist, best_idx, exclude);
    }

    void free_tree(Node* node) {
        if (!node) return;
        free_tree(node->left);
        free_tree(node->right);
        delete node;
    }
};

// CSV Handle

static void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

std::vector<Event> read_events(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << path << "\n";
        return {};
    }

    std::vector<Event> events;
    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        strip_cr(line);
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string token;
        Event e;
        std::getline(ss, token, ','); e.pt     = std::stod(token);
        std::getline(ss, token, ','); e.y      = std::stod(token);
        std::getline(ss, token, ','); e.weight = std::stod(token);
        events.push_back(e);
    }
    return events;
}

void write_events(const std::string& path, const std::vector<Event>& events) {
    std::ofstream file(path);
    file << std::setprecision(15);
    file << "pt,y,weight\n";
    for (const auto& e : events)
        file << e.pt << "," << e.y << "," << e.weight << "\n";
}

//stats on data
struct Stats {
    size_t total;
    size_t n_negative;
    double sum_w;
    double sum_abs_w;
};

Stats compute_stats(const std::vector<Event>& events) {
    Stats s{events.size(), 0, 0.0, 0.0};
    for (const auto& e : events) {
        s.sum_w     += e.weight;
        s.sum_abs_w += std::abs(e.weight);
        if (e.weight < 0) s.n_negative++;
    }
    return s;
}

// Negative weight fraction r- and inflation factor N(r-)/N(0).
//r- = (sum of |negative weights|) / (sum of |all weights|)
//N(r-)/N(0) = 1 / (1 - 2*r-)^2
//referencing these equations from the paper.
//The inflation factor tells you how many more events you need to achieve the same statistical precision as a purely positive sample.
static double neg_fraction(const Stats& s) {
    if (s.sum_abs_w == 0) return 0;
    double sum_neg = (s.sum_abs_w - s.sum_w) / 2.0;
    return sum_neg / s.sum_abs_w;
}

static double inflation_factor(double r_minus) {
    double denom = 1.0 - 2.0 * r_minus;
    if (std::abs(denom) < 1e-15) return std::numeric_limits<double>::infinity();
    return 1.0 / (denom * denom);
}

void print_stats(const std::string& label, const Stats& s) {
    double r = neg_fraction(s);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << label << ":\n"
              << "  Events:         " << s.total << "\n"
              << "  Negative:       " << s.n_negative << "\n"
              << "  Sum(w):         " << s.sum_w << "\n"
              << "  Sum(|w|):       " << s.sum_abs_w << "\n"
              << "  r- :            " << r << "\n"
              << "  N(r-)/N(0):     " << inflation_factor(r) << "\n\n";
}


//----Cell Resampling---
//algo explained in README.md
std::vector<CellRecord> cell_resample(std::vector<Event>& events) {
    KDTree tree(events);

    //Sort by most negative first — this ensures the seeds that need the
    //Largest cells are processed while the surrounding positive-weight
    std::vector<int> seeds;
    for (size_t i = 0; i < events.size(); i++) {
        if (events[i].weight < 0)
            seeds.push_back(static_cast<int>(i));
    }
    std::sort(seeds.begin(), seeds.end(), [&](int a, int b) {
        return events[a].weight < events[b].weight; //for most negative first strategy
    });

    std::vector<CellRecord> records;
    int skipped = 0;

    for (int seed_idx : seeds) {
        //skipping seeds whose weight became non negative from a previous cell
        if (events[seed_idx].weight >= 0) {
            skipped++;
            continue;
        }

        //growing cell around this seed
        std::vector<int> cell = {seed_idx};
        std::unordered_set<int> in_cell = {seed_idx};
        double cell_weight = events[seed_idx].weight;
        double max_radius  = 0.0;

        while (cell_weight < 0) {
            int nn = tree.find_nearest(events[seed_idx].pt,
                                       events[seed_idx].y,
                                       in_cell);
            if (nn == -1) {
                std::cerr << "Warning: exhausted all neighbors for seed "
                          << seed_idx << " (cell weight = "
                          << cell_weight << ")\n";
                break;
            }

            cell.push_back(nn);
            in_cell.insert(nn);
            cell_weight += events[nn].weight;

            double r = scaled_distance(events[seed_idx].pt, events[seed_idx].y,
                                       events[nn].pt, events[nn].y);
            max_radius = std::max(max_radius, r);
        }

        //weight distribution from the eqn 2.2 mentioned in the arXiv 2109.07851
        if (cell_weight >= 0) {
            double sum_abs = 0.0;
            for (int i : cell)
                sum_abs += std::abs(events[i].weight);

            if (sum_abs > 0) {
                for (int i : cell)
                    events[i].weight = (std::abs(events[i].weight) / sum_abs)
                                       * cell_weight;
            }
        }

        records.push_back({static_cast<int>(cell.size()), max_radius});
    }
    std::cout << "Cell construction:\n"
              << "  Total seeds:    " << seeds.size() << "\n"
              << "  Skipped:        " << skipped
              << " (weight became non-negative from earlier cells)\n"
              << "  Cells formed:   " << records.size() << "\n";

    if (!records.empty()) {
        double avg_size = 0;
        int max_size = 0;
        for (const auto& r : records) {
            avg_size += r.size;
            max_size = std::max(max_size, r.size);
        }
        avg_size /= records.size();

        std::vector<double> radii;
        for (const auto& r : records)
            radii.push_back(r.radius);
        std::sort(radii.begin(), radii.end());
        double median_radius = radii[radii.size() / 2];

        std::cout << std::fixed << std::setprecision(2)
                  << "  Avg cell size:  " << avg_size << " events\n"
                  << "  Max cell size:  " << max_size << " events\n"
                  << "  Median radius:  " << median_radius << "\n"
                  << "  Max radius:     " << radii.back() << "\n\n";
    }

    return records;
}

int main(int argc, char* argv[]) {
    std::string input_path  = "combined_before.csv";
    std::string output_path = "combined_after.csv";

    if (argc >= 2) input_path  = argv[1];
    if (argc >= 3) output_path = argv[2];

    auto events = read_events(input_path);
    if (events.empty()) {
        std::cerr << "No events loaded.\n";
        return 1;
    }
    Stats before = compute_stats(events);
    print_stats("Before resampling", before);
    auto records = cell_resample(events);

    Stats after = compute_stats(events);
    print_stats("After resampling", after);

    double delta = std::abs(after.sum_w - before.sum_w);
    std::cout << "Weight conservation check:\n"
              << "  Before: " << std::fixed << std::setprecision(6)
              << before.sum_w << "\n"
              << "  After:  " << after.sum_w << "\n"
              << "  Delta:  " << std::scientific << std::setprecision(2)
              << delta << "\n"
              << "  Status: " << (delta < 1e-6 ? "PASS" : "FAIL") << "\n\n";

    std::cout << "Negative weight check:\n"
              << "  Remaining: " << after.n_negative << "\n"
              << "  Status:    "
              << (after.n_negative == 0 ? "PASS" : "FAIL") << "\n\n";

    write_events(output_path, events);
    std::cout << "Written resampled events to " << output_path << "\n";

    return 0;
}
