/*
Born Projection (Task 1)

Maps real emission events from (n+1)-body phase space into born (n-body) kinematics and combines them with virtual events into a single dataset for cell resampling.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>

struct Event {
    double pt;
    double y;
    double weight;
};

//csv processing

static void strip_cr(std::string& s) {
    if (!s.empty() && s.back() == '\r')
        s.pop_back();
}

static double parse_field(std::istringstream& ss) {
    std::string token;
    std::getline(ss, token, ',');
    return std::stod(token);
}

static void skip_field(std::istringstream& ss) {
    std::string token;
    std::getline(ss, token, ',');
}

//virtual_events.csv: id, pt, y, weight
std::vector<Event> read_virtual(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << path << "\n";
        return {};
    }

    std::vector<Event> events;
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        strip_cr(line);
        if (line.empty()) continue;

        std::istringstream ss(line);
        skip_field(ss);

        Event e;
        e.pt = parse_field(ss);
        e.y = parse_field(ss);
        e.weight = parse_field(ss);
        events.push_back(e);
    }
    return events;
}

//real_events.csv: id, pt_real, y_real, z_gluon, weight
//Born projection applied inline by using pt = pt_real + z_gluon, y = y_real
std::vector<Event> read_real_projected(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << path << "\n";
        return {};
    }

    std::vector<Event> events;
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        strip_cr(line);
        if (line.empty()) continue;

        std::istringstream ss(line);
        skip_field(ss);

        double pt_real=parse_field(ss);
        double y_real=parse_field(ss);
        double z_gluon=parse_field(ss);
        double weight=parse_field(ss);

        Event e;
        e.pt = pt_real+z_gluon;//note this
        e.y = y_real;//note this
        e.weight = weight;
        events.push_back(e);
    }
    return events;
}


void write_events_csv(const std::string& path, const std::vector<Event>& events) {
    std::ofstream file(path);
    file << std::setprecision(15);
    file << "pt,y,weight\n";
    for (const auto& e : events)
        file << e.pt << "," << e.y << "," << e.weight << "\n";
}

//diagnostics

struct DatasetStats {
    size_t total;
    size_t n_negative;
    double sum_w;
    double sum_abs_w;
    double pt_min, pt_max;
    double y_min, y_max;
};

DatasetStats compute_stats(const std::vector<Event>& events) {
    DatasetStats s{};
    s.total = events.size();
    s.pt_min = s.y_min = 1e30;
    s.pt_max = s.y_max = -1e30;

    for (const auto& e : events) {
        s.sum_w+= e.weight;
        s.sum_abs_w+=std::abs(e.weight);
        if (e.weight < 0) s.n_negative++;
        s.pt_min=std::min(s.pt_min, e.pt);
        s.pt_max=std::max(s.pt_max, e.pt);
        s.y_min=std::min(s.y_min, e.y);
        s.y_max=std::max(s.y_max, e.y);
    }
    return s;
}

void print_stats(const std::string& label, const DatasetStats& s) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << label << ":\n"
              << "  Events:       " << s.total << " (" << s.n_negative << " negative)\n"
              << "  Sum(w):       " << s.sum_w << "\n"
              << "  Sum(|w|):     " << s.sum_abs_w << "\n"
              << "  pt range:     [" << s.pt_min << ", " << s.pt_max << "]\n"
              << "  y  range:     [" << s.y_min  << ", " << s.y_max  << "]\n\n";
}



int main(int argc, char* argv[]) {
    std::string virtual_path = "data/virtual_events.csv";
    std::string real_path    = "data/real_events.csv";
    std::string output_path  = "combined_before.csv";

    if (argc >= 3) {
        virtual_path=argv[1];
        real_path=argv[2];
    }

    auto virtual_events=read_virtual(virtual_path);
    auto real_projected=read_real_projected(real_path);

    print_stats("Virtual events (as-read)", compute_stats(virtual_events));
    print_stats("Real events (after Born projection)", compute_stats(real_projected));

    std::vector<Event> combined;
    combined.reserve(virtual_events.size() + real_projected.size());
    combined.insert(combined.end(), virtual_events.begin(), virtual_events.end());
    combined.insert(combined.end(), real_projected.begin(), real_projected.end());

    print_stats("Combined (before resampling)", compute_stats(combined));
    write_events_csv(output_path, combined);

    std::cout << "Written " << combined.size() << " events to " << output_path << "\n";
    return 0;
}
