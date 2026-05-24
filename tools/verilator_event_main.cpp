#ifndef EVENT_TOP_HEADER
#include "Vtop.h"
using EventTop = Vtop;
#else
#include EVENT_TOP_HEADER
using EventTop = EVENT_TOP_CLASS;
#endif
#include "verilated.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#ifndef SNN_N_TOTAL
#error "SNN_N_TOTAL must be passed with -DSNN_N_TOTAL=<value>"
#endif
#ifndef SNN_N_INPUT
#error "SNN_N_INPUT must be passed with -DSNN_N_INPUT=<value>"
#endif
#ifndef SNN_N_OUTPUT
#error "SNN_N_OUTPUT must be passed with -DSNN_N_OUTPUT=<value>"
#endif
#ifndef SNN_ID_BIAS
#error "SNN_ID_BIAS must be passed with -DSNN_ID_BIAS=<value>"
#endif
#ifndef SNN_ID_OUTPUT_BASE
#error "SNN_ID_OUTPUT_BASE must be passed with -DSNN_ID_OUTPUT_BASE=<value>"
#endif
#ifndef SNN_T_STEPS
#error "SNN_T_STEPS must be passed with -DSNN_T_STEPS=<value>"
#endif
#ifndef EVENT_DESIGN_NAME
#define EVENT_DESIGN_NAME "event_driven_verilator"
#endif
#ifndef SNN_MEMBRANE_WIDTH
#define SNN_MEMBRANE_WIDTH 16
#endif

namespace {

constexpr int N_TOTAL = SNN_N_TOTAL;
constexpr int N_INPUT = SNN_N_INPUT;
constexpr int N_OUTPUT = SNN_N_OUTPUT;
constexpr int ID_BIAS = SNN_ID_BIAS;
constexpr int OUT_BASE = SNN_ID_OUTPUT_BASE;
constexpr int T_STEPS = SNN_T_STEPS;
constexpr int MEMBRANE_WIDTH = SNN_MEMBRANE_WIDTH;
constexpr int SPIKE_WORDS = (N_TOTAL + 31) / 32;
using Clock = std::chrono::steady_clock;

int parse_plusarg_int(int argc, char** argv, const std::string& name, int fallback) {
    const std::string prefix = "+" + name + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind(prefix, 0) == 0) {
            return std::stoi(arg.substr(prefix.size()));
        }
    }
    return fallback;
}

std::string parse_plusarg_string(int argc, char** argv, const std::string& name,
                                 const std::string& fallback) {
    const std::string prefix = "+" + name + "=";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind(prefix, 0) == 0) {
            return arg.substr(prefix.size());
        }
    }
    return fallback;
}

std::string format_duration(double seconds) {
    const int total = static_cast<int>(seconds + 0.5);
    const int hours = total / 3600;
    const int minutes = (total / 60) % 60;
    const int secs = total % 60;
    std::ostringstream out;
    if (hours > 0) {
        out << hours << "h";
    }
    if (hours > 0 || minutes > 0) {
        out << minutes << "m";
    }
    out << secs << "s";
    return out.str();
}

std::string progress_bar(int done, int total) {
    constexpr int width = 30;
    const int filled = std::max(0, std::min(width, (done * width) / std::max(total, 1)));
    return "[" + std::string(filled, '#') + std::string(width - filled, '.') + "]";
}

void print_progress(int done, int total, int correct, Clock::time_point start) {
    const auto now = Clock::now();
    const double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - start).count();
    const double rate = done / std::max(elapsed, 1e-9);
    const double eta = (total - done) / std::max(rate, 1e-9);
    const double pct = 100.0 * done / std::max(total, 1);
    const double acc = 100.0 * correct / std::max(done, 1);

    std::cout << "  progress " << progress_bar(done, total)
              << " " << std::setw(4) << done << "/" << total
              << "  " << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%"
              << "  acc=" << std::setprecision(2) << acc << "%"
              << "  rate=" << std::setprecision(2) << rate << " img/s"
              << "  elapsed=" << format_duration(elapsed)
              << "  eta=" << format_duration(eta) << "\n";
}

std::vector<std::string> read_spike_words(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<std::string> words;
    std::string token;
    while (in >> token) {
        if (static_cast<int>(token.size()) != N_INPUT) {
            throw std::runtime_error(path + " contains a spike word with wrong width");
        }
        words.push_back(token);
    }
    return words;
}

std::vector<int> read_labels(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<int> labels;
    std::string token;
    while (in >> token) {
        labels.push_back(std::stoi(token));
    }
    return labels;
}

bool spike_word_bit(const std::string& word, int pixel) {
    // Verilog $readmemb maps the rightmost character to bit 0.
    return word[N_INPUT - 1 - pixel] == '1';
}

bool output_spike_bit(const EventTop& top, int neuron_id) {
    return ((top.neuron_spikes[neuron_id / 32] >> (neuron_id % 32)) & 1U) != 0;
}

void count_output_spikes(const EventTop& top, std::vector<int>& out_count) {
    for (int o = 0; o < N_OUTPUT; ++o) {
        if (output_spike_bit(top, OUT_BASE + o)) {
            ++out_count[o];
        }
    }
}

void tick(EventTop& top, std::vector<int>* out_count = nullptr) {
    top.clk = 0;
    top.eval();
    top.clk = 1;
    top.eval();
    if (out_count != nullptr) {
        count_output_spikes(top, *out_count);
    }
}

void reset(EventTop& top) {
    top.rst_n = 0;
    top.ext_spike_id = 0;
    top.ext_spike_valid = 0;
    top.start_step = 0;
#ifdef EVENT_SCORE_READOUT
    top.score_read_idx = 0;
    top.score_read_en = 0;
#endif
    for (int i = 0; i < 4; ++i) tick(top);
    top.rst_n = 1;
    for (int i = 0; i < 2; ++i) tick(top);

    int guard = 0;
    while (top.step_busy) {
        if (++guard > (N_TOTAL * 4 + 1024)) {
            throw std::runtime_error("timeout waiting for reset clear");
        }
        tick(top);
    }
}

void inject_spike(EventTop& top, int neuron_id, std::vector<int>& out_count, int& cycles) {
    top.ext_spike_id = neuron_id;
    top.ext_spike_valid = 1;
    tick(top, &out_count);
    ++cycles;
    top.ext_spike_valid = 0;
    tick(top, &out_count);
    ++cycles;
}

void run_step(EventTop& top, std::vector<int>& out_count, int& cycles,
              int max_cycles_per_image, int image_index) {
    top.start_step = 1;
    tick(top, &out_count);
    ++cycles;
    top.start_step = 0;

    while (!top.step_busy) {
        if (++cycles > max_cycles_per_image) {
            throw std::runtime_error("timeout waiting for step start on image " +
                                     std::to_string(image_index));
        }
        tick(top, &out_count);
    }

    while (top.step_busy) {
        if (++cycles > max_cycles_per_image) {
            throw std::runtime_error("timeout while classifying image " +
                                     std::to_string(image_index));
        }
        tick(top, &out_count);
    }

    tick(top, &out_count);
    ++cycles;
}

int argmax_first(const std::vector<int>& values) {
    int best_idx = 0;
    int best = values[0];
    for (int i = 1; i < static_cast<int>(values.size()); ++i) {
        if (values[i] > best) {
            best = values[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int sign_extend(uint32_t value, int width) {
    if (width <= 0 || width >= 32) {
        return static_cast<int>(value);
    }
    const uint32_t sign = 1u << (width - 1);
    const uint32_t mask = (1u << width) - 1u;
    value &= mask;
    return static_cast<int>((value ^ sign) - sign);
}

#ifdef EVENT_SCORE_READOUT
std::vector<int> read_output_scores(EventTop& top, int& cycles) {
    std::vector<int> scores(N_OUTPUT, 0);
    for (int o = 0; o < N_OUTPUT; ++o) {
        top.score_read_idx = o;
        top.score_read_en = 1;
        tick(top);
        ++cycles;
        top.score_read_en = 0;
        tick(top);
        ++cycles;
        if (!top.score_read_valid) {
            tick(top);
            ++cycles;
        }
        scores[o] = sign_extend(static_cast<uint32_t>(top.score_read_data), MEMBRANE_WIDTH);
    }
    return scores;
}
#endif

struct ClassificationResult {
    int prediction = 0;
    std::vector<int> scores;
};

ClassificationResult classify(EventTop& top, const std::vector<std::string>& spike_words, int image_index,
                              int max_cycles_per_image, std::ofstream* trace_csv = nullptr) {
    reset(top);
    std::vector<int> out_count(N_OUTPUT, 0);
    int cycles = 0;

    for (int t = 0; t < T_STEPS; ++t) {
        const std::string& word = spike_words[image_index * T_STEPS + t];
        for (int p = 0; p < N_INPUT; ++p) {
            if (spike_word_bit(word, p)) {
                inject_spike(top, p, out_count, cycles);
            }
        }
        inject_spike(top, ID_BIAS, out_count, cycles);
        run_step(top, out_count, cycles, max_cycles_per_image, image_index);
#ifdef EVENT_SCORE_READOUT
        if (trace_csv != nullptr) {
            const auto step_scores = read_output_scores(top, cycles);
            *trace_csv << t;
            for (int value : step_scores) {
                *trace_csv << "," << value;
            }
            *trace_csv << "\n";
        }
#else
        if (trace_csv != nullptr) {
            *trace_csv << t;
            for (int value : out_count) {
                *trace_csv << "," << value;
            }
            *trace_csv << "\n";
        }
#endif
    }

    tick(top);
#ifdef EVENT_SCORE_READOUT
    const auto scores = read_output_scores(top, cycles);
    return {argmax_first(scores), scores};
#else
    return {argmax_first(out_count), out_count};
#endif
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const int first = parse_plusarg_int(argc, argv, "first", 0);
    int last = parse_plusarg_int(argc, argv, "last", 4);
    const std::string tag = parse_plusarg_string(argc, argv, "tag", "_vl");
    const int max_cycles = parse_plusarg_int(argc, argv, "max_cycles", 50000000);
    const int trace_image = parse_plusarg_int(argc, argv, "trace_image", -1);

    const auto spike_words = read_spike_words("snn_spikes.mem");
    const auto labels = read_labels("snn_labels.mem");
    const int num_images = static_cast<int>(labels.size());
    if (spike_words.size() != static_cast<size_t>(num_images * T_STEPS)) {
        throw std::runtime_error("snn_spikes.mem size does not match labels/timesteps");
    }
    last = std::min(last, num_images - 1);
    const int nrun = last - first + 1;
    if (first < 0 || nrun <= 0) {
        throw std::runtime_error("invalid image range");
    }

    std::ofstream classify_csv("classify_event" + tag + ".csv");
    classify_csv << "image,label,prediction,correct,active_cycles,deliveries,spikes";
    for (int o = 0; o < N_OUTPUT; ++o) {
        classify_csv << ",score" << o;
    }
    classify_csv << "\n";

    EventTop top;
    reset(top);

    long long total_active = 0;
    long long total_deliveries = 0;
    long long total_events = 0;
    long long total_spikes = 0;
    int correct = 0;

    std::cout << "==========================================================\n";
    std::cout << "  Verilator Event-Driven SNN Accelerator\n";
    std::cout << "  " << N_TOTAL << " neurons, " << T_STEPS << " timesteps/image\n";
    std::cout << "  images " << first << ".." << last << "\n";
    std::cout << "==========================================================\n";

    const auto run_start = Clock::now();
    const int progress_interval = std::max(1, nrun / 20);

    for (int img = first; img <= last; ++img) {
        std::ofstream trace_csv;
        std::ofstream* trace_ptr = nullptr;
        if (img == trace_image) {
            trace_csv.open("trace_event" + tag + "_img" + std::to_string(img) + ".csv");
            trace_csv << "timestep";
            for (int o = 0; o < N_OUTPUT; ++o) {
                trace_csv << ",score" << o;
            }
            trace_csv << "\n";
            trace_ptr = &trace_csv;
        }

        const auto result = classify(top, spike_words, img, max_cycles, trace_ptr);
        const int pred = result.prediction;
        const int ok = (pred == labels[img]) ? 1 : 0;
        correct += ok;
        total_active += static_cast<long long>(top.active_cycles);
        total_deliveries += static_cast<long long>(top.router_deliveries);
        total_events += static_cast<long long>(top.router_events);
        total_spikes += static_cast<long long>(top.total_spikes_fired);
        classify_csv << img << "," << labels[img] << "," << pred << "," << ok
                     << "," << top.active_cycles << "," << top.router_deliveries
                     << "," << top.total_spikes_fired;
        for (int o = 0; o < N_OUTPUT; ++o) {
            classify_csv << "," << result.scores[o];
        }
        classify_csv << "\n";
        if (img - first < 12) {
            std::cout << "  image " << img << ": label=" << labels[img]
                      << " predict=" << pred << (ok ? "  OK" : "  x") << "\n";
        }
        const int done = img - first + 1;
        if (done == nrun || (done >= 12 && done % progress_interval == 0)) {
            print_progress(done, nrun, correct, run_start);
        }
    }

    std::ofstream metrics_csv("metrics_classify_event" + tag + ".csv");
    metrics_csv << "metric,value\n";
    metrics_csv << "design," << EVENT_DESIGN_NAME << "\n";
    metrics_csv << "first_image," << first << "\n";
    metrics_csv << "last_image," << last << "\n";
    metrics_csv << "num_images," << nrun << "\n";
    metrics_csv << "correct," << correct << "\n";
    metrics_csv << "accuracy_pct," << std::fixed << std::setprecision(2)
                << (100.0 * correct / std::max(nrun, 1)) << "\n";
#ifdef EVENT_SCORE_READOUT
    metrics_csv << "readout,membrane\n";
#else
    metrics_csv << "readout,spike_count\n";
#endif
    metrics_csv << "active_cycles," << total_active << "\n";
    metrics_csv << "synapse_ops," << total_deliveries << "\n";
    metrics_csv << "router_events," << total_events << "\n";
    metrics_csv << "spikes_fired," << total_spikes << "\n";

    std::cout << "\n==========================================================\n";
    std::cout << "  Classification accuracy: " << correct << " / " << nrun
              << " = " << std::fixed << std::setprecision(2)
              << (100.0 * correct / std::max(nrun, 1)) << "%\n";
    std::cout << "  Event-driven work:\n";
    std::cout << "    Active cycles:       " << total_active << "\n";
    std::cout << "    Synapse deliveries:  " << total_deliveries << "\n";
    std::cout << "    Router events:       " << total_events << "\n";
    std::cout << "    Spikes fired:        " << total_spikes << "\n";
    std::cout << "==========================================================\n";

    top.final();
    return 0;
}
