#include "Vtop_ann.h"
#include "verilated.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int N_INPUT = 784;
constexpr int PIXEL_WIDTH = 8;
constexpr int N_WORDS = (N_INPUT * PIXEL_WIDTH + 31) / 32;

void tick(Vtop_ann& top) {
    top.clk = 0;
    top.eval();
    top.clk = 1;
    top.eval();
}

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

std::vector<uint8_t> read_hex_bytes(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<uint8_t> values;
    std::string token;
    while (in >> token) {
        values.push_back(static_cast<uint8_t>(std::stoul(token, nullptr, 16) & 0xff));
    }
    return values;
}

std::vector<int> read_labels(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<int> labels;
    std::string token;
    while (in >> token) {
        labels.push_back(static_cast<int>(std::stoul(token, nullptr, 16) & 0xf));
    }
    return labels;
}

void clear_pixels(Vtop_ann& top) {
    for (int w = 0; w < N_WORDS; ++w) {
        top.image_pixels[w] = 0;
    }
}

void load_image(Vtop_ann& top, const std::vector<uint8_t>& pixels, int image_index) {
    clear_pixels(top);
    const int base = image_index * N_INPUT;
    for (int p = 0; p < N_INPUT; ++p) {
        const int bit = p * PIXEL_WIDTH;
        const int word = bit / 32;
        const int shift = bit % 32;
        top.image_pixels[word] |= static_cast<uint32_t>(pixels[base + p]) << shift;
    }
}

void reset(Vtop_ann& top) {
    top.rst_n = 0;
    top.start = 0;
    clear_pixels(top);
    for (int i = 0; i < 4; ++i) tick(top);
    top.rst_n = 1;
    for (int i = 0; i < 2; ++i) tick(top);
}

int classify(Vtop_ann& top, const std::vector<uint8_t>& pixels, int image_index,
             int max_cycles_per_image) {
    load_image(top, pixels, image_index);
    tick(top);
    top.start = 1;
    tick(top);
    top.start = 0;

    int cycles = 0;
    while (!top.done) {
        tick(top);
        if (++cycles > max_cycles_per_image) {
            throw std::runtime_error("timeout while classifying image " +
                                     std::to_string(image_index));
        }
    }
    const int pred = static_cast<int>(top.prediction);
    tick(top);
    return pred;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const int first = parse_plusarg_int(argc, argv, "first", 0);
    int last = parse_plusarg_int(argc, argv, "last", 4);
    const std::string tag = parse_plusarg_string(argc, argv, "tag", "_vl");
    const int max_cycles = parse_plusarg_int(argc, argv, "max_cycles", 1000000);

    const auto pixels = read_hex_bytes("ann_pixels.mem");
    const auto labels = read_labels("snn_labels.mem");
    const int num_images = static_cast<int>(labels.size());
    if (pixels.size() != static_cast<size_t>(num_images * N_INPUT)) {
        throw std::runtime_error("ann_pixels.mem size does not match labels/input width");
    }
    last = std::min(last, num_images - 1);
    const int nrun = last - first + 1;
    if (first < 0 || nrun <= 0) {
        throw std::runtime_error("invalid image range");
    }

    std::ofstream classify_csv("classify_ann" + tag + ".csv");
    classify_csv << "image,label,prediction,correct,active_cycles,mac_ops\n";

    Vtop_ann top;
    reset(top);

    long long total_active = 0;
    long long total_macs = 0;
    int correct = 0;

    std::cout << "==========================================================\n";
    std::cout << "  Verilator INT8 ANN Baseline\n";
    std::cout << "  images " << first << ".." << last << "\n";
    std::cout << "==========================================================\n";

    for (int img = first; img <= last; ++img) {
        const int pred = classify(top, pixels, img, max_cycles);
        const int ok = (pred == labels[img]) ? 1 : 0;
        correct += ok;
        total_active += static_cast<long long>(top.active_cycles);
        total_macs += static_cast<long long>(top.total_mac_ops);
        classify_csv << img << "," << labels[img] << "," << pred << "," << ok
                     << "," << top.active_cycles << "," << top.total_mac_ops << "\n";
        if (img - first < 12) {
            std::cout << "  image " << img << ": label=" << labels[img]
                      << " predict=" << pred << (ok ? "  OK" : "  x") << "\n";
        }
    }

    std::ofstream metrics_csv("metrics_classify_ann" + tag + ".csv");
    metrics_csv << "metric,value\n";
    metrics_csv << "design,int8_ann_verilator\n";
    metrics_csv << "num_images," << nrun << "\n";
    metrics_csv << "correct," << correct << "\n";
    metrics_csv << "accuracy_pct," << std::fixed << std::setprecision(2)
                << (100.0 * correct / std::max(nrun, 1)) << "\n";
    metrics_csv << "active_cycles," << total_active << "\n";
    metrics_csv << "mac_ops," << total_macs << "\n";
    metrics_csv << "synapse_ops," << total_macs << "\n";

    std::cout << "\n==========================================================\n";
    std::cout << "  Classification accuracy: " << correct << " / " << nrun
              << " = " << std::fixed << std::setprecision(2)
              << (100.0 * correct / std::max(nrun, 1)) << "%\n";
    std::cout << "  INT8 ANN work:\n";
    std::cout << "    Active cycles:       " << total_active << "\n";
    std::cout << "    MAC ops:             " << total_macs << "\n";
    std::cout << "==========================================================\n";

    top.final();
    return 0;
}

