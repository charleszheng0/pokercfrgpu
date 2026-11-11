// Training data for the leaf value network: one file per solved flop.
//
// A file holds the state at every line that ends the flop in a full DCFR solve: the
// average-strategy reach of each hand of both players and the value of each hand.
#pragma once
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "game.hpp"

namespace poker {

struct VnetLeaf {
  double contrib = 0, stack_left = 0;
  bool all_in = false;
  std::vector<float> reach[2], value[2];
};

struct VnetFile {
  std::string preset;
  std::vector<Card> board;
  double pot = 0, stack = 0, expl_pct = 0;
  int iterations = 0;
  std::vector<uint16_t> combos[2];  // combo index of each hand (hand order of the Game)
  std::vector<VnetLeaf> leaves;
};

namespace vnet_io {
template <class T> void put(FILE* f, const T& v) { std::fwrite(&v, sizeof v, 1, f); }
template <class T> void get(FILE* f, T& v) {
  if (std::fread(&v, sizeof v, 1, f) != 1) throw std::runtime_error("vnet file: short read");
}
template <class T> void put_vec(FILE* f, const std::vector<T>& v) {
  put<uint32_t>(f, static_cast<uint32_t>(v.size()));
  if (!v.empty()) std::fwrite(v.data(), sizeof(T), v.size(), f);
}
template <class T> void get_vec(FILE* f, std::vector<T>& v) {
  uint32_t n;
  get(f, n);
  v.resize(n);
  if (n && std::fread(v.data(), sizeof(T), n, f) != n) throw std::runtime_error("vnet file: short read");
}
}  // namespace vnet_io

inline void write_vnet_file(const std::string& path, const VnetFile& d) {
  using namespace vnet_io;
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  std::fwrite("VNET1", 1, 5, f);
  std::vector<char> name(d.preset.begin(), d.preset.end());
  put_vec(f, name);
  put_vec(f, d.board);
  put(f, d.pot), put(f, d.stack), put(f, d.expl_pct), put<int32_t>(f, d.iterations);
  put_vec(f, d.combos[0]), put_vec(f, d.combos[1]);
  put<uint32_t>(f, static_cast<uint32_t>(d.leaves.size()));
  for (const auto& l : d.leaves) {
    put(f, l.contrib), put(f, l.stack_left), put<uint8_t>(f, l.all_in);
    for (int p = 0; p < 2; ++p) put_vec(f, l.reach[p]), put_vec(f, l.value[p]);
  }
  std::fclose(f);
}

inline VnetFile read_vnet_file(const std::string& path) {
  using namespace vnet_io;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot read " + path);
  VnetFile d;
  char magic[5];
  if (std::fread(magic, 1, 5, f) != 5 || std::string(magic, 5) != "VNET1") {
    std::fclose(f);
    throw std::runtime_error(path + ": not a vnet file");
  }
  std::vector<char> name;
  get_vec(f, name);
  d.preset.assign(name.begin(), name.end());
  get_vec(f, d.board);
  get(f, d.pot), get(f, d.stack), get(f, d.expl_pct);
  int32_t it;
  get(f, it);
  d.iterations = it;
  get_vec(f, d.combos[0]), get_vec(f, d.combos[1]);
  uint32_t nl;
  get(f, nl);
  d.leaves.resize(nl);
  for (auto& l : d.leaves) {
    uint8_t a;
    get(f, l.contrib), get(f, l.stack_left), get(f, a);
    l.all_in = a != 0;
    for (int p = 0; p < 2; ++p) get_vec(f, l.reach[p]), get_vec(f, l.value[p]);
  }
  std::fclose(f);
  return d;
}

}  // namespace poker
