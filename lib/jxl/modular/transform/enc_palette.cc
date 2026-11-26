// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/enc_palette.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/enc_transform.h"
#include "lib/jxl/modular/transform/palette.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

namespace palette_internal {

static constexpr bool kEncodeToHighQualityImplicitPalette = true;

// Inclusive.
static constexpr int kMinImplicitPaletteIndex = -(2 * 72 - 1);

float ColorDistance(const std::vector<float>& JXL_RESTRICT a,
                    const std::vector<pixel_type>& JXL_RESTRICT b) {
  JXL_DASSERT(a.size() == b.size());
  float distance = 0;
  float ave3 = 0;
  if (a.size() >= 3) {
    ave3 = (a[0] + b[0] + a[1] + b[1] + a[2] + b[2]) * (1.21f / 3.0f);
  }
  float sum_a = 0;
  float sum_b = 0;
  for (size_t c = 0; c < a.size(); ++c) {
    const float difference =
        static_cast<float>(a[c]) - static_cast<float>(b[c]);
    float weight = c == 0 ? 3 : c == 1 ? 5 : 2;
    if (c < 3 && (a[c] + b[c] >= ave3)) {
      const float add_w[3] = {
          1.15,
          1.15,
          1.12,
      };
      weight += add_w[c];
      if (c == 2 && ((a[2] + b[2]) < 1.22 * ave3)) {
        weight -= 0.5;
      }
    }
    distance += difference * difference * weight * weight;
    const int sum_weight = c == 0 ? 3 : c == 1 ? 5 : 1;
    sum_a += a[c] * sum_weight;
    sum_b += b[c] * sum_weight;
  }
  distance *= 4;
  float sum_difference = sum_a - sum_b;
  distance += sum_difference * sum_difference;
  return distance;
}

static int QuantizeColorToImplicitPaletteIndex(
    const std::vector<pixel_type>& color, const int palette_size,
    const int bit_depth, bool high_quality) {
  int index = 0;
  int quant = (1 << bit_depth) - 1;
  // When bit_depth == 1, half would be equal quant without this fuse.
  int half = (bit_depth > 1) ? (1 << (bit_depth - 1)) : 0;
  if (high_quality) {
    int multiplier = 1;
    for (int value : color) {
      int quantized = ((kLargeCube - 1) * value + half) / quant;
      JXL_DASSERT((quantized % kLargeCube) == quantized);
      index += quantized * multiplier;
      multiplier *= kLargeCube;
    }
    return index + palette_size + kLargeCubeOffset;
  } else {
    int multiplier = 1;
    for (int value : color) {
      value -= 1 << (std::max(0, bit_depth - 3));
      value = std::max(0, value);
      int quantized = ((kLargeCube - 1) * value + half) / quant;
      JXL_DASSERT((quantized % kLargeCube) == quantized);
      if (quantized > kSmallCube - 1) {
        quantized = kSmallCube - 1;
      }
      index += quantized * multiplier;
      multiplier *= kSmallCube;
    }
    return index + palette_size;
  }
}

}  // namespace palette_internal

namespace {

struct FastColorMap {
  struct Entry {
    std::vector<pixel_type> color;
    size_t idx;
    bool occupied = false;
  };
  std::vector<Entry> table;
  size_t mask = 0;

  void Init(const std::vector<std::vector<pixel_type>>& palette) {
    size_t cap = 16;
    while (cap < palette.size() * 3) cap <<= 1;
    table.clear();
    table.resize(cap);
    mask = cap - 1;
    for (size_t i = 0; i < palette.size(); ++i) {
      uint64_t h = Hash(palette[i]);
      size_t pos = h & mask;
      while (table[pos].occupied) {
        pos = (pos + 1) & mask;
      }
      table[pos].color = palette[i];
      table[pos].idx = i;
      table[pos].occupied = true;
    }
  }

  inline uint64_t Hash(const std::vector<pixel_type>& c) const {
    uint64_t h = 14695981039346656037ULL;
    for (pixel_type v : c) {
      h = (h ^ static_cast<uint64_t>(v)) * 1099511628211ULL;
    }
    return h;
  }

  inline int Find(const std::vector<pixel_type>& c) const {
    if (table.empty()) return -1;
    uint64_t h = Hash(c);
    size_t pos = h & mask;
    while (table[pos].occupied) {
      if (table[pos].color == c) return static_cast<int>(table[pos].idx);
      pos = (pos + 1) & mask;
    }
    return -1;
  }
};

int64_t ComputeOLACost(const std::vector<size_t>& remapping,
                       const std::vector<std::vector<uint32_t>>& matrix) {
  size_t n = remapping.size();
  int64_t total = 0;
  for (size_t i = 0; i < n; i++) {
    size_t vi = remapping[i];
    for (size_t j = i + 1; j < n; j++) {
      size_t vj = remapping[j];
      if (matrix[vi][vj] > 0) {
        total += static_cast<int64_t>(matrix[vi][vj]) * (j - i);
      }
    }
  }
  return total;
}

struct WeightedEdge {
  size_t to;
  int64_t weight;
};

using AdjList = std::vector<std::vector<WeightedEdge>>;

AdjList BuildAdjList(const std::vector<std::vector<uint32_t>>& matrix) {
  size_t n = matrix.size();
  AdjList adj(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i != j && matrix[i][j] > 0) {
        adj[i].push_back({j, static_cast<int64_t>(matrix[i][j])});
      }
    }
  }
  return adj;
}

std::vector<size_t> EZengReindexWithSeed(
    const std::vector<std::vector<uint32_t>>& matrix, size_t seed_u,
    size_t seed_v) {
  size_t num_colors = matrix.size();
  if (num_colors <= 2) {
    std::vector<size_t> remapping(num_colors);
    for (size_t i = 0; i < num_colors; ++i) remapping[i] = i;
    return remapping;
  }

  std::vector<size_t> remapping = {seed_u, seed_v};

  std::vector<std::pair<size_t, uint32_t>> sums;
  size_t best_sum_pos = 0;
  uint32_t best_sum_val = 0;
  for (size_t i = 0; i < num_colors; i++) {
    if (i == remapping[0] || i == remapping[1]) continue;
    uint32_t sum = matrix[remapping[0]][i] + matrix[remapping[1]][i];
    if (sum >= best_sum_val) {
      best_sum_pos = sums.size();
      best_sum_val = sum;
    }
    sums.push_back({i, sum});
  }

  while (!sums.empty()) {
    size_t best_index = sums[best_sum_pos].first;
    size_t m = remapping.size();
    size_t best_pos = 0;
    int64_t best_cost = std::numeric_limits<int64_t>::max();
    int64_t cross_cost = 0;
    int64_t new_cost = 0;
    int64_t left_w = 0;
    int64_t right_w = 0;
    const auto& best_row = matrix[best_index];
    for (size_t k = 0; k < m; k++) {
      int64_t w = best_row[remapping[k]];
      new_cost += w * static_cast<int64_t>(k + 1);
      right_w += w;
    }

    for (size_t p = 0; p <= m; p++) {
      int64_t total = new_cost + cross_cost;
      if (total < best_cost) {
        best_cost = total;
        best_pos = p;
      }

      if (p < m) {
        size_t rp = remapping[p];
        int64_t w_rp = best_row[rp];
        right_w -= w_rp;
        new_cost += left_w - right_w;
        left_w += w_rp;
        const auto& rp_row = matrix[rp];
        for (size_t b = p + 1; b < m; b++) {
          cross_cost += rp_row[remapping[b]];
        }
        for (size_t a = 0; a < p; a++) {
          cross_cost -= rp_row[remapping[a]];
        }
      }
    }
    remapping.insert(remapping.begin() + best_pos, best_index);

    std::swap(sums[best_sum_pos], sums.back());
    sums.pop_back();

    if (!sums.empty()) {
      best_sum_pos = 0;
      best_sum_val = 0;
      for (size_t i = 0; i < sums.size(); i++) {
        sums[i].second += best_row[sums[i].first];
        if (sums[i].second >= best_sum_val) {
          best_sum_pos = i;
          best_sum_val = sums[i].second;
        }
      }
    }
  }

  return remapping;
}

std::vector<size_t> EZengReindex(
    const std::vector<std::vector<uint32_t>>& matrix) {
  size_t num_colors = matrix.size();
  if (num_colors <= 2) {
    std::vector<size_t> remapping(num_colors);
    for (size_t i = 0; i < num_colors; ++i) remapping[i] = i;
    return remapping;
  }

  size_t best_u = 0, best_v = 1;
  uint32_t max_w = 0;
  for (size_t i = 0; i < num_colors; i++) {
    for (size_t j = i + 1; j < num_colors; j++) {
      if (matrix[i][j] >= max_w) {
        max_w = matrix[i][j];
        best_u = i;
        best_v = j;
      }
    }
  }

  return EZengReindexWithSeed(matrix, best_u, best_v);
}

void PairwiseSwapSearch(std::vector<size_t>& remapping, const AdjList& adj,
                        uint8_t max_dist, int max_iter = 50) {
  size_t num_colors = remapping.size();
  size_t b_limit =
      (max_dist == 0) ? num_colors : (static_cast<size_t>(max_dist) + 1);

  std::vector<size_t> pos(num_colors);
  for (size_t i = 0; i < num_colors; ++i) {
    pos[remapping[i]] = i;
  }

  int swaps = 1;
  int iter = 0;
  while (swaps > 0 && iter < max_iter) {
    swaps = 0;
    iter++;
    for (size_t a = 0; a + 1 < num_colors; a++) {
      size_t limit = std::min(num_colors, a + b_limit);
      int64_t pos_a = static_cast<int64_t>(a);
      for (size_t b = a + 1; b < limit; b++) {
        size_t va = remapping[a];
        size_t vb = remapping[b];
        int64_t pos_b = static_cast<int64_t>(b);
        int64_t delta = 0;
        for (const auto& edge : adj[va]) {
          if (edge.to == vb) continue;
          int64_t pos_i = static_cast<int64_t>(pos[edge.to]);
          delta +=
              edge.weight * (std::abs(pos_b - pos_i) - std::abs(pos_a - pos_i));
        }
        for (const auto& edge : adj[vb]) {
          if (edge.to == va) continue;
          int64_t pos_i = static_cast<int64_t>(pos[edge.to]);
          delta -=
              edge.weight * (std::abs(pos_b - pos_i) - std::abs(pos_a - pos_i));
        }
        if (delta < 0) {
          std::swap(remapping[a], remapping[b]);
          pos[va] = b;
          pos[vb] = a;
          swaps += 1;
        }
      }
    }
  }
}

void TwoOptSearch(std::vector<size_t>& remapping, const AdjList& adj) {
  size_t n = remapping.size();
  std::vector<size_t> pos(n);
  for (size_t i = 0; i < n; ++i) {
    pos[remapping[i]] = i;
  }

  bool improved = true;
  int iter = 0;
  while (improved && iter < 10) {
    improved = false;
    iter++;
    for (size_t a = 0; a + 1 < n; a++) {
      size_t b_limit = std::min(n, a + 33);
      for (size_t b = a + 1; b < b_limit; b++) {
        int64_t delta = 0;
        int64_t ab = static_cast<int64_t>(a + b);
        for (size_t k = a; k <= b; k++) {
          int64_t factor = ab - 2 * static_cast<int64_t>(k);
          if (factor == 0) continue;
          size_t vk = remapping[k];
          int64_t diff_k = 0;
          for (const auto& edge : adj[vk]) {
            size_t pi = pos[edge.to];
            if (pi < a) {
              diff_k += edge.weight;
            } else if (pi > b) {
              diff_k -= edge.weight;
            }
          }
          delta += factor * diff_k;
        }
        if (delta < 0) {
          std::reverse(remapping.begin() + a, remapping.begin() + b + 1);
          for (size_t k = a; k <= b; ++k) {
            pos[remapping[k]] = k;
          }
          improved = true;
        }
      }
    }
  }
}

// Identifies pixels covered by exact 2D (current + top row) repeated runs
// within each 256x256 tile. Because any palette reordering is a bijection,
// ClampedGradient residuals inside such repeated runs are identical across
// all orderings and get collapsed by LZ77 independently of palette order.
std::vector<uint8_t> ComputeLZ77MatchMask(const std::vector<uint16_t>& idx_img,
                                          size_t w, size_t h) {
  std::vector<uint8_t> mask(w * h, 0);
  const bool tiled = (w > 256 || h > 256);
  const size_t tile_dim = tiled ? 256 : std::max(w, h);
  constexpr size_t kHashSize = 4096;
  constexpr size_t kHashMask = kHashSize - 1;
  std::vector<int32_t> head(kHashSize, -1);

  for (size_t y0 = 0; y0 < h; y0 += tile_dim) {
    const size_t y1 = std::min(h, y0 + tile_dim);
    for (size_t x0 = 0; x0 < w; x0 += tile_dim) {
      const size_t x1 = std::min(w, x0 + tile_dim);
      const size_t tw = x1 - x0;
      if (tw < 4 || y1 - y0 < 2) continue;
      std::fill(head.begin(), head.end(), -1);

      for (size_t y = y0 + 1; y < y1; ++y) {
        const uint16_t* row = &idx_img[y * w];
        const uint16_t* prev = &idx_img[(y - 1) * w];
        size_t x = x0;
        while (x + 4 <= x1) {
          // Skip flat runs (already handled by zero-run RLE).
          if (row[x] == row[x + 1] && row[x] == row[x + 2] &&
              row[x] == row[x + 3] && row[x] == prev[x] &&
              row[x] == prev[x + 1] && row[x] == prev[x + 2] &&
              row[x] == prev[x + 3]) {
            x++;
            continue;
          }
          uint64_t hval = (static_cast<uint64_t>(row[x]) * 2654435761ULL) ^
                          (static_cast<uint64_t>(row[x + 1]) * 2246822519ULL) ^
                          (static_cast<uint64_t>(row[x + 2]) * 3266489917ULL) ^
                          (static_cast<uint64_t>(row[x + 3]) * 668265263ULL) ^
                          (static_cast<uint64_t>(prev[x]) * 374761393ULL) ^
                          (static_cast<uint64_t>(prev[x + 1]) * 1274126177ULL) ^
                          (static_cast<uint64_t>(prev[x + 2]) * 2869860233ULL) ^
                          (static_cast<uint64_t>(prev[x + 3]) * 1640531527ULL);
          size_t bucket = (hval >> 20) & kHashMask;
          int32_t prev_pos = head[bucket];
          int32_t cur_pos = static_cast<int32_t>((y - y0) * tw + (x - x0));
          head[bucket] = cur_pos;

          if (prev_pos >= 0) {
            size_t qy = y0 + static_cast<size_t>(prev_pos) / tw;
            size_t qx = x0 + static_cast<size_t>(prev_pos) % tw;
            if (qx + 4 <= x1) {
              const uint16_t* qrow = &idx_img[qy * w];
              const uint16_t* qprev = &idx_img[(qy - 1) * w];
              if (row[x] == qrow[qx] && row[x + 1] == qrow[qx + 1] &&
                  row[x + 2] == qrow[qx + 2] && row[x + 3] == qrow[qx + 3] &&
                  prev[x] == qprev[qx] && prev[x + 1] == qprev[qx + 1] &&
                  prev[x + 2] == qprev[qx + 2] &&
                  prev[x + 3] == qprev[qx + 3]) {
                size_t len = 4;
                size_t max_len = std::min(x1 - x, x1 - qx);
                while (len < max_len && row[x + len] == qrow[qx + len] &&
                       prev[x + len] == qprev[qx + len]) {
                  len++;
                }
                // Residuals at x+1..x+len-1 are guaranteed identical under any
                // palette permutation (since cur, left, top, topleft all
                // match).
                for (size_t k = 1; k < len; ++k) {
                  mask[y * w + x + k] = 1;
                }
                x += len;
                continue;
              }
            }
          }
          x++;
        }
      }
    }
  }
  return mask;
}

double EvaluatePaletteOrderCost(
    const std::vector<uint16_t>& idx_img, const std::vector<uint8_t>& lz_mask,
    size_t w, size_t h, const std::vector<std::vector<pixel_type>>& palette,
    const std::vector<size_t>& remapping, size_t nb) {
  const size_t n = remapping.size();
  std::vector<int32_t> pos(n);
  for (size_t i = 0; i < n; ++i) {
    pos[remapping[i]] = static_cast<int32_t>(i);
  }

  // Approximate bit cost of HybridUint (4, 1, 0) + multi-context Shannon
  // entropy on ClampedGradient residuals over the palette index channel,
  // with zero-run discounting to model LZ77 + MA tree behavior.
  constexpr size_t kNumCtx = 8;
  constexpr size_t kMaxTokens = 64;
  uint64_t counts[kNumCtx][kMaxTokens] = {};
  uint64_t ctx_totals[kNumCtx] = {};
  double extra_bits = 0.0;

  auto push_token = [&](size_t ctx, int32_t res) {
    uint32_t v =
        static_cast<uint32_t>(res >= 0 ? (res << 1) : ((-res << 1) - 1));
    uint32_t tok;
    if (v < 16) {
      tok = v;
    } else {
      uint32_t nbits = FloorLog2Nonzero(v);
      tok = 16 + ((nbits - 4) << 1) + ((v >> (nbits - 1)) & 1);
      if (tok >= kMaxTokens) tok = kMaxTokens - 1;
      extra_bits += static_cast<double>(nbits - 1);
    }
    counts[ctx][tok]++;
    ctx_totals[ctx]++;
  };

  auto flush_zero_run = [&](uint32_t& run_len, size_t run_ctx) {
    if (run_len == 0) return;
    if (run_len <= 2) {
      counts[run_ctx][0] += run_len;
      ctx_totals[run_ctx] += run_len;
    } else {
      // Model an LZ77 distance-1 match of length `run_len`.
      counts[run_ctx][0]++;
      ctx_totals[run_ctx]++;
      extra_bits += 2.0 + std::log2(static_cast<double>(run_len));
    }
    run_len = 0;
  };

  // Subsample rows on very large images to keep evaluation fast (~2ms).
  const size_t y_step = (w * h > 524288) ? 2 : 1;
  const int32_t implicit_idx = static_cast<int32_t>(n);
  const bool simulate_groups = (w > 256 || h > 256);
  const size_t tile_dim = simulate_groups ? 256 : std::max(w, h);
  double compact_pal_bits = 0.0;

  std::vector<uint8_t> present(n + 1);
  std::vector<int32_t> local_pos(n + 1);

  for (size_t y0 = 0; y0 < h; y0 += tile_dim) {
    const size_t y1 = std::min(h, y0 + tile_dim);
    for (size_t x0 = 0; x0 < w; x0 += tile_dim) {
      const size_t x1 = std::min(w, x0 + tile_dim);
      const size_t tw = x1 - x0;
      const size_t th = y1 - y0;

      // Simulate per-group ChannelCompact (1D channel palette) that
      // enc_modular.cc applies to sparse 256x256 groups after a global palette.
      bool use_compact = false;
      if (simulate_groups && n > 16) {
        std::fill(present.begin(), present.end(), 0);
        int32_t min_v = static_cast<int32_t>(n);
        int32_t max_v = 0;
        for (size_t y = y0; y < y1; ++y) {
          const uint16_t* row = &idx_img[y * w];
          for (size_t x = x0; x < x1; ++x) {
            uint16_t raw = row[x];
            int32_t v = (raw < n) ? pos[raw] : implicit_idx;
            present[v] = 1;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
          }
        }
        int32_t span = max_v - min_v + 1;
        int32_t distinct = 0;
        for (int32_t v = min_v; v <= max_v; ++v) {
          if (present[v]) distinct++;
        }
        // Matches default channel_colors_percent (80%) and nb_pixels / 16 cap.
        if (distinct > 1 && distinct * 5 <= span * 4 &&
            static_cast<size_t>(distinct) <= (tw * th) / 16) {
          use_compact = true;
          int32_t idx = 0;
          int32_t prev_v = 0;
          for (int32_t v = min_v; v <= max_v; ++v) {
            if (present[v]) {
              local_pos[v] = idx++;
              int32_t gap = (idx == 1) ? v : (v - prev_v);
              compact_pal_bits +=
                  std::log2(static_cast<double>(gap) + 1.0) + 1.0;
              prev_v = v;
            }
          }
        }
      }

      for (size_t y = y0; y < y1; y += y_step) {
        const uint16_t* row = &idx_img[y * w];
        const uint16_t* prev_row = (y > y0) ? &idx_img[(y - 1) * w] : nullptr;
        int32_t left = 0;
        int32_t topleft = 0;
        uint32_t zero_run = 0;
        size_t run_ctx = 0;
        for (size_t x = x0; x < x1; ++x) {
          uint16_t raw = row[x];
          int32_t cur = (raw < n) ? pos[raw] : implicit_idx;
          if (use_compact) cur = local_pos[cur];
          int32_t top;
          if (prev_row) {
            uint16_t raw_top = prev_row[x];
            top = (raw_top < n) ? pos[raw_top] : implicit_idx;
            if (use_compact) top = local_pos[top];
          } else {
            top = left;
          }
          if (lz_mask[y * w + x]) {
            flush_zero_run(zero_run, run_ctx);
            topleft = top;
            left = cur;
            continue;
          }
          int32_t guess = (x == x0 && y == y0) ? 0
                          : (y == y0)          ? left
                          : (x == x0) ? top
                                      : ClampedGradient(top, left, topleft);
          uint32_t grad = static_cast<uint32_t>(std::abs(left - topleft) +
                                                std::abs(top - topleft));
          size_t ctx =
              (grad == 0)
                  ? 0
                  : std::min<size_t>(kNumCtx - 1, 1 + FloorLog2Nonzero(grad));
          int32_t res = cur - guess;
          if (res == 0) {
            if (zero_run == 0) run_ctx = ctx;
            zero_run++;
          } else {
            flush_zero_run(zero_run, run_ctx);
            push_token(ctx, res);
          }
          topleft = top;
          left = cur;
        }
        flush_zero_run(zero_run, run_ctx);
      }
    }
  }

  double img_bits = extra_bits;
  for (size_t c = 0; c < kNumCtx; ++c) {
    if (ctx_totals[c] == 0) continue;
    double inv_total = 1.0 / static_cast<double>(ctx_totals[c]);
    for (size_t t = 0; t < kMaxTokens; ++t) {
      if (counts[c][t] > 0) {
        double p = static_cast<double>(counts[c][t]) * inv_total;
        img_bits -= static_cast<double>(counts[c][t]) * std::log2(p);
      }
    }
  }
  img_bits = img_bits * static_cast<double>(y_step) + compact_pal_bits;

  // Add approximate bit cost of storing the palette meta-channel itself
  // (2D ClampedGradient over nb x n, matching modular meta-channel encoding).
  uint64_t pal_counts[kMaxTokens] = {};
  uint64_t pal_total = 0;
  double pal_extra = 0.0;
  for (size_t c = 0; c < nb; ++c) {
    int32_t left = 0;
    int32_t topleft = 0;
    for (size_t i = 0; i < n; ++i) {
      int32_t cur = palette[remapping[i]][c];
      int32_t top = (c > 0) ? palette[remapping[i]][c - 1] : left;
      int32_t guess = (i == 0 && c == 0) ? 0
                      : (c == 0)         ? left
                      : (i == 0)         ? top
                                         : ClampedGradient(top, left, topleft);
      int32_t res = cur - guess;
      uint32_t v =
          static_cast<uint32_t>(res >= 0 ? (res << 1) : ((-res << 1) - 1));
      uint32_t tok;
      if (v < 16) {
        tok = v;
      } else {
        uint32_t nbits = FloorLog2Nonzero(v);
        tok = 16 + ((nbits - 4) << 1) + ((v >> (nbits - 1)) & 1);
        if (tok >= kMaxTokens) tok = kMaxTokens - 1;
        pal_extra += static_cast<double>(nbits - 1);
      }
      pal_counts[tok]++;
      pal_total++;
      topleft = top;
      left = cur;
    }
  }
  double pal_bits = pal_extra;
  if (pal_total > 0) {
    double inv_total = 1.0 / static_cast<double>(pal_total);
    for (size_t t = 0; t < kMaxTokens; ++t) {
      if (pal_counts[t] > 0) {
        double p = static_cast<double>(pal_counts[t]) * inv_total;
        pal_bits -= static_cast<double>(pal_counts[t]) * std::log2(p);
      }
    }
  }
  return img_bits + pal_bits;
}

bool PaletteHasNonMonotonousDeltas(
    const std::vector<std::vector<pixel_type>>& palette, size_t nb) {
  if (palette.size() <= 2) return false;
  for (size_t c = 0; c < nb; ++c) {
    bool has_pos = false;
    bool has_neg = false;
    for (size_t i = 1; i < palette.size(); ++i) {
      int64_t d = static_cast<int64_t>(palette[i][c]) -
                  static_cast<int64_t>(palette[i - 1][c]);
      if (d > 0) has_pos = true;
      if (d < 0) has_neg = true;
      if (has_pos && has_neg) return true;
    }
  }
  return false;
}

void OrientRemapping(
    std::vector<size_t>& remapping,
    const std::vector<std::vector<pixel_type>>& palette,
    const std::map<std::vector<pixel_type>, size_t>& color_freq_map) {
  if (remapping.size() <= 2) return;
  size_t f_front = 0;
  size_t f_back = 0;
  auto it_f = color_freq_map.find(palette[remapping.front()]);
  if (it_f != color_freq_map.end()) f_front = it_f->second;
  auto it_b = color_freq_map.find(palette[remapping.back()]);
  if (it_b != color_freq_map.end()) f_back = it_b->second;
  if (f_back > f_front) {
    std::reverse(remapping.begin(), remapping.end());
  }
}

}  // namespace

int RoundInt(int value, int div) {  // symmetric rounding around 0
  if (value < 0) return -RoundInt(-value, div);
  return (value + div / 2) / div;
}

struct PaletteIterationData {
  static constexpr int kMaxDeltas = 128;
  bool final_run = false;
  std::vector<pixel_type> deltas[3];
  std::vector<double> delta_distances;
  std::vector<pixel_type> frequent_deltas[3];

  // Populates `frequent_deltas` with items from `deltas` based on frequencies
  // and color distances.
  void FindFrequentColorDeltas(int num_pixels, int bitdepth) {
    using pixel_type_3d = std::array<pixel_type, 3>;
    std::map<pixel_type_3d, double> delta_frequency_map;
    pixel_type bucket_size = 3 << std::max(0, bitdepth - 8);
    // Store frequency weighted by delta distance from quantized value.
    for (size_t i = 0; i < deltas[0].size(); ++i) {
      pixel_type_3d delta = {
          {RoundInt(deltas[0][i], bucket_size),
           RoundInt(deltas[1][i], bucket_size),
           RoundInt(deltas[2][i], bucket_size)}};  // a basic form of clustering
      if (delta[0] == 0 && delta[1] == 0 && delta[2] == 0) continue;
      delta_frequency_map[delta] += sqrt(sqrt(delta_distances[i]));
    }

    const float delta_distance_multiplier = 1.0f / num_pixels;

    // Weigh frequencies by magnitude and normalize.
    for (auto& delta_frequency : delta_frequency_map) {
      std::vector<pixel_type> current_delta = {delta_frequency.first[0],
                                               delta_frequency.first[1],
                                               delta_frequency.first[2]};
      float delta_distance =
          std::sqrt(palette_internal::ColorDistance({0, 0, 0}, current_delta)) +
          1;
      delta_frequency.second *=
          static_cast<double>(delta_distance) * delta_distance_multiplier;
    }

    // Sort by weighted frequency.
    using pixel_type_3d_frequency = std::pair<pixel_type_3d, double>;
    std::vector<pixel_type_3d_frequency> sorted_delta_frequency_map(
        delta_frequency_map.begin(), delta_frequency_map.end());
    std::sort(
        sorted_delta_frequency_map.begin(), sorted_delta_frequency_map.end(),
        [](const pixel_type_3d_frequency& a, const pixel_type_3d_frequency& b) {
          return a.second > b.second;
        });

    // Store the top deltas.
    for (auto& delta_frequency : sorted_delta_frequency_map) {
      if (frequent_deltas[0].size() >= kMaxDeltas) break;
      // Number obtained by optimizing on jyrki31 corpus:
      if (delta_frequency.second < 17) break;
      for (int c = 0; c < 3; ++c) {
        frequent_deltas[c].push_back(delta_frequency.first[c] * bucket_size);
      }
    }
  }
};

Status FwdPaletteIteration(Image& input, uint32_t begin_c, uint32_t end_c,
                           uint32_t& nb_colors, uint32_t& nb_deltas,
                           bool ordered, bool lossy, Predictor& predictor,
                           const weighted::Header& wp_header,
                           PaletteIterationData& palette_iteration_data,
                           bool zero_predictor_mode = false,
                           int palette_effort = 2) {
  JXL_QUIET_RETURN_IF_ERROR(CheckEqualChannels(input, begin_c, end_c));
  JXL_ENSURE(begin_c >= input.nb_meta_channels);
  JxlMemoryManager* memory_manager = input.memory_manager();
  uint32_t nb = end_c - begin_c + 1;

  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;
  if (input.bitdepth >= 32) return false;
  if (!lossy && nb_colors < 2) return false;

  if (!lossy && nb == 1) {
    // Channel palette special case
    if (nb_colors == 0) return false;
    std::vector<pixel_type> lookup;
    pixel_type minval;
    pixel_type maxval;
    compute_minmax(input.channel[begin_c], &minval, &maxval);
    size_t lookup_table_size =
        static_cast<int64_t>(maxval) - static_cast<int64_t>(minval) + 1;
    if (lookup_table_size > palette_internal::kMaxPaletteLookupTableSize) {
      // a lookup table would use too much memory, instead use a slower approach
      // with std::set
      std::set<pixel_type> chpalette;
      pixel_type idx = 0;
      for (size_t y = 0; y < h; y++) {
        const pixel_type* p = input.channel[begin_c].Row(y);
        for (size_t x = 0; x < w; x++) {
          const bool new_color = chpalette.insert(p[x]).second;
          if (new_color) {
            idx++;
            if (idx > static_cast<int>(nb_colors)) return false;
          }
        }
      }
      JXL_DEBUG_V(6, "Channel %i uses only %i colors.", begin_c, idx);
      JXL_ASSIGN_OR_RETURN(Channel pch,
                           Channel::Create(memory_manager, idx, 1));
      pch.hshift = -1;
      pch.vshift = -1;
      nb_colors = idx;
      idx = 0;
      pixel_type* JXL_RESTRICT p_palette = pch.Row(0);
      for (pixel_type p : chpalette) {
        p_palette[idx++] = p;
      }
      for (size_t y = 0; y < h; y++) {
        pixel_type* p = input.channel[begin_c].Row(y);
        for (size_t x = 0; x < w; x++) {
          for (idx = 0;
               p[x] != p_palette[idx] && idx < static_cast<int>(nb_colors);
               idx++) {
            // no-op
          }
          JXL_DASSERT(idx < static_cast<int>(nb_colors));
          p[x] = idx;
        }
      }
      predictor = Predictor::Zero;
      input.nb_meta_channels++;
      input.channel.insert(input.channel.begin(), std::move(pch));

      return true;
    }
    lookup.resize(lookup_table_size, 0);
    pixel_type idx = 0;
    for (size_t y = 0; y < h; y++) {
      const pixel_type* p = input.channel[begin_c].Row(y);
      for (size_t x = 0; x < w; x++) {
        if (lookup[p[x] - minval] == 0) {
          lookup[p[x] - minval] = 1;
          idx++;
          if (idx > static_cast<int>(nb_colors)) return false;
        }
      }
    }
    JXL_DEBUG_V(6, "Channel %i uses only %i colors.", begin_c, idx);
    JXL_ASSIGN_OR_RETURN(Channel pch, Channel::Create(memory_manager, idx, 1));
    pch.hshift = -1;
    pch.vshift = -1;
    nb_colors = idx;
    idx = 0;
    pixel_type* JXL_RESTRICT p_palette = pch.Row(0);
    for (size_t i = 0; i < lookup_table_size; i++) {
      if (lookup[i]) {
        p_palette[idx] = i + minval;
        lookup[i] = idx;
        idx++;
      }
    }
    for (size_t y = 0; y < h; y++) {
      pixel_type* p = input.channel[begin_c].Row(y);
      for (size_t x = 0; x < w; x++) p[x] = lookup[p[x] - minval];
    }
    predictor = Predictor::Zero;
    input.nb_meta_channels++;
    input.channel.insert(input.channel.begin(), std::move(pch));
    return true;
  }

  Image quantized_input(memory_manager);
  if (lossy) {
    JXL_ASSIGN_OR_RETURN(quantized_input, Image::Create(memory_manager, w, h,
                                                        input.bitdepth, nb));
    for (size_t c = 0; c < nb; c++) {
      JXL_RETURN_IF_ERROR(CopyImageTo(input.channel[begin_c + c].plane,
                                      &quantized_input.channel[c].plane));
    }
  }

  JXL_DEBUG_V(
      7, "Trying to represent channels %i-%i using at most a %i-color palette.",
      begin_c, end_c, nb_colors);
  nb_deltas = 0;
  bool delta_used = false;
  std::set<std::vector<pixel_type>> candidate_palette;
  std::vector<std::vector<pixel_type>> candidate_palette_imageorder;
  std::vector<pixel_type> color(nb);
  std::vector<float> color_with_error(nb);
  std::vector<const pixel_type*> p_in(nb);
  std::map<std::vector<pixel_type>, size_t> inv_palette;

  if (lossy) {
    palette_iteration_data.FindFrequentColorDeltas(w * h, input.bitdepth);
    nb_deltas = palette_iteration_data.frequent_deltas[0].size();

    // Count color frequency for colors that make a cross.
    std::map<std::vector<pixel_type>, size_t> color_freq_map;
    for (size_t y = 1; y + 1 < h; y++) {
      for (uint32_t c = 0; c < nb; c++) {
        p_in[c] = input.channel[begin_c + c].Row(y);
      }
      for (size_t x = 1; x + 1 < w; x++) {
        for (uint32_t c = 0; c < nb; c++) {
          color[c] = p_in[c][x];
        }
        int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        bool makes_cross = true;
        for (int i = 0; i < 4 && makes_cross; ++i) {
          int dx = offsets[i][0];
          int dy = offsets[i][1];
          for (uint32_t c = 0; c < nb && makes_cross; c++) {
            if (input.channel[begin_c + c].Row(y + dy)[x + dx] != color[c]) {
              makes_cross = false;
            }
          }
        }
        if (makes_cross) color_freq_map[color] += 1;
      }
    }
    // Add colors satisfying frequency condition to the palette.
    constexpr float kImageFraction = 0.01f;
    size_t color_frequency_lower_bound = 5 + input.h * input.w * kImageFraction;
    for (const auto& color_freq : color_freq_map) {
      if (color_freq.second > color_frequency_lower_bound) {
        candidate_palette.insert(color_freq.first);
        candidate_palette_imageorder.push_back(color_freq.first);
      }
    }
  }

  std::map<std::vector<pixel_type>, bool> implicit_color;
  std::vector<std::vector<pixel_type>> implicit_colors;
  implicit_colors.reserve(palette_internal::kImplicitPaletteSize);
  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    for (size_t i = 0; i < nb; i++) {
      color[i] = palette_internal::GetPaletteValue(nullptr, k, i, 0, 0,
                                                   input.bitdepth);
    }
    implicit_color[color] = true;
    implicit_colors.push_back(color);
  }

  std::map<std::vector<pixel_type>, size_t> color_freq_map;
  uint32_t implicit_colors_used = 0;
  for (size_t y = 0; y < h; y++) {
    for (uint32_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
    }
    for (size_t x = 0; x < w; x++) {
      if (lossy && candidate_palette.size() >= nb_colors) break;
      for (uint32_t c = 0; c < nb; c++) {
        color[c] = p_in[c][x];
      }
      const size_t count = ++color_freq_map[color];
      // candidate_palette is unused in lossless mode
      // detect new colors via count (save one lookup per pixel)
      const bool new_color =
          lossy ? candidate_palette.insert(color).second : count == 1;
      if (new_color) {
        if (implicit_color[color]) {
          implicit_colors_used++;
        } else {
          candidate_palette_imageorder.push_back(color);
          if (candidate_palette_imageorder.size() > nb_colors) {
            return false;  // too many colors
          }
        }
      }
    }
  }

  nb_colors = nb_deltas + candidate_palette_imageorder.size();

  // not useful to make a single-color palette
  if (!lossy && nb_colors + implicit_colors_used == 1) return false;
  // TODO(jon): if this happens (e.g. solid white group), special-case it for
  // faster encode

  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    color = implicit_colors[k];
    // still add the color to the explicit palette if it is frequent enough
    if (color_freq_map[color] > 10) {
      nb_colors++;
      candidate_palette_imageorder.push_back(color);
    }
  }
  for (size_t k = 0; k < palette_internal::kImplicitPaletteSize; k++) {
    color = implicit_colors[k];
    inv_palette[color] = nb_colors + k;
  }

  JXL_DEBUG_V(6, "Channels %i-%i can be represented using a %i-color palette.",
              begin_c, end_c, nb_colors);

  JXL_ASSIGN_OR_RETURN(Channel pch,
                       Channel::Create(memory_manager, nb_colors, nb));
  pch.hshift = -1;
  pch.vshift = -1;
  pixel_type* JXL_RESTRICT p_palette = pch.Row(0);
  ptrdiff_t onerow = pch.plane.PixelsPerRow();
  ptrdiff_t onerow_image = input.channel[begin_c].plane.PixelsPerRow();
  const int bit_depth = std::min(input.bitdepth, 24);

  if (lossy) {
    for (uint32_t i = 0; i < nb_deltas; i++) {
      for (size_t c = 0; c < 3; c++) {
        p_palette[c * onerow + i] =
            palette_iteration_data.frequent_deltas[c][i];
      }
    }
  }
  int clr = 0;
  if (ordered && !candidate_palette_imageorder.empty()) {
    if (zero_predictor_mode || palette_effort == 0 ||
        candidate_palette_imageorder.size() > 1024) {
      JXL_DEBUG_V(7,
                  "Palette of %i colors, using lexicographical/delta order "
                  "(zero predictor / fast / large palette mode)",
                  nb_colors);
      std::sort(candidate_palette_imageorder.begin(),
                candidate_palette_imageorder.end());
      const size_t n_pal = candidate_palette_imageorder.size();
      if (zero_predictor_mode && n_pal > 2) {
        size_t max_freq = 0;
        size_t max_freq_idx = 0;
        for (size_t i = 0; i < n_pal; ++i) {
          size_t f = color_freq_map[candidate_palette_imageorder[i]];
          if (f > max_freq) {
            max_freq = f;
            max_freq_idx = i;
          }
        }
        if (max_freq_idx > n_pal / 2) {
          std::reverse(candidate_palette_imageorder.begin(),
                       candidate_palette_imageorder.end());
        }
        const size_t dom_thresh = (w * h > 65536) ? 64 : (w * h) / 4;
        std::stable_sort(candidate_palette_imageorder.begin(),
                         candidate_palette_imageorder.end(),
                         [&](const std::vector<pixel_type>& a,
                             const std::vector<pixel_type>& b) {
                           size_t fa = color_freq_map[a];
                           size_t fb = color_freq_map[b];
                           bool dom_a = (fa >= dom_thresh);
                           bool dom_b = (fb >= dom_thresh);
                           if (dom_a != dom_b) return dom_a > dom_b;
                           if (dom_a && dom_b) return fa > fb;
                           return false;
                         });
      } else if (!zero_predictor_mode &&
                 (n_pal <= 1024 || palette_effort >= 1) &&
                 PaletteHasNonMonotonousDeltas(candidate_palette_imageorder,
                                               nb)) {
        auto rgb_dist = [&](const std::vector<pixel_type>& a,
                            const std::vector<pixel_type>& b) -> int64_t {
          int64_t d = 0;
          for (size_t c = 0; c < nb; ++c) {
            d += std::abs(static_cast<int64_t>(a[c]) -
                          static_cast<int64_t>(b[c]));
          }
          return d;
        };
        for (size_t i = 1; i < n_pal; ++i) {
          size_t best_ix = i;
          int64_t best_score = std::numeric_limits<int64_t>::max();
          const auto& prev = candidate_palette_imageorder[i - 1];
          for (size_t k = i; k < n_pal; ++k) {
            int64_t score = rgb_dist(prev, candidate_palette_imageorder[k]);
            if (score < best_score) {
              best_score = score;
              best_ix = k;
            }
          }
          if (best_ix != i) {
            auto chosen = std::move(candidate_palette_imageorder[best_ix]);
            for (size_t k = best_ix; k > i; --k) {
              candidate_palette_imageorder[k] =
                  std::move(candidate_palette_imageorder[k - 1]);
            }
            candidate_palette_imageorder[i] = std::move(chosen);
          }
        }
        if (n_pal > 2) {
          size_t f_front = color_freq_map[candidate_palette_imageorder.front()];
          size_t f_back = color_freq_map[candidate_palette_imageorder.back()];
          if (f_back > f_front) {
            std::reverse(candidate_palette_imageorder.begin(),
                         candidate_palette_imageorder.end());
          }
        }
      }
    } else if (palette_effort == 1) {
      // Medium-effort path (efforts 4-6): EZeng + light pairwise swap without
      // 2-opt or multi-candidate pixel-level cost evaluation.
      size_t num_colors = candidate_palette_imageorder.size();
      FastColorMap color_map;
      color_map.Init(candidate_palette_imageorder);

      std::vector<std::vector<uint32_t>> matrix(
          num_colors, std::vector<uint32_t>(num_colors, 0));
      std::vector<int> prev_row(w, -1);

      for (size_t y = 0; y < h; y++) {
        for (uint32_t c = 0; c < nb; c++) {
          p_in[c] = input.channel[begin_c + c].Row(y);
        }
        int prev_val = -1;
        for (size_t x = 0; x < w; x++) {
          for (uint32_t c = 0; c < nb; c++) {
            color[c] = p_in[c][x];
          }
          int val = color_map.Find(color);
          if (val != -1) {
            if (prev_val != -1) {
              matrix[prev_val][val]++;
            }
            if (prev_row[x] != -1) {
              matrix[prev_row[x]][val]++;
            }
            prev_val = val;
            prev_row[x] = val;
          } else {
            prev_val = -1;
            prev_row[x] = -1;
          }
        }
      }

      for (size_t i = 0; i < num_colors; i++) {
        for (size_t j = 0; j < num_colors; j++) {
          matrix[j][i] += matrix[i][j];
          matrix[i][j] = matrix[j][i];
        }
      }

      AdjList adj = BuildAdjList(matrix);
      std::vector<size_t> remapping = EZengReindex(matrix);
      PairwiseSwapSearch(remapping, adj, 20, 10);
      OrientRemapping(remapping, candidate_palette_imageorder, color_freq_map);

      std::vector<std::vector<pixel_type>> new_order(num_colors);
      for (size_t i = 0; i < num_colors; ++i) {
        new_order[i] = candidate_palette_imageorder[remapping[i]];
      }
      candidate_palette_imageorder = std::move(new_order);
    } else {
      size_t num_colors = candidate_palette_imageorder.size();
      FastColorMap color_map;
      color_map.Init(candidate_palette_imageorder);

      std::vector<std::vector<uint32_t>> matrix(
          num_colors, std::vector<uint32_t>(num_colors, 0));
      std::vector<uint16_t> idx_img(w * h, 0xFFFF);
      std::vector<int> prev_row(w, -1);

      for (size_t y = 0; y < h; y++) {
        for (uint32_t c = 0; c < nb; c++) {
          p_in[c] = input.channel[begin_c + c].Row(y);
        }
        int prev_val = -1;
        uint16_t* idx_row = &idx_img[y * w];
        for (size_t x = 0; x < w; x++) {
          for (uint32_t c = 0; c < nb; c++) {
            color[c] = p_in[c][x];
          }
          int val = color_map.Find(color);
          if (val != -1) {
            idx_row[x] = static_cast<uint16_t>(val);
            if (prev_val != -1) {
              matrix[prev_val][val]++;
            }
            if (prev_row[x] != -1) {
              matrix[prev_row[x]][val]++;
            }
            prev_val = val;
            prev_row[x] = val;
          } else {
            prev_val = -1;
            prev_row[x] = -1;
          }
        }
      }

      for (size_t i = 0; i < num_colors; i++) {
        for (size_t j = 0; j < num_colors; j++) {
          matrix[j][i] += matrix[i][j];
          matrix[i][j] = matrix[j][i];
        }
      }

      // Candidate 1: Luma order (partitioned into common vs rare colors).
      std::vector<size_t> remapping_luma(num_colors);
      std::vector<size_t> freqs(num_colors, 0);
      for (size_t i = 0; i < num_colors; ++i) {
        remapping_luma[i] = i;
        auto it = color_freq_map.find(candidate_palette_imageorder[i]);
        if (it != color_freq_map.end()) freqs[i] = it->second;
      }
      if (nb >= 3) {
        std::sort(
            remapping_luma.begin(), remapping_luma.end(),
            [&](size_t ai, size_t bi) {
              const auto& ap = candidate_palette_imageorder[ai];
              const auto& bp = candidate_palette_imageorder[bi];
              float ay =
                  (0.299f * ap[0] + 0.587f * ap[1] + 0.114f * ap[2] + 0.1f);
              if (ap.size() > 3) ay *= 1.f + ap[3];
              float by =
                  (0.299f * bp[0] + 0.587f * bp[1] + 0.114f * bp[2] + 0.1f);
              if (bp.size() > 3) by *= 1.f + bp[3];
              ay = freqs[ai] > 4 ? -ay : ay;
              by = freqs[bi] > 4 ? -by : by;
              if (std::abs(ay - by) > 1e-4f) return ay < by;
              float acb = -0.1687f * ap[0] - 0.3313f * ap[1] + 0.5f * ap[2];
              float bcb = -0.1687f * bp[0] - 0.3313f * bp[1] + 0.5f * bp[2];
              return acb < bcb;
            });
      }

      // Candidate 2: WebP MinimizeDeltas (greedy nearest-neighbor RGB L1).
      std::vector<size_t> remapping_delta(num_colors);
      for (size_t i = 0; i < num_colors; ++i) remapping_delta[i] = i;
      std::sort(remapping_delta.begin(), remapping_delta.end(),
                [&](size_t ai, size_t bi) {
                  return candidate_palette_imageorder[ai] <
                         candidate_palette_imageorder[bi];
                });
      auto rgb_dist_idx = [&](size_t ai, size_t bi) -> int64_t {
        const auto& a = candidate_palette_imageorder[ai];
        const auto& b = candidate_palette_imageorder[bi];
        int64_t d = 0;
        for (size_t c = 0; c < nb; ++c) {
          d +=
              std::abs(static_cast<int64_t>(a[c]) - static_cast<int64_t>(b[c]));
        }
        return d;
      };
      for (size_t i = 1; i < num_colors; ++i) {
        size_t best_ix = i;
        int64_t best_score = std::numeric_limits<int64_t>::max();
        size_t prev = remapping_delta[i - 1];
        for (size_t k = i; k < num_colors; ++k) {
          int64_t score = rgb_dist_idx(prev, remapping_delta[k]);
          if (score < best_score) {
            best_score = score;
            best_ix = k;
          }
        }
        if (best_ix != i) {
          size_t chosen = remapping_delta[best_ix];
          for (size_t k = best_ix; k > i; --k) {
            remapping_delta[k] = remapping_delta[k - 1];
          }
          remapping_delta[i] = chosen;
        }
      }
      OrientRemapping(remapping_delta, candidate_palette_imageorder,
                      color_freq_map);

      // Candidate 3: EZeng + PairwiseSwap + TwoOpt on co-occurrence graph.
      AdjList adj = BuildAdjList(matrix);
      std::vector<size_t> remapping_ezeng = EZengReindex(matrix);
      PairwiseSwapSearch(remapping_ezeng, adj, 50);
      TwoOptSearch(remapping_ezeng, adj);
      OrientRemapping(remapping_ezeng, candidate_palette_imageorder,
                      color_freq_map);

      // Pick the ordering with the lowest estimated modular bit cost
      // (2D LZ77-masked, per-group ChannelCompact-aware ClampedGradient
      // entropy + 2D palette meta-channel entropy).
      std::vector<uint8_t> lz_mask = ComputeLZ77MatchMask(idx_img, w, h);
      double cost_luma = EvaluatePaletteOrderCost(idx_img, lz_mask, w, h,
                                                  candidate_palette_imageorder,
                                                  remapping_luma, nb);
      double cost_delta = EvaluatePaletteOrderCost(idx_img, lz_mask, w, h,
                                                   candidate_palette_imageorder,
                                                   remapping_delta, nb);
      double cost_ezeng = EvaluatePaletteOrderCost(idx_img, lz_mask, w, h,
                                                   candidate_palette_imageorder,
                                                   remapping_ezeng, nb);

      const std::vector<size_t>* best_remapping = &remapping_ezeng;
      double best_cost = cost_ezeng;
      if (cost_luma < best_cost) {
        best_cost = cost_luma;
        best_remapping = &remapping_luma;
      }
      if (cost_delta < best_cost) {
        best_cost = cost_delta;
        best_remapping = &remapping_delta;
      }

      std::vector<std::vector<pixel_type>> new_order(num_colors);
      for (size_t i = 0; i < num_colors; ++i) {
        new_order[i] = candidate_palette_imageorder[(*best_remapping)[i]];
      }
      candidate_palette_imageorder = std::move(new_order);
    }
  } else {
    JXL_DEBUG_V(7, "Palette of %i colors, using image order", nb_colors);
  }

  for (auto pcol : candidate_palette_imageorder) {
    JXL_DEBUG_V(9, "  Color %i :  ", clr);
    for (size_t i = 0; i < nb; i++) {
      p_palette[nb_deltas + i * onerow + clr] = pcol[i];
      JXL_DEBUG_V(9, "%i ", pcol[i]);
    }
    inv_palette[pcol] = clr;
    clr++;
  }
  std::vector<weighted::State> wp_states;
  for (size_t c = 0; c < nb; c++) {
    wp_states.emplace_back(wp_header, w, h);
  }
  std::vector<pixel_type*> p_quant(nb);
  // Three rows of error for dithering: y to y + 2.
  // Each row has two pixels of padding in the ends, which is
  // beneficial for both precision and encoding speed.
  std::vector<std::vector<float>> error_row[3];
  if (lossy) {
    for (auto& row : error_row) {
      row.resize(nb);
      for (size_t c = 0; c < nb; ++c) {
        row[c].resize(w + 4);
      }
    }
  }
  for (size_t y = 0; y < h; y++) {
    for (size_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
      if (lossy) p_quant[c] = quantized_input.channel[c].Row(y);
    }
    pixel_type* JXL_RESTRICT p = input.channel[begin_c].Row(y);
    for (size_t x = 0; x < w; x++) {
      int index;
      if (!lossy) {
        for (size_t c = 0; c < nb; c++) color[c] = p_in[c][x];
        index = inv_palette[color];
      } else {
        int best_index = 0;
        bool best_is_delta = false;
        float best_distance = std::numeric_limits<float>::infinity();
        std::vector<pixel_type> best_val(nb, 0);
        std::vector<pixel_type> ideal_residual(nb, 0);
        std::vector<pixel_type> quantized_val(nb);
        std::vector<pixel_type> predictions(nb);
        for (double diffusion_multiplier : {0.55, 0.75}) {
          for (size_t c = 0; c < nb; c++) {
            color_with_error[c] =
                p_in[c][x] + (palette_iteration_data.final_run ? 1 : 0) *
                                 diffusion_multiplier * error_row[0][c][x + 2];
            color[c] = Clamp1(lround(color_with_error[c]), 0l,
                              (1l << input.bitdepth) - 1);
          }

          for (size_t c = 0; c < nb; ++c) {
            predictions[c] = PredictNoTreeWP(w, p_quant[c] + x, onerow_image, x,
                                             y, predictor, &wp_states[c])
                                 .guess;
          }
          const auto TryIndex = [&](const int index) {
            for (size_t c = 0; c < nb; c++) {
              quantized_val[c] = palette_internal::GetPaletteValue(
                  p_palette, index, /*c=*/c,
                  /*palette_size=*/nb_colors,
                  /*onerow=*/onerow, /*bit_depth=*/bit_depth);
              if (index < static_cast<int>(nb_deltas)) {
                quantized_val[c] += predictions[c];
              }
            }
            const float color_distance =
                32.0 / (1LL << std::max(0, 2 * (bit_depth - 8))) *
                palette_internal::ColorDistance(color_with_error,
                                                quantized_val);
            float index_penalty = 0;
            if (index == -1) {
              index_penalty = -124;
            } else if (index < 0) {
              index_penalty = -2 * index;
            } else if (index < static_cast<int>(nb_deltas)) {
              index_penalty = 250;
            } else if (index < static_cast<int>(nb_colors)) {
              index_penalty = 150;
            } else if (index < static_cast<int>(nb_colors) +
                                   palette_internal::kLargeCubeOffset) {
              index_penalty = 70;
            } else {
              index_penalty = 256;
            }
            const float distance = color_distance + index_penalty;
            if (distance < best_distance) {
              best_distance = distance;
              best_index = index;
              best_is_delta = index < static_cast<int>(nb_deltas);
              best_val.swap(quantized_val);
              for (size_t c = 0; c < nb; ++c) {
                ideal_residual[c] = color_with_error[c] - predictions[c];
              }
            }
          };
          for (index = palette_internal::kMinImplicitPaletteIndex;
               index < static_cast<int32_t>(nb_colors); index++) {
            TryIndex(index);
          }
          TryIndex(palette_internal::QuantizeColorToImplicitPaletteIndex(
              color, nb_colors, bit_depth,
              /*high_quality=*/false));
          if (palette_internal::kEncodeToHighQualityImplicitPalette) {
            TryIndex(palette_internal::QuantizeColorToImplicitPaletteIndex(
                color, nb_colors, bit_depth,
                /*high_quality=*/true));
          }
        }
        index = best_index;
        delta_used |= best_is_delta;
        if (!palette_iteration_data.final_run) {
          for (size_t c = 0; c < 3; ++c) {
            palette_iteration_data.deltas[c].push_back(ideal_residual[c]);
          }
          palette_iteration_data.delta_distances.push_back(best_distance);
        }

        for (size_t c = 0; c < nb; ++c) {
          wp_states[c].UpdateErrors(best_val[c], x, y, w);
          p_quant[c][x] = best_val[c];
        }
        float len_error = 0;
        for (size_t c = 0; c < nb; ++c) {
          float local_error = color_with_error[c] - best_val[c];
          len_error += local_error * local_error;
        }
        len_error = std::sqrt(len_error);
        float modulate = 1.0;
        int len_limit = 38 << std::max(0, bit_depth - 8);
        if (len_error > len_limit) {
          modulate *= len_limit / len_error;
        }
        for (size_t c = 0; c < nb; ++c) {
          float total_error = (color_with_error[c] - best_val[c]);

          // If the neighboring pixels have some error in the opposite
          // direction of total_error, cancel some or all of it out before
          // spreading among them.
          constexpr int offsets[12][2] = {{1, 2}, {0, 3}, {0, 4}, {1, 1},
                                          {1, 3}, {2, 2}, {1, 0}, {1, 4},
                                          {2, 1}, {2, 3}, {2, 0}, {2, 4}};
          float total_available = 0;
          for (int i = 0; i < 11; ++i) {
            const int row = offsets[i][0];
            const int col = offsets[i][1];
            if (std::signbit(error_row[row][c][x + col]) !=
                std::signbit(total_error)) {
              total_available += error_row[row][c][x + col];
            }
          }
          float weight =
              std::abs(total_error) / (std::abs(total_available) + 1e-3);
          weight = std::min(weight, 1.0f);
          for (int i = 0; i < 11; ++i) {
            const int row = offsets[i][0];
            const int col = offsets[i][1];
            if (std::signbit(error_row[row][c][x + col]) !=
                std::signbit(total_error)) {
              total_error += weight * error_row[row][c][x + col];
              error_row[row][c][x + col] *= (1 - weight);
            }
          }
          total_error *= modulate;
          const float remaining_error = (1.0f / 14.) * total_error;
          error_row[0][c][x + 3] += 2 * remaining_error;
          error_row[0][c][x + 4] += remaining_error;
          error_row[1][c][x + 0] += remaining_error;
          for (int i = 0; i < 5; ++i) {
            error_row[1][c][x + i] += remaining_error;
            error_row[2][c][x + i] += remaining_error;
          }
        }
      }
      if (palette_iteration_data.final_run) p[x] = index;
    }
    if (lossy) {
      for (size_t c = 0; c < nb; ++c) {
        error_row[0][c].swap(error_row[1][c]);
        error_row[1][c].swap(error_row[2][c]);
        std::fill(error_row[2][c].begin(), error_row[2][c].end(), 0.f);
      }
    }
  }
  if (!delta_used) {
    predictor = Predictor::Zero;
  }
  if (palette_iteration_data.final_run) {
    input.nb_meta_channels++;
    input.channel.erase(input.channel.begin() + begin_c + 1,
                        input.channel.begin() + end_c + 1);
    input.channel.insert(input.channel.begin(), std::move(pch));
  }
  nb_colors -= nb_deltas;
  return true;
}

Status FwdPalette(Image& input, uint32_t begin_c, uint32_t end_c,
                  uint32_t& nb_colors, uint32_t& nb_deltas, bool ordered,
                  bool lossy, Predictor& predictor,
                  const weighted::Header& wp_header, bool zero_predictor_mode,
                  int palette_effort) {
  PaletteIterationData palette_iteration_data;
  uint32_t nb_colors_orig = nb_colors;
  uint32_t nb_deltas_orig = nb_deltas;
  // preprocessing pass in case of lossy palette
  if (lossy && input.bitdepth >= 8) {
    JXL_RETURN_IF_ERROR(FwdPaletteIteration(
        input, begin_c, end_c, nb_colors_orig, nb_deltas_orig, ordered, lossy,
        predictor, wp_header, palette_iteration_data, zero_predictor_mode,
        palette_effort));
  }
  palette_iteration_data.final_run = true;
  return FwdPaletteIteration(
      input, begin_c, end_c, nb_colors, nb_deltas, ordered, lossy, predictor,
      wp_header, palette_iteration_data, zero_predictor_mode, palette_effort);
}

}  // namespace jxl
