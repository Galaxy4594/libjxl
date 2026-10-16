// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_coeff_order.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/lehmer_code.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

std::pair<uint32_t, uint32_t> ComputeUsedOrders(
    const SpeedTier speed, const AcStrategyImage& ac_strategy,
    const Rect& rect) {
  // No coefficient reordering in Falcon or faster.
  // Only uses DCT8 = 0, so bitfield = 1.
  if (speed >= SpeedTier::kFalcon) return {1, 1};

  uint32_t ret = 0;
  uint32_t ret_customize = 0;
  size_t xsize_blocks = rect.xsize();
  size_t ysize_blocks = rect.ysize();
  // TODO(veluca): precompute when doing DCT.
  for (size_t by = 0; by < ysize_blocks; ++by) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      int ord = kStrategyOrder[acs_row[bx].RawStrategy()];
      // Do not customize coefficient orders for blocks bigger than 32x32.
      ret |= 1u << ord;
      if (ord > 6) {
        continue;
      }
      ret_customize |= 1u << ord;
    }
  }
  // Use default orders for small images.
  if (ac_strategy.xsize() < 5 && ac_strategy.ysize() < 5) return {ret, 0};
  return {ret, ret_customize};
}

Status FinalizeCoeffOrderFromNumZeros(
    JxlMemoryManager* memory_manager,
    const std::vector<int64_t>& num_zeros,
    uint32_t& all_used_orders, uint32_t prev_used_acs,
    uint32_t current_used_acs,
    uint32_t current_used_orders,
    coeff_order_t* JXL_RESTRICT order) {
  struct PosAndCount {
    uint32_t pos;
    // Saving index breaks the ties for non-stable sort
    uint64_t count_and_idx;
  };
  size_t mem_bytes = AcStrategy::kMaxCoeffArea * sizeof(PosAndCount);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));

  std::vector<coeff_order_t> natural_order_buffer;

  uint16_t computed = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    size_t sz = kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
    // Expected maximal size is 256 x 256.
    JXL_DASSERT(sz <= (1 << 16));

    // Do nothing for transforms that don't appear.
    if ((1 << ord) & ~current_used_acs) continue;

    // Do nothing if we already committed to this custom order previously.
    if ((1 << ord) & prev_used_acs) continue;
    if ((1 << ord) & all_used_orders) continue;

    if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
    acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());

    // Ensure natural coefficient order is not permuted if the order is
    // not transmitted.
    if ((1 << ord) & ~current_used_orders) {
      for (size_t c = 0; c < 3; c++) {
        size_t offset = CoeffOrderOffset(ord, c);
        JXL_ENSURE(CoeffOrderOffset(ord, c + 1) - offset == sz);
        memcpy(&order[offset], natural_order_buffer.data(),
               sz * sizeof(*order));
      }
      continue;
    }

    bool is_nondefault = false;
    for (uint8_t c = 0; c < 3; c++) {
      // Apply zig-zag order.
      PosAndCount* pos_and_val = mem.address<PosAndCount>();
      size_t offset = CoeffOrderOffset(ord, c);
      JXL_ENSURE(CoeffOrderOffset(ord, c + 1) - offset == sz);
      float inv_sqrt_sz = 1.0f / std::sqrt(sz);
      for (size_t i = 0; i < sz; ++i) {
        size_t pos = natural_order_buffer[i];
        pos_and_val[i].pos = pos;
        // We don't care for the exact number -> quantize number of zeros,
        // to get less permuted order.
        uint64_t count = num_zeros[offset + pos] * inv_sqrt_sz + 0.1f;
        // Worst case: all dct8x8, all zeroes: count <= nb_pixels/64/8
        // nb_pixels is limited to 2^40 (Level 10 limit)
        // so count is limited to 2^31
        JXL_DASSERT(count < (uint64_t{1} << 48));
        pos_and_val[i].count_and_idx = (count << 16) | i;
      }

      // Stable-sort -> elements with same number of zeros will preserve their
      // order.
      auto comparator = [](const PosAndCount& a, const PosAndCount& b) -> bool {
        return a.count_and_idx < b.count_and_idx;
      };
      std::sort(pos_and_val, pos_and_val + sz, comparator);

      // Grab indices.
      for (size_t i = 0; i < sz; ++i) {
        order[offset + i] = pos_and_val[i].pos;
        is_nondefault |= natural_order_buffer[i] != pos_and_val[i].pos;
      }
    }
    if (!is_nondefault) {
      current_used_orders &= ~(1 << ord);
    }
  }
  all_used_orders |= current_used_orders;
  return true;
}

Status ComputeCoeffOrder(SpeedTier speed, const ACImage& ac_image,
                         const AcStrategyImage& ac_strategy,
                         const FrameDimensions& frame_dim,
                         uint32_t& all_used_orders, uint32_t prev_used_acs,
                         uint32_t current_used_acs,
                         uint32_t current_used_orders,
                         coeff_order_t* JXL_RESTRICT order,
                         const YCbCrChromaSubsampling& cs) {
  JxlMemoryManager* memory_manager = ac_strategy.memory_manager();
  std::vector<int64_t> num_zeros(kCoeffOrderMaxSize);
  // If compressing at high speed and only using 8x8 DCTs, only consider a
  // subset of blocks.
  double block_fraction = 1.0f;
  // TODO(veluca): figure out why sampling blocks if non-8x8s are used makes
  // encoding significantly less dense.
  if (speed >= SpeedTier::kSquirrel && current_used_orders == 1) {
    block_fraction = 0.5f;
  }
  // No need to compute number of zero coefficients if all orders are the
  // default.
  if (current_used_orders != 0) {
    uint64_t threshold =
        (std::numeric_limits<uint64_t>::max() >> 32) * block_fraction;
    uint64_t s[2] = {static_cast<uint64_t>(0x94D049BB133111EBull),
                     static_cast<uint64_t>(0xBF58476D1CE4E5B9ull)};
    // Xorshift128+ adapted from xorshift128+-inl.h
    auto use_sample = [&]() {
      auto s1 = s[0];
      const auto s0 = s[1];
      const auto bits = s1 + s0;  // b, c
      s[0] = s0;
      s1 ^= s1 << 23;
      s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
      s[1] = s1;
      return (bits >> 32) <= threshold;
    };

    // Count number of zero coefficients, separately for each DCT band.
    // TODO(veluca): precompute when doing DCT.
    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         group_index++) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      const Rect rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                      kGroupDimInBlocks, kGroupDimInBlocks,
                      frame_dim.xsize_blocks, frame_dim.ysize_blocks);
      ConstACPtr rows[3];
      ACType type = ac_image.Type();
      for (size_t c = 0; c < 3; c++) {
        rows[c] = ac_image.PlaneRow(c, group_index, 0);
      }
      size_t ac_offset[3] = {};

      // TODO(veluca): SIMDfy.
      for (size_t by = 0; by < rect.ysize(); ++by) {
        AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
        for (size_t bx = 0; bx < rect.xsize(); ++bx) {
          AcStrategy acs = acs_row[bx];
          if (!acs.IsFirstBlock()) continue;
          bool sample = use_sample();
          size_t size = kDCTBlockSize << acs.log2_covered_blocks();
          for (size_t c = 0; c < 3; ++c) {
            if ((bx >> cs.HShift(c)) << cs.HShift(c) != bx) continue;
            if ((by >> cs.VShift(c)) << cs.VShift(c) != by) continue;
            if (sample) {
              const size_t order_offset =
                  CoeffOrderOffset(kStrategyOrder[acs.RawStrategy()], c);
              if (type == ACType::k16) {
                for (size_t k = 0; k < size; k++) {
                  bool is_zero = rows[c].ptr16[ac_offset[c] + k] == 0;
                  num_zeros[order_offset + k] += is_zero ? 1 : 0;
                }
              } else {
                for (size_t k = 0; k < size; k++) {
                  bool is_zero = rows[c].ptr32[ac_offset[c] + k] == 0;
                  num_zeros[order_offset + k] += is_zero ? 1 : 0;
                }
              }
              // Ensure LLFs are first in the order.
              size_t cx = acs.covered_blocks_x();
              size_t cy = acs.covered_blocks_y();
              CoefficientLayout(&cy, &cx);
              for (size_t iy = 0; iy < cy; iy++) {
                for (size_t ix = 0; ix < cx; ix++) {
                  num_zeros[order_offset + iy * kBlockDim * cx + ix] = -1;
                }
              }
            }
            ac_offset[c] += size;
          }
        }
      }
    }
  }
  return FinalizeCoeffOrderFromNumZeros(
      memory_manager, num_zeros, all_used_orders, prev_used_acs,
      current_used_acs, current_used_orders, order);
}

Status ComputeCoeffOrderJPEG(
    SpeedTier speed, const jpeg::JPEGData& jpeg_data,
    const FrameHeader& frame_header,
    const FrameDimensions& frame_dim,
    uint32_t& all_used_orders, uint32_t prev_used_acs,
    uint32_t current_used_acs,
    uint32_t current_used_orders,
    coeff_order_t* JXL_RESTRICT order,
    const ColorCorrelationMap* cmap,
    const std::vector<int32_t>& scaled_qtable,
    bool force_cfl, JxlMemoryManager* memory_manager) {
  std::vector<int64_t> num_zeros(kCoeffOrderMaxSize, 0);
  double block_fraction = 1.0f;
  if (speed >= SpeedTier::kSquirrel && current_used_orders == 1) {
    block_fraction = 0.5f;
  }
  if (current_used_orders != 0) {
    uint64_t threshold =
        (std::numeric_limits<uint64_t>::max() >> 32) * block_fraction;
    uint64_t s[2] = {static_cast<uint64_t>(0x94D049BB133111EBull),
                     static_cast<uint64_t>(0xBF58476D1CE4E5B9ull)};
    auto use_sample = [&]() {
      auto s1 = s[0];
      const auto s0 = s[1];
      const auto bits = s1 + s0;
      s[0] = s0;
      s1 ^= s1 << 23;
      s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
      s[1] = s1;
      return (bits >> 32) <= threshold;
    };

    auto jpeg_c_map =
        JpegOrder(frame_header.color_transform, jpeg_data.components.size() == 1);
    const YCbCrChromaSubsampling& cs = frame_header.chroma_subsampling;

    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         group_index++) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      const Rect rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                      kGroupDimInBlocks, kGroupDimInBlocks,
                      frame_dim.xsize_blocks, frame_dim.ysize_blocks);

      for (size_t by = 0; by < rect.ysize(); ++by) {
        for (size_t bx = 0; bx < rect.xsize(); ++bx) {
          bool sample = use_sample();
          for (size_t c = 0; c < 3; ++c) {
            if ((bx >> cs.HShift(c)) << cs.HShift(c) != bx) continue;
            if ((by >> cs.VShift(c)) << cs.VShift(c) != by) continue;
            if (sample) {
              const size_t order_offset = CoeffOrderOffset(0, c);
              if (jpeg_data.components.size() == 1 && c != 1) {
                for (size_t k = 0; k < 64; k++) {
                  num_zeros[order_offset + k]++;
                }
                num_zeros[order_offset] = -1;
                continue;
              }
              size_t block_x = (rect.x0() + bx) >> cs.HShift(c);
              size_t block_y = (rect.y0() + by) >> cs.VShift(c);
              size_t jpeg_c = jpeg_c_map[c];
              const auto& comp = jpeg_data.components[jpeg_c];
              const int16_t* inputjpeg =
                  comp.coeffs.data() +
                  (block_y * comp.width_in_blocks + block_x) * kDCTBlockSize;
              if (c == 1 || !force_cfl || !cs.Is444() || cmap == nullptr) {
                for (size_t y = 0; y < 8; y++) {
                  for (size_t x = 0; x < 8; x++) {
                    bool is_zero = (inputjpeg[y * 8 + x] == 0);
                    num_zeros[order_offset + x * 8 + y] += is_zero ? 1 : 0;
                  }
                }
              } else {
                size_t jpeg_y_c = jpeg_c_map[1];
                const auto& compY = jpeg_data.components[jpeg_y_c];
                const int16_t* inputjpegY =
                    compY.coeffs.data() +
                    (block_y * compY.width_in_blocks + block_x) * kDCTBlockSize;
                const ImageSB& map =
                    (c == 0 ? cmap->ytox_map : cmap->ytob_map);
                const int8_t* cm =
                    map.ConstRow((rect.y0() + by) / kColorTileDimInBlocks);
                const int32_t scale = ColorCorrelation::RatioJPEG(
                    cm[(rect.x0() + bx) / kColorTileDimInBlocks]);
                for (size_t y = 0; y < 8; y++) {
                  for (size_t x = 0; x < 8; x++) {
                    int coeffpos = y * 8 + x;
                    int Y = inputjpegY[coeffpos];
                    int QChroma = inputjpeg[coeffpos];
                    int coeff_scale =
                        (scale * scaled_qtable[kDCTBlockSize * c + coeffpos] +
                         (1 << (kCFLFixedPointPrecision - 1))) >>
                        kCFLFixedPointPrecision;
                    int cfl_factor =
                        (Y * coeff_scale +
                         (1 << (kCFLFixedPointPrecision - 1))) >>
                        kCFLFixedPointPrecision;
                    int QCR = QChroma - cfl_factor;
                    int16_t val =
                        static_cast<int16_t>(Clamp1(-32768, QCR, 32767));
                    bool is_zero = (val == 0);
                    num_zeros[order_offset + x * 8 + y] += is_zero ? 1 : 0;
                  }
                }
              }
              num_zeros[order_offset] = -1;
            }
          }
        }
      }
    }
  }

  return FinalizeCoeffOrderFromNumZeros(
      memory_manager, num_zeros, all_used_orders, prev_used_acs,
      current_used_acs, current_used_orders, order);
}

namespace {

Status TokenizePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                           size_t size, std::vector<Token>* tokens) {
  std::vector<LehmerT> lehmer(size);
  std::vector<uint32_t> temp(size + 1);
  JXL_RETURN_IF_ERROR(
      ComputeLehmerCode(order, temp.data(), size, lehmer.data()));
  size_t end = size;
  while (end > skip && lehmer[end - 1] == 0) {
    --end;
  }
  tokens->emplace_back(CoeffOrderContext(size),
                       static_cast<uint32_t>(end - skip));
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    tokens->emplace_back(CoeffOrderContext(last), lehmer[i]);
    last = lehmer[i];
  }
  return true;
}

}  // namespace

Status EncodePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                         size_t size, BitWriter* writer, LayerType layer,
                         AuxOut* aux_out) {
  JxlMemoryManager* memory_manager = writer->memory_manager();
  std::vector<std::vector<Token>> tokens(1);
  JXL_RETURN_IF_ERROR(TokenizePermutation(order, skip, size, tokens.data()));
  EntropyEncodingData codes;
  JXL_ASSIGN_OR_RETURN(
      size_t cost, BuildAndEncodeHistograms(memory_manager, HistogramParams(),
                                            kPermutationContexts, tokens,
                                            &codes, writer, layer, aux_out));
  (void)cost;
  JXL_RETURN_IF_ERROR(WriteTokens(tokens[0], codes, 0, writer, layer, aux_out));
  return true;
}

namespace {
Status EncodeCoeffOrder(const coeff_order_t* JXL_RESTRICT order, AcStrategy acs,
                        std::vector<Token>* tokens, coeff_order_t* order_zigzag,
                        std::vector<coeff_order_t>& natural_order_lut) {
  const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = kDCTBlockSize * llf;
  for (size_t i = 0; i < size; ++i) {
    order_zigzag[i] = natural_order_lut[order[i]];
  }
  JXL_RETURN_IF_ERROR(TokenizePermutation(order_zigzag, llf, size, tokens));
  return true;
}
}  // namespace

Status EncodeCoeffOrders(uint16_t used_orders,
                         const coeff_order_t* JXL_RESTRICT order,
                         BitWriter* writer, LayerType layer,
                         AuxOut* JXL_RESTRICT aux_out) {
  JxlMemoryManager* memory_manager = writer->memory_manager();
  size_t mem_bytes = AcStrategy::kMaxCoeffArea * sizeof(coeff_order_t);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));
  uint16_t computed = 0;
  std::vector<std::vector<Token>> tokens(1);
  std::vector<coeff_order_t> natural_order_lut;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    if ((used_orders & (1 << ord)) == 0) continue;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
    const size_t size = kDCTBlockSize * llf;
    if (natural_order_lut.size() < size) natural_order_lut.resize(size);
    acs.ComputeNaturalCoeffOrderLut(natural_order_lut.data());
    for (size_t c = 0; c < 3; c++) {
      JXL_RETURN_IF_ERROR(
          EncodeCoeffOrder(&order[CoeffOrderOffset(ord, c)], acs, tokens.data(),
                           mem.address<coeff_order_t>(), natural_order_lut));
    }
  }
  // Do not write anything if no order is used.
  if (used_orders != 0) {
    EntropyEncodingData codes;
    JXL_ASSIGN_OR_RETURN(
        size_t cost, BuildAndEncodeHistograms(memory_manager, HistogramParams(),
                                              kPermutationContexts, tokens,
                                              &codes, writer, layer, aux_out));
    (void)cost;
    JXL_RETURN_IF_ERROR(
        WriteTokens(tokens[0], codes, 0, writer, layer, aux_out));
  }
  return true;
}

}  // namespace jxl
