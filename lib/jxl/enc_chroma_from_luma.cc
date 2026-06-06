// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_chroma_from_luma.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <limits>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quant_weights.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_chroma_from_luma.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/simd_util.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;

static HWY_FULL(float) df;
struct WeightProfile {
  float w[64];
  float b_weight[64];
  WeightProfile() {
    float max_r = std::sqrt(7.0f * 7.0f + 7.0f * 7.0f);
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        if (x == 0 && y == 0) {
          w[0] = 0.0f;
          b_weight[0] = 1.0f;
        } else {
          float r = std::sqrt(static_cast<float>(x * x + y * y));
          w[y * 8 + x] = 1.0f + 2.0f * (1.0f - r / max_r);
          b_weight[y * 8 + x] = 1.0f - 0.5f * (r / max_r);
        }
      }
    }
  }
};
static const WeightProfile kWeightProfile;



Status InitDCStorage(JxlMemoryManager* memory_manager, size_t num_blocks,
                     ImageF* dc_values) {
  // First row: Y channel
  // Second row: X channel
  // Third row: Y channel
  // Fourth row: B channel
  JXL_ASSIGN_OR_RETURN(
      *dc_values,
      ImageF::Create(memory_manager, RoundUpTo(num_blocks, Lanes(df)), 4));

  JXL_ENSURE(dc_values->xsize() != 0);
  // Zero-fill the last lanes
  for (size_t y = 0; y < 4; y++) {
    for (size_t x = dc_values->xsize() - Lanes(df); x < dc_values->xsize();
         x++) {
      dc_values->Row(y)[x] = 0;
    }
  }
  return true;
}

Status ComputeTile(const Image3F& opsin, const Rect& opsin_rect,
                   const DequantMatrices& dequant,
                   const AcStrategyImage* ac_strategy,
                   const ImageI* raw_quant_field, const Quantizer* quantizer,
                   const CompressParams& cparams, const Rect& rect, bool fast,
                   bool use_dct8, ImageSB* map_x, ImageSB* map_b,
                   ImageF* dc_values, Span<float> mem) {
  static_assert(kEncTileDimInBlocks == kColorTileDimInBlocks,
                "Invalid color tile dim");
  size_t xsize_blocks = opsin_rect.xsize() / kBlockDim;
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;

  const size_t y0 = rect.y0();
  const size_t x0 = rect.x0();
  const size_t x1 = rect.x0() + rect.xsize();
  const size_t y1 = rect.y0() + rect.ysize();

  int ty = y0 / kColorTileDimInBlocks;
  int tx = x0 / kColorTileDimInBlocks;

  int8_t* JXL_RESTRICT row_out_x = map_x->Row(ty);
  int8_t* JXL_RESTRICT row_out_b = map_b->Row(ty);

  float* JXL_RESTRICT dc_values_yx = dc_values->Row(0);
  float* JXL_RESTRICT dc_values_x = dc_values->Row(1);
  float* JXL_RESTRICT dc_values_yb = dc_values->Row(2);
  float* JXL_RESTRICT dc_values_b = dc_values->Row(3);

  // All are aligned.
  float* HWY_RESTRICT block_y = mem.begin();
  float* HWY_RESTRICT block_x = block_y + AcStrategy::kMaxCoeffArea;
  float* HWY_RESTRICT block_b = block_x + AcStrategy::kMaxCoeffArea;
  JXL_ENSURE(mem.remove_prefix(3 * AcStrategy::kMaxCoeffArea));
  float* HWY_RESTRICT coeffs_yx = mem.begin();
  float* HWY_RESTRICT coeffs_x = coeffs_yx + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_yb = coeffs_x + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_b = coeffs_yb + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_w = coeffs_b + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_w_b = coeffs_w + kColorTileDim * kColorTileDim;
  JXL_ENSURE(mem.remove_prefix(6 * kColorTileDim * kColorTileDim));
  constexpr size_t dc_size =
      AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks;
  float* HWY_RESTRICT dc_y = mem.begin();
  float* HWY_RESTRICT dc_x = dc_y + dc_size;
  float* HWY_RESTRICT dc_b = dc_x + dc_size;
  JXL_ENSURE(mem.remove_prefix(3 * dc_size));
  float* HWY_RESTRICT scratch_space = mem.begin();
  JXL_ENSURE(mem.size() == 2 * AcStrategy::kMaxCoeffArea + dct_scratch_size);

  size_t num_ac = 0;

  for (size_t y = y0; y < y1; ++y) {
    const float* JXL_RESTRICT row_y =
        opsin_rect.ConstPlaneRow(opsin, 1, y * kBlockDim);
    const float* JXL_RESTRICT row_x =
        opsin_rect.ConstPlaneRow(opsin, 0, y * kBlockDim);
    const float* JXL_RESTRICT row_b =
        opsin_rect.ConstPlaneRow(opsin, 2, y * kBlockDim);
    size_t stride = opsin.PixelsPerRow();

    for (size_t x = x0; x < x1; x++) {
      AcStrategy acs = use_dct8
                           ? AcStrategy::FromRawStrategy(AcStrategyType::DCT)
                           : ac_strategy->ConstRow(y)[x];
      if (!acs.IsFirstBlock()) continue;
      size_t xs = acs.covered_blocks_x();
      TransformFromPixels(acs.Strategy(), row_y + x * kBlockDim, stride,
                          block_y, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_y, dc_y, xs, scratch_space);
      TransformFromPixels(acs.Strategy(), row_x + x * kBlockDim, stride,
                          block_x, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_x, dc_x, xs, scratch_space);
      TransformFromPixels(acs.Strategy(), row_b + x * kBlockDim, stride,
                          block_b, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_b, dc_b, xs, scratch_space);
      const float* const JXL_RESTRICT qm_x =
          dequant.InvMatrix(acs.Strategy(), 0);
      const float* const JXL_RESTRICT qm_b =
          dequant.InvMatrix(acs.Strategy(), 2);
      float q_dc_x = use_dct8 ? 1 : 1.0f / quantizer->GetInvDcStep(0);
      float q_dc_b = use_dct8 ? 1 : 1.0f / quantizer->GetInvDcStep(2);

      // Copy DCs in dc_values.
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < xs; ix++) {
          dc_values_yx[(iy + y) * xsize_blocks + ix + x] =
              dc_y[iy * xs + ix] * q_dc_x;
          dc_values_x[(iy + y) * xsize_blocks + ix + x] =
              dc_x[iy * xs + ix] * q_dc_x;
          dc_values_yb[(iy + y) * xsize_blocks + ix + x] =
              dc_y[iy * xs + ix] * q_dc_b;
          dc_values_b[(iy + y) * xsize_blocks + ix + x] =
              dc_b[iy * xs + ix] * q_dc_b;
        }
      }

      // Do not use this block for computing AC CfL.
      if (acs.covered_blocks_x() + x0 > x1 ||
          acs.covered_blocks_y() + y0 > y1) {
        continue;
      }

      // Copy AC coefficients in the local block and compute per-coefficient
      // frequency weights. Coefficients are stored in raster order within the
      // CoefficientLayout-canonical block dimensions (cx >= cy).
      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      CoefficientLayout(&cy, &cx);
      // Zero out LFs. This introduces terms in the optimization loop that
      // don't affect the result, as they are all 0, but allow for simpler
      // SIMDfication.
      for (size_t iy = 0; iy < cy; iy++) {
        for (size_t ix = 0; ix < cx; ix++) {
          block_y[cx * kBlockDim * iy + ix] = 0;
          block_x[cx * kBlockDim * iy + ix] = 0;
          block_b[cx * kBlockDim * iy + ix] = 0;
        }
      }
      // Unclear why this is like it is. (This works slightly better
      // than the previous approach which was also a hack.)
      const float qq =
          (raw_quant_field == nullptr) ? 1.0f : raw_quant_field->Row(y)[x];
      // Experimentally values 128-130 seem best -- I don't know why we
      // need this multiplier.
      const float kStrangeMultiplier = 128;
      float q = use_dct8 ? 1 : quantizer->Scale() * kStrangeMultiplier * qq;
      const auto qv = Set(df, q);
      size_t sx = cx * 8;
      size_t sy = cy * 8;
      size_t shift_x = __builtin_ctz(cx);
      size_t shift_y = __builtin_ctz(cy);
      for (size_t iy = 0; iy < sy; ++iy) {
        size_t iy_norm = iy >> shift_y;
        for (size_t ix = 0; ix < sx; ++ix) {
          size_t ix_norm = ix >> shift_x;
          coeffs_w[num_ac + iy * sx + ix] = kWeightProfile.w[iy_norm * 8 + ix_norm];
          coeffs_w_b[num_ac + iy * sx + ix] = kWeightProfile.b_weight[iy_norm * 8 + ix_norm];
        }
      }
      
      for (size_t i = 0; i < cx * cy * 64; i += Lanes(df)) {
        const auto b_y = Load(df, block_y + i);
        const auto b_x = Load(df, block_x + i);
        const auto b_b = Load(df, block_b + i);
        const auto qqm_x = Mul(qv, Load(df, qm_x + i));
        const auto qqm_b = Mul(qv, Load(df, qm_b + i));
        Store(Mul(b_y, qqm_x), df, coeffs_yx + num_ac);
        Store(Mul(b_x, qqm_x), df, coeffs_x + num_ac);
        Store(Mul(b_y, qqm_b), df, coeffs_yb + num_ac);
        Store(Mul(b_b, qqm_b), df, coeffs_b + num_ac);

        num_ac += Lanes(df);
      }
    }
  }
  constexpr float kOversatPenaltyFactor = 1.2f;
  constexpr float kMultiplierBitCost = 0.1f;

  auto evaluate_candidate = [&](const float* m, const float* s,
                                float multiplier, float base, bool is_b) {
    const auto zero = Zero(df);
    const auto factor = Set(df, base + multiplier / kDefaultColorFactor);
    const auto oversat_penalty = Set(df, kOversatPenaltyFactor);
    const auto mul_v = Set(df, multiplier);

    auto total_cost_v = zero;

    for (size_t i = 0; i < num_ac; i += Lanes(df)) {
      const auto m_v = Load(df, m + i);
      const auto s_v = Load(df, s + i);
      const auto w_v = Load(df, coeffs_w + i);
      const auto b_w_v = is_b ? Load(df, coeffs_w_b + i) : Set(df, 1.0f);

      const auto res_v = Sub(s_v, Mul(factor, m_v));
      const auto abs_res_v = Abs(res_v);
      auto cost_v = Mul(Mul(w_v, b_w_v), abs_res_v);

      // Psycho-visual RDO heuristic: Oversaturation Penalty
      // If the multiplier causes the predicted chroma to overshoot the original
      // chroma such that the residual (C_orig - C) has the opposite sign of the
      // applied correlation, it means we are artificially boosting colors.
      // Oversaturated artifacts are visually jarring, so penalizing them
      // significantly improves perceptual metrics (SSIMULACRA2, Butteraugli).
      const auto is_oversat = Lt(Mul(mul_v, res_v), zero);
      cost_v = IfThenElse(is_oversat, Mul(cost_v, oversat_penalty), cost_v);

      total_cost_v = Add(total_cost_v, cost_v);
    }
    float total_cost = GetLane(SumOfLanes(df, total_cost_v));
    total_cost += std::abs(multiplier) * kMultiplierBitCost;
    return total_cost;
  };

  auto get_pred = [](const ImageSB* map, int tx, int ty) {
    if (tx == 0 && ty == 0) return 0;
    if (tx == 0) return (int)map->Row(ty - 1)[tx];
    if (ty == 0) return (int)map->Row(ty)[tx - 1];
    int left = map->Row(ty)[tx - 1];
    int top = map->Row(ty - 1)[tx];
    int topleft = map->Row(ty - 1)[tx - 1];
    int gradient = left + top - topleft;
    int min_val = std::min(left, top);
    int max_val = std::max(left, top);
    return jxl::Clamp1(gradient, min_val, max_val);
  };

  int pred_x = get_pred(map_x, tx, ty);
  int pred_b = get_pred(map_b, tx, ty);

  struct Candidate {
    int cand;
    float cost;
    bool operator<(const Candidate& other) const { return cost < other.cost; }
  };
  Candidate best_x_cands[256];
  Candidate best_b_cands[256];
  size_t num_x_cands = 0;
  size_t num_b_cands = 0;

  int step = (cparams.speed_tier > SpeedTier::kSquirrel || fast) ? 4 : 1;

  for (int cand = -128; cand <= 127; cand += step) {
    float cost_x = evaluate_candidate(coeffs_yx, coeffs_x, cand, 0.0f, false);
    best_x_cands[num_x_cands++] = {cand, cost_x};
    float cost_b = evaluate_candidate(coeffs_yb, coeffs_b, cand, jxl::cms::kYToBRatio, true);
    best_b_cands[num_b_cands++] = {cand, cost_b};
  }

  std::sort(best_x_cands, best_x_cands + num_x_cands);
  std::sort(best_b_cands, best_b_cands + num_b_cands);

  int top_n = std::min<int>(4, num_x_cands);
  
  if (step > 1) {
    int best_coarse_x = best_x_cands[0].cand;
    int best_coarse_b = best_b_cands[0].cand;
    for (int cand = best_coarse_x - step + 1; cand < best_coarse_x + step; ++cand) {
       if (cand < -128 || cand > 127 || cand == best_coarse_x) continue;
       best_x_cands[num_x_cands++] = {cand, evaluate_candidate(coeffs_yx, coeffs_x, cand, 0.0f, false)};
    }
    for (int cand = best_coarse_b - step + 1; cand < best_coarse_b + step; ++cand) {
       if (cand < -128 || cand > 127 || cand == best_coarse_b) continue;
       best_b_cands[num_b_cands++] = {cand, evaluate_candidate(coeffs_yb, coeffs_b, cand, jxl::cms::kYToBRatio, true)};
    }
    std::sort(best_x_cands, best_x_cands + num_x_cands);
    std::sort(best_b_cands, best_b_cands + num_b_cands);
  }

  auto evaluate_joint_candidate = [&](float mult_x, float mult_b) {
    const auto zero = Zero(df);
    const auto factor_x = Set(df, mult_x / kDefaultColorFactor);
    const auto factor_b = Set(df, jxl::cms::kYToBRatio + mult_b / kDefaultColorFactor);
    const auto oversat_penalty = Set(df, kOversatPenaltyFactor);
    const auto mul_v_x = Set(df, mult_x);
    const auto mul_v_b = Set(df, mult_b);

    auto total_cost_v = zero;

    for (size_t i = 0; i < num_ac; i += Lanes(df)) {
      const auto m_v_x = Load(df, coeffs_yx + i);
      const auto s_v_x = Load(df, coeffs_x + i);
      const auto m_v_b = Load(df, coeffs_yb + i);
      const auto s_v_b = Load(df, coeffs_b + i);
      const auto w_v = Load(df, coeffs_w + i);
      const auto b_w_v = Load(df, coeffs_w_b + i);

      const auto res_v_x = Sub(s_v_x, Mul(factor_x, m_v_x));
      const auto abs_res_v_x = Abs(res_v_x);
      
      const auto res_v_b = Sub(s_v_b, Mul(factor_b, m_v_b));
      const auto abs_res_v_b = Mul(b_w_v, Abs(res_v_b));

      // Fast L2 approximation: max + 0.4 * min
      const auto max_res = Max(abs_res_v_x, abs_res_v_b);
      const auto min_res = Min(abs_res_v_x, abs_res_v_b);
      auto cost_v = Mul(w_v, Add(max_res, Mul(Set(df, 0.4f), min_res)));

      const auto is_oversat_x = Lt(Mul(mul_v_x, res_v_x), zero);
      cost_v = IfThenElse(is_oversat_x, Mul(cost_v, oversat_penalty), cost_v);
      
      const auto is_oversat_b = Lt(Mul(mul_v_b, res_v_b), zero);
      cost_v = IfThenElse(is_oversat_b, Mul(cost_v, oversat_penalty), cost_v);

      total_cost_v = Add(total_cost_v, cost_v);
    }
    return GetLane(SumOfLanes(df, total_cost_v));
  };

  int32_t best_x = 0;
  int32_t best_b = 0;
  float best_joint_cost = std::numeric_limits<float>::max();

  constexpr float lambda_bits = 0.05f;

  for (int i = 0; i < top_n; ++i) {
    for (int j = 0; j < top_n; ++j) {
      int cand_x = best_x_cands[i].cand;
      int cand_b = best_b_cands[j].cand;

      float joint_cost = evaluate_joint_candidate(cand_x, cand_b);
      
      float bit_cost_x = lambda_bits * std::log2(1.0f + std::abs(cand_x - pred_x));
      float bit_cost_b = lambda_bits * std::log2(1.0f + std::abs(cand_b - pred_b));
      
      float total_cost = joint_cost + bit_cost_x + bit_cost_b + 
                         (std::abs(cand_x) + std::abs(cand_b)) * kMultiplierBitCost;

      if (total_cost < best_joint_cost) {
        best_joint_cost = total_cost;
        best_x = cand_x;
        best_b = cand_b;
      }
    }
  }

  row_out_x[tx] = best_x;
  row_out_b[tx] = best_b;
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(InitDCStorage);
HWY_EXPORT(ComputeTile);

Status CfLHeuristics::Init(const Rect& rect) {
  size_t xsize_blocks = rect.xsize() / kBlockDim;
  size_t ysize_blocks = rect.ysize() / kBlockDim;
  return HWY_DYNAMIC_DISPATCH(InitDCStorage)(
      memory_manager, xsize_blocks * ysize_blocks, &dc_values);
}

Status CfLHeuristics::ComputeTile(const Rect& r, const Image3F& opsin,
                                  const Rect& opsin_rect,
                                  const DequantMatrices& dequant,
                                  const AcStrategyImage* ac_strategy,
                                  const ImageI* raw_quant_field,
                                  const Quantizer* quantizer,
                                  const CompressParams& cparams, bool fast,
                                  size_t thread, ColorCorrelationMap* cmap) {
  bool use_dct8 = ac_strategy == nullptr;
  Span<float> scratch(mem.address<float>() + thread * ItemsPerThread(),
                      ItemsPerThread());
  return HWY_DYNAMIC_DISPATCH(ComputeTile)(
      opsin, opsin_rect, dequant, ac_strategy, raw_quant_field, quantizer,
      cparams, r, fast, use_dct8, &cmap->ytox_map, &cmap->ytob_map, &dc_values,
      scratch);
}

Status ColorCorrelationEncodeDC(const ColorCorrelation& color_correlation,
                                BitWriter* writer, LayerType layer,
                                AuxOut* aux_out) {
  float color_factor = color_correlation.GetColorFactor();
  float base_correlation_x = color_correlation.GetBaseCorrelationX();
  float base_correlation_b = color_correlation.GetBaseCorrelationB();
  int32_t ytox_dc = color_correlation.GetYToXDC();
  int32_t ytob_dc = color_correlation.GetYToBDC();

  return writer->WithMaxBits(
      1 + 2 * kBitsPerByte + 12 + 32, layer, aux_out, [&]() -> Status {
        if (ytox_dc == 0 && ytob_dc == 0 &&
            color_factor == kDefaultColorFactor && base_correlation_x == 0.0f &&
            base_correlation_b == jxl::cms::kYToBRatio) {
          writer->Write(1, 1);
          return true;
        }
        writer->Write(1, 0);
        JXL_RETURN_IF_ERROR(
            U32Coder::Write(kColorFactorDist, color_factor, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_x, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_b, writer));
        writer->Write(kBitsPerByte,
                      ytox_dc - std::numeric_limits<int8_t>::min());
        writer->Write(kBitsPerByte,
                      ytob_dc - std::numeric_limits<int8_t>::min());
        return true;
      });
}

}  // namespace jxl
#endif  // HWY_ONCE
