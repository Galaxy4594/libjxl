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
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::SumOfLanes;
using hwy::HWY_NAMESPACE::Zero;

static HWY_FULL(float) df;
// 2D radial Contrast Sensitivity Function (CSF) weights prioritizing low-frequency AC.
struct WeightProfile {
  float w[64];
  WeightProfile() {
    float max_r = std::sqrt(7.0f * 7.0f + 7.0f * 7.0f);
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        if (x == 0 && y == 0) {
          w[0] = 0.0f;
        } else {
          float r = std::sqrt(static_cast<float>(x * x + y * y));
          w[y * 8 + x] = 1.0f + 2.0f * (1.0f - r / max_r);
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
  JXL_ENSURE(mem.remove_prefix(5 * kColorTileDim * kColorTileDim));
  constexpr size_t dc_size =
      AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks;
  float* HWY_RESTRICT dc_y = mem.begin();
  float* HWY_RESTRICT dc_x = dc_y + dc_size;
  float* HWY_RESTRICT dc_b = dc_x + dc_size;
  JXL_ENSURE(mem.remove_prefix(3 * dc_size));
  float* HWY_RESTRICT scratch_space = mem.begin();
  JXL_ENSURE(mem.size() == 2 * AcStrategy::kMaxCoeffArea + dct_scratch_size);

  size_t num_ac = 0;
  float tile_q = 0.0f;
  int num_blocks = 0;

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
      float q = use_dct8 ? 1.0f : (quantizer->Scale() * qq);
      tile_q += q;
      num_blocks++;
      const auto qv = Set(df, q);
      size_t sx = cx * 8;
      size_t sy = cy * 8;
      size_t shift_x = __builtin_ctz(cx);
      size_t shift_y = __builtin_ctz(cy);
      for (size_t iy = 0; iy < sy; ++iy) {
        size_t iy_norm = iy >> shift_y;
        for (size_t ix = 0; ix < sx; ++ix) {
          size_t ix_norm = ix >> shift_x;
          coeffs_w[num_ac + iy * sx + ix] =
              kWeightProfile.w[iy_norm * 8 + ix_norm];
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
  constexpr float kMultiplierBitCost = 0.05f;

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

  tile_q = num_blocks > 0 ? tile_q / num_blocks : 1.0f;

  auto optimize_channel_simd = [&](const float* m, const float* s, int pred, float base) {
    if (num_ac == 0) {
      return (std::abs(pred) <= 4) ? pred : 0;
    }

    int step = 1;
    if (fast) {
      step = 8;
    } else if (cparams.speed_tier >= SpeedTier::kCheetah) {
      step = 8;
    } else if (cparams.speed_tier >= SpeedTier::kHare) {
      step = 4;
    } else if (cparams.speed_tier >= SpeedTier::kSquirrel) {
      step = 2;
    } else {
      step = 1;
    }

    size_t lanes = Lanes(df);
    
    // Vectorized correlation and energy analysis
    auto v_dot_ms = Zero(df);
    auto v_dot_mm = Zero(df);
    auto v_dot_ss = Zero(df);
    auto v_sum_s = Zero(df);
    for (size_t i = 0; i < num_ac; i += lanes) {
      const auto m_v = LoadU(df, m + i);
      const auto s_v = LoadU(df, s + i);
      const auto w_v = LoadU(df, coeffs_w + i);
      const auto wm_v = Mul(w_v, m_v);
      v_dot_ms = MulAdd(wm_v, s_v, v_dot_ms);
      v_dot_mm = MulAdd(wm_v, m_v, v_dot_mm);
      v_dot_ss = MulAdd(w_v, Mul(s_v, s_v), v_dot_ss);
      v_sum_s = Add(v_sum_s, Abs(s_v));
    }
    float dot_ms = GetLane(SumOfLanes(df, v_dot_ms));
    float dot_mm = GetLane(SumOfLanes(df, v_dot_mm));
    float dot_ss = GetLane(SumOfLanes(df, v_dot_ss));
    float sum_s = GetLane(SumOfLanes(df, v_sum_s));
    float energy = (sum_s / num_ac);

    // Analytical least-squares target multiplier: argmin_f sum w * (s - f*m)^2
    float target_factor = (dot_mm > 1e-6f) ? (dot_ms / dot_mm) : base;
    int target_cand = jxl::Clamp1(
        static_cast<int>(std::round((target_factor - base) * kDefaultColorFactor)),
        -128, 127);
    float corr = (dot_mm > 1e-6f && dot_ss > 1e-6f)
                     ? (std::abs(dot_ms) / std::sqrt(dot_mm * dot_ss))
                     : 0.0f;

    // SIMD candidate evaluation (outer: candidate vector, inner: AC coefficients).
    auto evaluate_simd = [&](const float* eval_cands, float* eval_costs, size_t num_eval) {
      const auto oversat_penalty = Set(df, kOversatPenaltyFactor);
      const auto inv_color_factor = Set(df, 1.0f / kDefaultColorFactor);
      const auto base_v = Set(df, base);

      for (size_t c = 0; c < num_eval; c += lanes) {
        const auto mul_v = LoadU(df, eval_cands + c);
        const auto factor_v = Add(base_v, Mul(mul_v, inv_color_factor));
        auto accum_v = Zero(df);

        for (size_t i = 0; i < num_ac; ++i) {
          const auto m_v = Set(df, m[i]);
          const auto s_v = Set(df, s[i]);
          const auto w_v = Set(df, coeffs_w[i]);
          const auto abs_s_v = Abs(s_v);

          const auto res_v = Sub(s_v, Mul(factor_v, m_v));
          const auto abs_res_v = Abs(res_v);
          auto cost_v = Mul(w_v, abs_res_v);

          // Penalize candidates that increase residual energy over unpredicted chroma.
          const auto is_oversat = Lt(abs_s_v, abs_res_v);
          cost_v = IfThenElse(is_oversat, Mul(cost_v, oversat_penalty), cost_v);

          accum_v = Add(accum_v, cost_v);
        }
        StoreU(accum_v, df, eval_costs + c);
      }
    };

    // Coarse candidate buffer setup.
    constexpr size_t kCandStride = 384;
    float* HWY_RESTRICT cands = scratch_space;
    float* HWY_RESTRICT costs = cands + kCandStride;
    float* HWY_RESTRICT f_cands = costs + kCandStride;
    float* HWY_RESTRICT f_costs = f_cands + kCandStride;
    memset(costs, 0, kCandStride * sizeof(float));

    int true_num_cands = 0;
    for (int cand = -128; cand <= 127; cand += step) {
      cands[true_num_cands++] = static_cast<float>(cand);
    }
    
    // Seed candidate set with 0, spatial predictor, LS target, and a local neighborhood.
    auto add_cand = [&](int cand) {
      cand = jxl::Clamp1(cand, -128, 127);
      for (int i = 0; i < true_num_cands; ++i) {
        if (static_cast<int>(cands[i]) == cand) return;
      }
      cands[true_num_cands++] = static_cast<float>(cand);
    };
    add_cand(0);
    add_cand(pred);
    add_cand(target_cand);
    for (int d = -4; d <= 4; ++d) {
      add_cand(target_cand + d);
      add_cand(pred + d);
    }

    int num_cands = true_num_cands;
    while (num_cands % lanes != 0) {
      cands[num_cands] = cands[0];
      costs[num_cands++] = 0.0f;
    }
    
    evaluate_simd(cands, costs, num_cands);

    // Compute tile DC chroma energy.
    const float* dc_channel = (base == 0.0f) ? dc_values_x : dc_values_b;
    float dc_sum = 0.0f;
    int dc_count = 0;
    for (size_t y = y0; y < y1; ++y) {
      for (size_t x = x0; x < x1; ++x) {
        float val = dc_channel[y * xsize_blocks + x];
        dc_sum += std::abs(val);
        dc_count++;
      }
    }
    float dc_avg = dc_count > 0 ? (dc_sum / dc_count) : 0.0f;

    // Apply higher deadzone penalties on neutral/low-energy tiles to suppress color noise.
    float dc_neutral_threshold = (base == 0.0f) ? 0.10f : 0.35f;
    float deadzone_penalty = 1.0f;
    if (corr < 0.20f && energy < 0.10f && dc_avg < dc_neutral_threshold) {
      deadzone_penalty = 2.5f;
    } else if (corr < 0.35f && energy < 0.05f && dc_avg < (dc_neutral_threshold * 1.5f)) {
      deadzone_penalty = 1.8f;
    }

    // Approximate bit cost for modular hybrid integer residual (cand - pred).
    auto estimate_entropy_bits = [&](int cand, int pred) {
      int res = std::abs(cand - pred);
      if (res == 0) return 0.8f; 
      if (res == 1) return 2.0f;
      return 2.5f + std::log2(static_cast<float>(res));
    };

    // Scale rate penalty by local AQ (tile_q) to allow more distortion in textured areas.
    int best_cand = 0;
    float best_cost = std::numeric_limits<float>::max();
    float local_lambda_bits = 0.005f * tile_q; 
    float local_multiplier_bit_cost = kMultiplierBitCost * tile_q;

    for (int c = 0; c < true_num_cands; ++c) {
      int cand = static_cast<int>(cands[c]);
      float bit_cost = estimate_entropy_bits(cand, pred);
      float cost = costs[c] + local_lambda_bits * bit_cost;
      cost += std::abs(cand) * local_multiplier_bit_cost * deadzone_penalty;
      if (cost < best_cost) {
        best_cost = cost;
        best_cand = cand;
      }
    }

    // Fine local search around the best coarse candidate and analytical target.
    if (step > 1) {
      memset(f_costs, 0, kCandStride * sizeof(float));

      int true_num_fine = 0;
      auto add_fine_cand = [&](int cand) {
        if (cand < -128 || cand > 127) return;
        for (int i = 0; i < true_num_fine; ++i) {
          if (static_cast<int>(f_cands[i]) == cand) return;
        }
        f_cands[true_num_fine++] = static_cast<float>(cand);
      };

      for (int d = -step; d <= step; ++d) {
        add_fine_cand(best_cand + d);
        add_fine_cand(target_cand + d);
      }
      
      if (true_num_fine > 0) {
        int num_fine = true_num_fine;
        while (num_fine % lanes != 0) {
          f_cands[num_fine] = f_cands[0];
          f_costs[num_fine++] = 0.0f;
        }
        
        evaluate_simd(f_cands, f_costs, num_fine);
        
        for (int c = 0; c < true_num_fine; ++c) {
          int cand = static_cast<int>(f_cands[c]);
          float bit_cost = estimate_entropy_bits(cand, pred);
          float cost = f_costs[c] + local_lambda_bits * bit_cost;
          cost += std::abs(cand) * local_multiplier_bit_cost * deadzone_penalty;
          if (cost < best_cost) {
            best_cost = cost;
            best_cand = cand;
          }
        }
      }
    }
    return best_cand;
  };

  row_out_x[tx] = optimize_channel_simd(coeffs_yx, coeffs_x, pred_x, 0.0f);
  row_out_b[tx] = optimize_channel_simd(coeffs_yb, coeffs_b, pred_b, jxl::cms::kYToBRatio);
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
