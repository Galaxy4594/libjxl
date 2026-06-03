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
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/color_encoding_cms.h"
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
// Weighted profile to prioritize low-frequency AC coefficients in CfL search.
static const float kWeightProfile[64] = {
    0.0f,  3.00f, 3.00f, 2.85f, 2.85f, 2.85f, 2.70f, 2.70f, 2.70f, 2.70f, 2.50f,
    2.50f, 2.50f, 2.50f, 2.50f, 2.30f, 2.30f, 2.30f, 2.30f, 2.30f, 2.30f, 2.10f,
    2.10f, 2.10f, 2.10f, 2.10f, 2.10f, 2.10f, 1.90f, 1.90f, 1.90f, 1.90f, 1.90f,
    1.90f, 1.90f, 1.90f, 1.70f, 1.70f, 1.70f, 1.70f, 1.70f, 1.70f, 1.70f, 1.50f,
    1.50f, 1.50f, 1.50f, 1.50f, 1.50f, 1.30f, 1.30f, 1.30f, 1.30f, 1.30f, 1.15f,
    1.15f, 1.15f, 1.15f, 1.05f, 1.05f, 1.05f, 1.00f, 1.00f, 1.00f};

struct CFLFunction {
  static constexpr float kCoeff = 1.f / 3;
  static constexpr float kThres = 100.0f;
  static constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
  CFLFunction(const float* values_m, const float* values_s, size_t num,
              float base, float distance_mul)
      : values_m(values_m),
        values_s(values_s),
        num(num),
        base(base),
        distance_mul(distance_mul) {
    JXL_DASSERT(num % Lanes(df) == 0);
  }

  // Returns f'(x), where f is 1/3 * sum ((|color residual| + 1)^2-1) +
  // distance_mul * x^2 * num.
  float Compute(float x, float eps, float* fpeps, float* fmeps) const {
    float first_derivative = 2 * distance_mul * num * x;
    float first_derivative_peps = 2 * distance_mul * num * (x + eps);
    float first_derivative_meps = 2 * distance_mul * num * (x - eps);

    const auto inv_color_factor = Set(df, kInvColorFactor);
    const auto thres = Set(df, kThres);
    const auto coeffx2 = Set(df, kCoeff * 2.0f);
    const auto one = Set(df, 1.0f);
    const auto zero = Set(df, 0.0f);
    const auto base_v = Set(df, base);
    const auto x_v = Set(df, x);
    const auto xpe_v = Set(df, x + eps);
    const auto xme_v = Set(df, x - eps);
    auto fd_v = Zero(df);
    auto fdpe_v = Zero(df);
    auto fdme_v = Zero(df);

    for (size_t i = 0; i < num; i += Lanes(df)) {
      // color residual = ax + b
      const auto a = Mul(inv_color_factor, Load(df, values_m + i));
      const auto b =
          Sub(Mul(base_v, Load(df, values_m + i)), Load(df, values_s + i));
      const auto v = MulAdd(a, x_v, b);
      const auto vpe = MulAdd(a, xpe_v, b);
      const auto vme = MulAdd(a, xme_v, b);
      const auto av = Abs(v);
      const auto avpe = Abs(vpe);
      const auto avme = Abs(vme);
      const auto acoeffx2 = Mul(coeffx2, a);
      auto d = Mul(acoeffx2, Add(av, one));
      auto dpe = Mul(acoeffx2, Add(avpe, one));
      auto dme = Mul(acoeffx2, Add(avme, one));
      d = IfThenElse(Lt(v, zero), Sub(zero, d), d);
      dpe = IfThenElse(Lt(vpe, zero), Sub(zero, dpe), dpe);
      dme = IfThenElse(Lt(vme, zero), Sub(zero, dme), dme);
      const auto above = Ge(av, thres);
      const auto w = LoadU(df, kWeightProfile + (i % 64));
      // TODO(eustas): use IfThenElseZero
      fd_v = Add(fd_v, Mul(w, IfThenElse(above, zero, d)));
      fdpe_v = Add(fdpe_v, Mul(w, IfThenElse(above, zero, dpe)));
      fdme_v = Add(fdme_v, Mul(w, IfThenElse(above, zero, dme)));
    }

    *fpeps = first_derivative_peps + GetLane(SumOfLanes(df, fdpe_v));
    *fmeps = first_derivative_meps + GetLane(SumOfLanes(df, fdme_v));
    return first_derivative + GetLane(SumOfLanes(df, fd_v));
  }

  const float* JXL_RESTRICT values_m;
  const float* JXL_RESTRICT values_s;
  size_t num;
  float base;
  float distance_mul;
};

// Chroma-from-luma search, values_m will have luma -- and values_s chroma.
int32_t FindBestMultiplier(const float* values_m, const float* values_s,
                           size_t num, float base, float distance_mul,
                           bool fast, float towards_zero) {
  if (num == 0) {
    return 0;
  }
  float x;
  if (fast) {
    static constexpr float kInvColorFactor = 1.0f / kDefaultColorFactor;
    auto ca = Zero(df);
    auto cb = Zero(df);
    const auto inv_color_factor = Set(df, kInvColorFactor);
    const auto base_v = Set(df, base);
    for (size_t i = 0; i < num; i += Lanes(df)) {
      const auto w = LoadU(df, kWeightProfile + (i % 64));
      // color residual = ax + b
      const auto a = Mul(inv_color_factor, Load(df, values_m + i));
      const auto b =
          Sub(Mul(base_v, Load(df, values_m + i)), Load(df, values_s + i));
      ca = MulAdd(Mul(w, a), a, ca);
      cb = MulAdd(Mul(w, a), b, cb);
    }
    // + distance_mul * x^2 * num
    x = -GetLane(SumOfLanes(df, cb)) /
        (GetLane(SumOfLanes(df, ca)) + num * distance_mul * 0.5f);
  } else {
    constexpr float eps = 100;
    constexpr float kClamp = 20.0f;
    CFLFunction fn(values_m, values_s, num, base, distance_mul);
    x = 0;
    // Up to 20 Newton iterations, with approximate derivatives.
    // Derivatives are approximate due to the high amount of noise in the exact
    // derivatives.
    for (size_t i = 0; i < 20; i++) {
      float dfpeps;
      float dfmeps;
      float d_f = fn.Compute(x, eps, &dfpeps, &dfmeps);
      float ddf = (dfpeps - dfmeps) / (2 * eps);
      float kExperimentalInsignificantStabilizer = 0.85;
      float step = d_f / (ddf + kExperimentalInsignificantStabilizer);
      x -= std::min(kClamp, std::max(-kClamp, step));
      if (std::abs(step) < 3e-3) break;
    }
  }
  // CFL seems to be tricky for larger transforms for HF components
  // close to zero. This heuristic brings the solutions closer to zero
  // and reduces red-green oscillations. A better approach would
  // look into variance of the multiplier within separate (e.g. 8x8)
  // areas and only apply this heuristic where there is a high variance.
  // This would give about 1 % more compression density.
  if (x >= towards_zero) {
    x -= towards_zero;
  } else if (x <= -towards_zero) {
    x += towards_zero;
  } else {
    x = 0;
  }
  return jxl::Clamp1(std::round(x), -128.0f, 127.0f);
}

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
                   ImageF* dc_values, Span<float> mem,
                   const ImageSB* old_ytox = nullptr,
                   const ImageSB* old_ytob = nullptr) {
  static_assert(kEncTileDimInBlocks == kColorTileDimInBlocks,
                "Invalid color tile dim");
  size_t xsize_blocks = opsin_rect.xsize() / kBlockDim;
  constexpr float kDistanceMultiplierAC = 1e-9f;
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
  JXL_ENSURE(mem.remove_prefix(4 * kColorTileDim * kColorTileDim));
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

      // Copy AC coefficients in the local block. The order in which
      // coefficients get stored does not matter.
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
  // Path 3: Adaptive CfL quantization.
  // The standard fixed deadzone (2.6) often wipes out chroma correlation on
  // saturated edges at low bitrates, leading to a "leap to white" artifact
  // (the Helmholtz-Kohlrausch effect where desaturated pixels look too bright).
  // By reducing this deadzone based on tile energy,
  // we preserve the chroma signal while still avoiding noise in near-neutral
  // areas. This improves perceptual metrics (SSIMULACRA2) and maintains
  // color fidelity on vivid edges.
  float towards_zero_x = 2.6f;
  float towards_zero_b = 2.6f;

  if (num_ac > 0) {
    auto sum_abs = [](const float* f, size_t n) {
      float s = 0.0f;
      for (size_t i = 0; i < n; ++i) s += std::abs(f[i]);
      return s / n;
    };
    float energy_x = sum_abs(coeffs_x, num_ac);
    float energy_b = sum_abs(coeffs_b, num_ac);

    // If there's significant AC energy in chroma, it's likely a saturated edge.
    if (energy_x > 0.1f) towards_zero_x = 1.2f;
    if (energy_b > 0.1f) towards_zero_b = 1.0f;
  }

  auto evaluate_candidate = [&](const float* m, const float* s,
                                float multiplier, float base) {
    float total_cost = 0;
    for (size_t i = 0; i < num_ac; ++i) {
      float res = s[i] - (base + multiplier / kDefaultColorFactor) * m[i];
      float abs_res = std::abs(res);
      float w = kWeightProfile[i % 64];
      // Perceptual cost: penalize desaturation (residual opposite to original)
      // more than oversaturation.
      float cost = w * abs_res;
      if (multiplier > 0 && res < 0) cost *= 1.2f;
      if (multiplier < 0 && res > 0) cost *= 1.2f;
      total_cost += cost;
    }
    return total_cost;
  };

  auto optimize_multipliers = [&](float init_x, float init_b) {
    if (cparams.speed_tier > SpeedTier::kSquirrel || fast) {
      return std::make_pair((int32_t)std::round(init_x),
                            (int32_t)std::round(init_b));
    }
    int32_t best_x = std::round(init_x);
    int32_t best_b = std::round(init_b);

    auto total_cost = [&](int32_t cand_x, int32_t cand_b) {
      float cost_x = evaluate_candidate(coeffs_yx, coeffs_x, cand_x, 0.0f) +
                     std::abs(cand_x) * 0.1f;
      float cost_b = evaluate_candidate(coeffs_yb, coeffs_b, cand_b,
                                        jxl::cms::kYToBRatio) +
                     std::abs(cand_b) * 0.1f;
      // Joint entropy penalty: encourages YToX and YToB to be closer,
      // saving bits in the Modular stream predicting one from the other.
      float joint_cost = std::abs(cand_x - cand_b) * 0.05f;

      if (old_ytox && old_ytob) {
        int32_t left_x = (tx > 0) ? old_ytox->Row(ty)[tx - 1] : 0;
        int32_t top_x = (ty > 0) ? old_ytox->Row(ty - 1)[tx] : 0;
        int32_t pred_x = (left_x + top_x) / 2;

        int32_t left_b = (tx > 0) ? old_ytob->Row(ty)[tx - 1] : 0;
        int32_t top_b = (ty > 0) ? old_ytob->Row(ty - 1)[tx] : 0;
        int32_t pred_b = (left_b + top_b) / 2;

        joint_cost += std::abs(cand_x - pred_x) * 0.05f;
        joint_cost += std::abs(cand_b - pred_b) * 0.05f;
      }
      return cost_x + cost_b + joint_cost;
    };

    float best_cost = total_cost(best_x, best_b);

    for (int dx : {-2, -1, 0, 1, 2}) {
      for (int db : {-2, -1, 0, 1, 2}) {
        if (dx == 0 && db == 0) continue;
        int32_t cand_x = jxl::Clamp1(best_x + dx, -128, 127);
        int32_t cand_b = jxl::Clamp1(best_b + db, -128, 127);
        float cost = total_cost(cand_x, cand_b);
        if (cost < best_cost) {
          best_cost = cost;
          best_x = cand_x;
          best_b = cand_b;
        }
      }
    }
    return std::make_pair(best_x, best_b);
  };

  float initial_x;
  float initial_b;
  if (old_ytox && old_ytob) {
    initial_x = old_ytox->Row(ty)[tx];
    initial_b = old_ytob->Row(ty)[tx];
  } else {
    initial_x = FindBestMultiplier(coeffs_yx, coeffs_x, num_ac, 0.0f,
                                   kDistanceMultiplierAC, fast, towards_zero_x);
    initial_b =
        FindBestMultiplier(coeffs_yb, coeffs_b, num_ac, jxl::cms::kYToBRatio,
                           kDistanceMultiplierAC, fast, towards_zero_b);
  }

  auto best_multipliers = optimize_multipliers(initial_x, initial_b);
  row_out_x[tx] = best_multipliers.first;
  row_out_b[tx] = best_multipliers.second;
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
      scratch, nullptr, nullptr);
}

Status CfLHeuristics::RefineMap(const Image3F& opsin, const Rect& opsin_rect,
                                const DequantMatrices& dequant,
                                const AcStrategyImage& ac_strategy,
                                const ImageI& raw_quant_field,
                                const Quantizer& quantizer,
                                const CompressParams& cparams, ThreadPool* pool,
                                ColorCorrelationMap* cmap) {
  JxlMemoryManager* memory_manager = opsin.memory_manager();

  // Create copies of the original maps
  JXL_ASSIGN_OR_RETURN(ImageSB old_ytox,
                       ImageSB::Create(memory_manager, cmap->ytox_map.xsize(),
                                       cmap->ytox_map.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(cmap->ytox_map, &old_ytox));

  JXL_ASSIGN_OR_RETURN(ImageSB old_ytob,
                       ImageSB::Create(memory_manager, cmap->ytob_map.xsize(),
                                       cmap->ytob_map.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(cmap->ytob_map, &old_ytob));

  size_t xsize_blocks = opsin_rect.xsize() / kBlockDim;
  size_t ysize_blocks = opsin_rect.ysize() / kBlockDim;
  size_t n_enc_tiles = DivCeil(xsize_blocks, kEncTileDimInBlocks);
  size_t num_tiles = n_enc_tiles * DivCeil(ysize_blocks, kEncTileDimInBlocks);

  auto process_tile = [&](const uint32_t tid, const size_t thread) -> Status {
    size_t tx = tid % n_enc_tiles;
    size_t ty = tid / n_enc_tiles;
    size_t by0 = ty * kEncTileDimInBlocks;
    size_t by1 = std::min((ty + 1) * kEncTileDimInBlocks, ysize_blocks);
    size_t bx0 = tx * kEncTileDimInBlocks;
    size_t bx1 = std::min((tx + 1) * kEncTileDimInBlocks, xsize_blocks);
    Rect r(bx0, by0, bx1 - bx0, by1 - by0);

    Span<float> scratch(mem.address<float>() + thread * ItemsPerThread(),
                        ItemsPerThread());

    return HWY_DYNAMIC_DISPATCH(ComputeTile)(
        opsin, opsin_rect, dequant, &ac_strategy, &raw_quant_field, &quantizer,
        cparams, r, /*fast=*/false, /*use_dct8=*/false, &cmap->ytox_map,
        &cmap->ytob_map, &dc_values, scratch, &old_ytox, &old_ytob);
  };

  return RunOnPool(pool, 0, num_tiles, ThreadPool::NoInit, process_tile,
                   "Refine CfL");
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
