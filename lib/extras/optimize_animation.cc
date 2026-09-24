// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/optimize_animation.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

namespace {

uint64_t Gcd(uint64_t a, uint64_t b) {
  while (b != 0) {
    uint64_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}

bool FramesEqual(const PackedFrame& f1, const PackedFrame& f2) {
  if (f1.color.xsize != f2.color.xsize || f1.color.ysize != f2.color.ysize) {
    return false;
  }
  if (f1.color.format.num_channels != f2.color.format.num_channels ||
      f1.color.format.data_type != f2.color.format.data_type) {
    return false;
  }
  if (f1.extra_channels.size() != f2.extra_channels.size()) {
    return false;
  }
  const size_t bytes_per_sample =
      PackedImage::BitsPerChannel(f1.color.format.data_type) / 8;
  const size_t row_bytes =
      f1.color.xsize * f1.color.format.num_channels * bytes_per_sample;
  for (size_t y = 0; y < f1.color.ysize; ++y) {
    const uint8_t* r1 =
        static_cast<const uint8_t*>(f1.color.pixels()) + y * f1.color.stride;
    const uint8_t* r2 =
        static_cast<const uint8_t*>(f2.color.pixels()) + y * f2.color.stride;
    if (memcmp(r1, r2, row_bytes) != 0) return false;
  }
  for (size_t ec = 0; ec < f1.extra_channels.size(); ++ec) {
    const auto& ec1 = f1.extra_channels[ec];
    const auto& ec2 = f2.extra_channels[ec];
    if (ec1.xsize != ec2.xsize || ec1.ysize != ec2.ysize ||
        ec1.format.num_channels != ec2.format.num_channels ||
        ec1.format.data_type != ec2.format.data_type) {
      return false;
    }
    const size_t ec_bytes_per_sample =
        PackedImage::BitsPerChannel(ec1.format.data_type) / 8;
    const size_t ec_row_bytes =
        ec1.xsize * ec1.format.num_channels * ec_bytes_per_sample;
    for (size_t y = 0; y < ec1.ysize; ++y) {
      const uint8_t* r1 =
          static_cast<const uint8_t*>(ec1.pixels()) + y * ec1.stride;
      const uint8_t* r2 =
          static_cast<const uint8_t*>(ec2.pixels()) + y * ec2.stride;
      if (memcmp(r1, r2, ec_row_bytes) != 0) return false;
    }
  }
  return true;
}

}  // namespace

Status OptimizeAnimation(PackedPixelFile* ppf) {
  if (!ppf || !ppf->info.have_animation || ppf->frames.size() <= 1) {
    return true;
  }
  const size_t W = ppf->info.xsize;
  const size_t H = ppf->info.ysize;
  if (W == 0 || H == 0) return true;

  // 1. Deduplicate consecutive identical frames
  for (size_t i = 1; i < ppf->frames.size();) {
    if (FramesEqual(ppf->frames[i], ppf->frames[i - 1])) {
      ppf->frames[i - 1].frame_info.duration +=
          ppf->frames[i].frame_info.duration;
      ppf->frames.erase(ppf->frames.begin() + i);
    } else {
      ++i;
    }
  }

  // 1b. Deduplicate identical frame at looping animation boundary (last frame
  // == first frame)
  if (ppf->frames.size() > 1 && (ppf->info.animation.num_loops == 0 ||
                                 ppf->info.animation.num_loops == 1 ||
                                 ppf->info.animation.have_timecodes == 0)) {
    if (FramesEqual(ppf->frames.back(), ppf->frames.front())) {
      ppf->frames[0].frame_info.duration +=
          ppf->frames.back().frame_info.duration;
      ppf->frames.pop_back();
    }
  }

  if (ppf->frames.size() <= 1) {
    if (!ppf->frames.empty()) {
      ppf->frames[0].frame_info.is_last = JXL_TRUE;
    }
    return true;
  }

  // 1c. Optimize timebase and frame durations by greatest common divisor (GCD).
  // When frame durations share a common factor (or uniform frame rate),
  // reducing durations minimizes the variable-length integer bit allocation in
  // each frame header (e.g. duration == 1 costs only 2 bits via Val(1) vs 10
  // bits).
  if (ppf->info.have_animation && ppf->frames.size() > 1 &&
      ppf->info.animation.tps_numerator > 0 &&
      ppf->info.animation.tps_denominator > 0) {
    uint32_t g = 0;
    for (const auto& frame : ppf->frames) {
      if (frame.frame_info.duration > 0) {
        g = (g == 0) ? frame.frame_info.duration
                     : static_cast<uint32_t>(Gcd(g, frame.frame_info.duration));
      }
    }
    if (g > 1) {
      for (uint32_t k = g; k >= 2; --k) {
        if (g % k != 0) continue;
        uint64_t num = ppf->info.animation.tps_numerator;
        uint64_t den =
            static_cast<uint64_t>(k) * ppf->info.animation.tps_denominator;
        uint64_t c = Gcd(num, den);
        num /= c;
        den /= c;
        if (den <= 1024 && num < (1ull << 30ull)) {
          ppf->info.animation.tps_numerator = static_cast<uint32_t>(num);
          ppf->info.animation.tps_denominator = static_cast<uint32_t>(den);
          for (auto& frame : ppf->frames) {
            frame.frame_info.duration /= k;
          }
          break;
        }
      }
    }
  }

  // 2. Check if all frames are full canvas
  bool all_full_canvas = true;
  for (const auto& frame : ppf->frames) {
    if (frame.frame_info.layer_info.have_crop != 0 ||
        frame.frame_info.layer_info.crop_x0 != 0 ||
        frame.frame_info.layer_info.crop_y0 != 0 || frame.color.xsize != W ||
        frame.color.ysize != H) {
      all_full_canvas = false;
      break;
    }
  }
  if (!all_full_canvas) {
    ppf->frames.back().frame_info.is_last = JXL_TRUE;
    return true;
  }

  // 2b. Clean dirty transparent pixels (where alpha == 0, zero out color
  // channels). Matches apngopt's optim_dirty() and libwebp's
  // WebPCleanupTransparentArea(). When a pixel is 100% transparent, whatever
  // RGB color is stored underneath is completely invisible, but
  // residual/palette color noise wastes entropy and causes artificial size
  // discrepancies between different input containers.
  for (PackedFrame& frame : ppf->frames) {
    if (!frame.extra_channels.empty()) {
      if (frame.color.format.data_type == JXL_TYPE_UINT8 &&
          frame.extra_channels[0].format.data_type == JXL_TYPE_UINT8) {
        const size_t cps = frame.color.format.num_channels;
        for (size_t y = 0; y < frame.color.ysize; ++y) {
          uint8_t* crow = static_cast<uint8_t*>(frame.color.pixels()) +
                          y * frame.color.stride;
          const uint8_t* arow =
              static_cast<const uint8_t*>(frame.extra_channels[0].pixels()) +
              y * frame.extra_channels[0].stride;
          for (size_t x = 0; x < frame.color.xsize; ++x) {
            if (arow[x] == 0) {
              memset(crow + x * cps, 0, cps);
            }
          }
        }
      }
    } else if (frame.color.format.num_channels > ppf->info.num_color_channels) {
      if (frame.color.format.data_type == JXL_TYPE_UINT8) {
        const size_t cps = frame.color.format.num_channels;
        const size_t aidx = ppf->info.num_color_channels;
        for (size_t y = 0; y < frame.color.ysize; ++y) {
          uint8_t* crow = static_cast<uint8_t*>(frame.color.pixels()) +
                          y * frame.color.stride;
          for (size_t x = 0; x < frame.color.xsize; ++x) {
            if (crow[x * cps + aidx] == 0) {
              memset(crow + x * cps, 0, aidx);
            }
          }
        }
      }
    }
  }

  // 3. Delta optimization pipeline
  ppf->frames[0].frame_info.layer_info.have_crop = 0;
  ppf->frames[0].frame_info.layer_info.crop_x0 = 0;
  ppf->frames[0].frame_info.layer_info.crop_y0 = 0;
  ppf->frames[0].frame_info.layer_info.xsize = W;
  ppf->frames[0].frame_info.layer_info.ysize = H;
  ppf->frames[0].frame_info.layer_info.blend_info.blendmode = JXL_BLEND_REPLACE;
  ppf->frames[0].frame_info.layer_info.blend_info.source = 0;
  ppf->frames[0].frame_info.layer_info.save_as_reference = 1;

  JXL_ASSIGN_OR_RETURN(PackedFrame canvas, ppf->frames[0].Copy());

  const bool has_extra_channel_alpha = !ppf->frames[0].extra_channels.empty();
  const bool has_interleaved_alpha =
      (!has_extra_channel_alpha &&
       ppf->frames[0].color.format.num_channels > ppf->info.num_color_channels);

  const size_t bytes_per_sample =
      PackedImage::BitsPerChannel(ppf->frames[0].color.format.data_type) / 8;
  const size_t color_pixel_stride =
      ppf->frames[0].color.format.num_channels * bytes_per_sample;

  size_t alpha_pixel_stride = 0;
  if (has_extra_channel_alpha) {
    size_t alpha_bytes_per_sample =
        PackedImage::BitsPerChannel(
            ppf->frames[0].extra_channels[0].format.data_type) /
        8;
    alpha_pixel_stride = ppf->frames[0].extra_channels[0].format.num_channels *
                         alpha_bytes_per_sample;
  }

  for (size_t i = 1; i < ppf->frames.size(); ++i) {
    PackedFrame& curr_frame = ppf->frames[i];

    size_t x_min = W, y_min = H, x_max = 0, y_max = 0;
    bool found_diff = false;

    for (size_t y = 0; y < H; ++y) {
      const uint8_t* c_color_row =
          static_cast<const uint8_t*>(canvas.color.pixels()) +
          y * canvas.color.stride;
      const uint8_t* f_color_row =
          static_cast<const uint8_t*>(curr_frame.color.pixels()) +
          y * curr_frame.color.stride;
      const uint8_t* c_alpha_row =
          has_extra_channel_alpha ? (static_cast<const uint8_t*>(
                                         canvas.extra_channels[0].pixels()) +
                                     y * canvas.extra_channels[0].stride)
                                  : nullptr;
      const uint8_t* f_alpha_row =
          has_extra_channel_alpha
              ? (static_cast<const uint8_t*>(
                     curr_frame.extra_channels[0].pixels()) +
                 y * curr_frame.extra_channels[0].stride)
              : nullptr;

      for (size_t x = 0; x < W; ++x) {
        bool diff = false;
        if (memcmp(c_color_row + x * color_pixel_stride,
                   f_color_row + x * color_pixel_stride,
                   color_pixel_stride) != 0) {
          diff = true;
        } else if (has_extra_channel_alpha &&
                   memcmp(c_alpha_row + x * alpha_pixel_stride,
                          f_alpha_row + x * alpha_pixel_stride,
                          alpha_pixel_stride) != 0) {
          diff = true;
        }
        if (diff) {
          found_diff = true;
          if (x < x_min) x_min = x;
          if (x > x_max) x_max = x;
          if (y < y_min) y_min = y;
          if (y > y_max) y_max = y;
        }
      }
    }

    if (!found_diff) {
      x_min = 0;
      y_min = 0;
      x_max = 0;
      y_max = 0;
    }

    size_t x0 = x_min;
    size_t y0 = y_min;
    size_t w = x_max - x_min + 1;
    size_t h = y_max - y_min + 1;

    bool can_use_blend = has_extra_channel_alpha || has_interleaved_alpha;
    if (can_use_blend && found_diff) {
      if (has_extra_channel_alpha) {
        for (size_t y = 0; y < h; ++y) {
          size_t cy = y0 + y;
          const uint8_t* c_color_row =
              static_cast<const uint8_t*>(canvas.color.pixels()) +
              cy * canvas.color.stride;
          const uint8_t* f_color_row =
              static_cast<const uint8_t*>(curr_frame.color.pixels()) +
              cy * curr_frame.color.stride;
          const uint8_t* c_alpha_row =
              static_cast<const uint8_t*>(canvas.extra_channels[0].pixels()) +
              cy * canvas.extra_channels[0].stride;
          const uint8_t* f_alpha_row =
              static_cast<const uint8_t*>(
                  curr_frame.extra_channels[0].pixels()) +
              cy * curr_frame.extra_channels[0].stride;

          for (size_t x = 0; x < w; ++x) {
            size_t cx = x0 + x;
            bool diff = (memcmp(c_color_row + cx * color_pixel_stride,
                                f_color_row + cx * color_pixel_stride,
                                color_pixel_stride) != 0 ||
                         memcmp(c_alpha_row + cx * alpha_pixel_stride,
                                f_alpha_row + cx * alpha_pixel_stride,
                                alpha_pixel_stride) != 0);
            if (diff) {
              if (curr_frame.extra_channels[0].format.data_type ==
                  JXL_TYPE_UINT8) {
                if (f_alpha_row[cx * alpha_pixel_stride] != 255) {
                  can_use_blend = false;
                  break;
                }
              }
            }
          }
          if (!can_use_blend) break;
        }
      } else if (has_interleaved_alpha) {
        size_t alpha_channel_offset =
            ppf->info.num_color_channels * bytes_per_sample;
        for (size_t y = 0; y < h; ++y) {
          size_t cy = y0 + y;
          const uint8_t* c_color_row =
              static_cast<const uint8_t*>(canvas.color.pixels()) +
              cy * canvas.color.stride;
          const uint8_t* f_color_row =
              static_cast<const uint8_t*>(curr_frame.color.pixels()) +
              cy * curr_frame.color.stride;

          for (size_t x = 0; x < w; ++x) {
            size_t cx = x0 + x;
            bool diff = (memcmp(c_color_row + cx * color_pixel_stride,
                                f_color_row + cx * color_pixel_stride,
                                color_pixel_stride) != 0);
            if (diff) {
              if (curr_frame.color.format.data_type == JXL_TYPE_UINT8) {
                if (f_color_row[cx * color_pixel_stride +
                                alpha_channel_offset] != 255) {
                  can_use_blend = false;
                  break;
                }
              }
            }
          }
          if (!can_use_blend) break;
        }
      }
    }

    JXL_ASSIGN_OR_RETURN(PackedFrame cropped,
                         PackedFrame::Create(w, h, curr_frame.color.format));
    cropped.frame_info = curr_frame.frame_info;
    cropped.name = curr_frame.name;
    for (const auto& ec : curr_frame.extra_channels) {
      JXL_ASSIGN_OR_RETURN(PackedImage cropped_ec,
                           PackedImage::Create(w, h, ec.format));
      cropped.extra_channels.emplace_back(std::move(cropped_ec));
    }

    for (size_t y = 0; y < h; ++y) {
      size_t cy = y0 + y;
      const uint8_t* c_color_row =
          static_cast<const uint8_t*>(canvas.color.pixels()) +
          cy * canvas.color.stride;
      const uint8_t* f_color_row =
          static_cast<const uint8_t*>(curr_frame.color.pixels()) +
          cy * curr_frame.color.stride;
      uint8_t* dst_color_row = static_cast<uint8_t*>(cropped.color.pixels()) +
                               y * cropped.color.stride;

      const uint8_t* c_alpha_row =
          has_extra_channel_alpha ? (static_cast<const uint8_t*>(
                                         canvas.extra_channels[0].pixels()) +
                                     cy * canvas.extra_channels[0].stride)
                                  : nullptr;
      const uint8_t* f_alpha_row =
          has_extra_channel_alpha
              ? (static_cast<const uint8_t*>(
                     curr_frame.extra_channels[0].pixels()) +
                 cy * curr_frame.extra_channels[0].stride)
              : nullptr;
      uint8_t* dst_alpha_row =
          has_extra_channel_alpha
              ? (static_cast<uint8_t*>(cropped.extra_channels[0].pixels()) +
                 y * cropped.extra_channels[0].stride)
              : nullptr;

      for (size_t x = 0; x < w; ++x) {
        size_t cx = x0 + x;
        bool diff = false;
        if (memcmp(c_color_row + cx * color_pixel_stride,
                   f_color_row + cx * color_pixel_stride,
                   color_pixel_stride) != 0) {
          diff = true;
        } else if (has_extra_channel_alpha &&
                   memcmp(c_alpha_row + cx * alpha_pixel_stride,
                          f_alpha_row + cx * alpha_pixel_stride,
                          alpha_pixel_stride) != 0) {
          diff = true;
        }

        if (can_use_blend && !diff) {
          if (has_extra_channel_alpha) {
            memset(dst_color_row + x * color_pixel_stride, 0,
                   color_pixel_stride);
            memset(dst_alpha_row + x * alpha_pixel_stride, 0,
                   alpha_pixel_stride);
          } else if (has_interleaved_alpha) {
            memset(dst_color_row + x * color_pixel_stride, 0,
                   color_pixel_stride);
          }
        } else {
          memcpy(dst_color_row + x * color_pixel_stride,
                 f_color_row + cx * color_pixel_stride, color_pixel_stride);
          if (has_extra_channel_alpha) {
            memcpy(dst_alpha_row + x * alpha_pixel_stride,
                   f_alpha_row + cx * alpha_pixel_stride, alpha_pixel_stride);
          }
        }
      }
      for (size_t ec = 1; ec < curr_frame.extra_channels.size(); ++ec) {
        const auto& src_ec = curr_frame.extra_channels[ec];
        auto& dst_ec = cropped.extra_channels[ec];
        size_t ec_stride_bytes =
            src_ec.format.num_channels *
            (PackedImage::BitsPerChannel(src_ec.format.data_type) / 8);
        const uint8_t* src_row = static_cast<const uint8_t*>(src_ec.pixels()) +
                                 cy * src_ec.stride + x0 * ec_stride_bytes;
        uint8_t* dst_row =
            static_cast<uint8_t*>(dst_ec.pixels()) + y * dst_ec.stride;
        memcpy(dst_row, src_row, w * ec_stride_bytes);
      }
    }

    cropped.frame_info.duration = curr_frame.frame_info.duration;
    cropped.frame_info.timecode = curr_frame.frame_info.timecode;
    cropped.frame_info.layer_info.have_crop =
        (x0 != 0 || y0 != 0 || w != W || h != H);
    cropped.frame_info.layer_info.crop_x0 = x0;
    cropped.frame_info.layer_info.crop_y0 = y0;
    cropped.frame_info.layer_info.xsize = w;
    cropped.frame_info.layer_info.ysize = h;
    cropped.frame_info.layer_info.blend_info.blendmode =
        can_use_blend ? JXL_BLEND_BLEND : JXL_BLEND_REPLACE;
    cropped.frame_info.layer_info.blend_info.source = 1;
    cropped.frame_info.layer_info.blend_info.alpha = 0;
    cropped.frame_info.layer_info.blend_info.clamp = 1;
    cropped.frame_info.layer_info.save_as_reference = 1;

    canvas = std::move(curr_frame);
    ppf->frames[i] = std::move(cropped);
  }

  ppf->frames.back().frame_info.is_last = JXL_TRUE;
  return true;
}

}  // namespace extras
}  // namespace jxl
