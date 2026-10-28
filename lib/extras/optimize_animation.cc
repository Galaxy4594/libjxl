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

bool DeltaEnablesPalette(const uint8_t* prev_color, size_t prev_stride,
                         const uint8_t* curr_color, size_t curr_stride,
                         size_t xsize, size_t ysize,
                         size_t color_pixel_stride) {
  std::vector<uint32_t> table(1024);
  const auto count_colors = [&](bool delta_only) -> size_t {
    memset(table.data(), 0, table.size() * sizeof(uint32_t));
    size_t count = 0;
    for (size_t y = 0; y < ysize; ++y) {
      const uint8_t* prev_row = prev_color + y * prev_stride;
      const uint8_t* curr_row = curr_color + y * curr_stride;
      for (size_t x = 0; x < xsize; ++x) {
        if (delta_only && memcmp(prev_row + x * color_pixel_stride,
                                 curr_row + x * color_pixel_stride,
                                 color_pixel_stride) == 0) {
          continue;
        }
        uint32_t c =
            (static_cast<uint32_t>(curr_row[x * color_pixel_stride]) << 16) |
            (static_cast<uint32_t>(curr_row[x * color_pixel_stride + 1]) << 8) |
            static_cast<uint32_t>(curr_row[x * color_pixel_stride + 2]);
        uint32_t entry = 0x01000000 | c;
        uint32_t h = (c * 2654435761u) & 1023;
        while (table[h] != 0 && table[h] != entry) {
          h = (h + 1) & 1023;
        }
        if (table[h] == 0) {
          table[h] = entry;
          if (++count > 256) return count;
        }
      }
    }
    return count;
  };

  size_t delta_colors = count_colors(/*delta_only=*/true);
  if (delta_colors > 256) return false;
  size_t full_colors = count_colors(/*delta_only=*/false);
  return (full_colors > 256 && delta_colors <= 256);
}

void CleanTransparentPixels(PackedFrame* frame, size_t num_color_channels) {
  if (!frame->extra_channels.empty()) {
    if (frame->color.format.data_type == JXL_TYPE_UINT8 &&
        frame->extra_channels[0].format.data_type == JXL_TYPE_UINT8) {
      const size_t num_channels = frame->color.format.num_channels;
      for (size_t y = 0; y < frame->color.ysize; ++y) {
        uint8_t* color_row = static_cast<uint8_t*>(frame->color.pixels()) +
                             y * frame->color.stride;
        const uint8_t* alpha_row =
            static_cast<const uint8_t*>(frame->extra_channels[0].pixels()) +
            y * frame->extra_channels[0].stride;
        for (size_t x = 0; x < frame->color.xsize; ++x) {
          if (alpha_row[x] == 0) {
            memset(color_row + x * num_channels, 0, num_channels);
          }
        }
      }
    }
  } else if (frame->color.format.num_channels > num_color_channels &&
             frame->color.format.data_type == JXL_TYPE_UINT8) {
    const size_t num_channels = frame->color.format.num_channels;
    const size_t alpha_idx = num_color_channels;
    for (size_t y = 0; y < frame->color.ysize; ++y) {
      uint8_t* color_row = static_cast<uint8_t*>(frame->color.pixels()) +
                           y * frame->color.stride;
      for (size_t x = 0; x < frame->color.xsize; ++x) {
        if (color_row[x * num_channels + alpha_idx] == 0) {
          memset(color_row + x * num_channels, 0, alpha_idx);
        }
      }
    }
  }
}

void NormalizeDurationTicks(PackedPixelFile* ppf) {
  if (!ppf->info.have_animation || ppf->frames.size() <= 1 ||
      ppf->info.animation.tps_numerator == 0 ||
      ppf->info.animation.tps_denominator == 0) {
    return;
  }
  uint32_t g = 0;
  for (const auto& frame : ppf->frames) {
    if (frame.frame_info.duration > 0) {
      g = (g == 0) ? frame.frame_info.duration
                   : static_cast<uint32_t>(Gcd(g, frame.frame_info.duration));
    }
  }
  if (g <= 1) return;

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

}  // namespace

Status OptimizeAnimation(PackedPixelFile* ppf) {
  if (!ppf || !ppf->info.have_animation || ppf->frames.size() <= 1) {
    return true;
  }
  const size_t xsize = ppf->info.xsize;
  const size_t ysize = ppf->info.ysize;
  if (xsize == 0 || ysize == 0) return true;

  for (size_t i = 1; i < ppf->frames.size();) {
    if (FramesEqual(ppf->frames[i], ppf->frames[i - 1])) {
      ppf->frames[i - 1].frame_info.duration +=
          ppf->frames[i].frame_info.duration;
      ppf->frames.erase(ppf->frames.begin() + i);
    } else {
      ++i;
    }
  }

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

  NormalizeDurationTicks(ppf);

  for (const auto& frame : ppf->frames) {
    if (frame.frame_info.layer_info.have_crop != 0 ||
        frame.frame_info.layer_info.crop_x0 != 0 ||
        frame.frame_info.layer_info.crop_y0 != 0 ||
        frame.color.xsize != xsize || frame.color.ysize != ysize) {
      ppf->frames.back().frame_info.is_last = JXL_TRUE;
      return true;
    }
  }

  for (PackedFrame& frame : ppf->frames) {
    CleanTransparentPixels(&frame, ppf->info.num_color_channels);
  }

  const bool initial_extra_alpha = !ppf->frames[0].extra_channels.empty();
  const bool initial_interleaved_alpha =
      (!initial_extra_alpha &&
       ppf->frames[0].color.format.num_channels > ppf->info.num_color_channels);

  if (!initial_extra_alpha && !initial_interleaved_alpha &&
      ppf->info.num_color_channels == 3 &&
      ppf->frames[0].color.format.data_type == JXL_TYPE_UINT8) {
    // Only allocate an extra alpha channel if at least one frame benefits from
    // delta blending; otherwise photographic/video-like animations pay extra
    // channel overhead without using transparency.
    bool needs_alpha_for_blending = false;
    const size_t color_pixel_stride = 3;
    for (size_t i = 1; i < ppf->frames.size(); ++i) {
      const PackedFrame& prev = ppf->frames[i - 1];
      const PackedFrame& curr = ppf->frames[i];
      size_t x_min = xsize;
      size_t y_min = ysize;
      size_t x_max = 0;
      size_t y_max = 0;
      size_t num_diff_pixels = 0;
      for (size_t y = 0; y < ysize; ++y) {
        const uint8_t* prev_row =
            static_cast<const uint8_t*>(prev.color.pixels()) +
            y * prev.color.stride;
        const uint8_t* curr_row =
            static_cast<const uint8_t*>(curr.color.pixels()) +
            y * curr.color.stride;
        for (size_t x = 0; x < xsize; ++x) {
          if (memcmp(prev_row + x * color_pixel_stride,
                     curr_row + x * color_pixel_stride,
                     color_pixel_stride) != 0) {
            ++num_diff_pixels;
            x_min = std::min(x_min, x);
            x_max = std::max(x_max, x);
            y_min = std::min(y_min, y);
            y_max = std::max(y_max, y);
          }
        }
      }
      if (num_diff_pixels > 0) {
        const size_t box_pixels = (x_max - x_min + 1) * (y_max - y_min + 1);
        if (box_pixels >= 0.85 * xsize * ysize) {
          const size_t same_pixels = box_pixels - num_diff_pixels;
          const double same_ratio = static_cast<double>(same_pixels) /
                                    static_cast<double>(box_pixels);
          if (same_ratio >= 0.70 ||
              (same_ratio >= 0.20 &&
               DeltaEnablesPalette(
                   static_cast<const uint8_t*>(prev.color.pixels()),
                   prev.color.stride,
                   static_cast<const uint8_t*>(curr.color.pixels()),
                   curr.color.stride, xsize, ysize, color_pixel_stride))) {
            needs_alpha_for_blending = true;
            break;
          }
        }
      }
    }

    if (needs_alpha_for_blending) {
      ppf->info.alpha_bits = 8;
      ppf->info.num_extra_channels = 1;
      PackedExtraChannel ec;
      ec.ec_info.type = JXL_CHANNEL_ALPHA;
      ec.ec_info.bits_per_sample = 8;
      ec.ec_info.dim_shift = 0;
      ec.index = 0;
      ppf->extra_channels_info.push_back(ec);

      const JxlPixelFormat alpha_format{
          /*num_channels=*/1u,
          /*data_type=*/JXL_TYPE_UINT8,
          /*endianness=*/JXL_NATIVE_ENDIAN,
          /*align=*/0,
      };

      for (PackedFrame& frame : ppf->frames) {
        JXL_ASSIGN_OR_RETURN(PackedImage frame_alpha,
                             PackedImage::Create(xsize, ysize, alpha_format));
        memset(frame_alpha.pixels(), 255, frame_alpha.pixels_size);
        frame.extra_channels.emplace_back(std::move(frame_alpha));
      }
    }
  }

  ppf->frames[0].frame_info.layer_info.have_crop = 0;
  ppf->frames[0].frame_info.layer_info.crop_x0 = 0;
  ppf->frames[0].frame_info.layer_info.crop_y0 = 0;
  ppf->frames[0].frame_info.layer_info.xsize = xsize;
  ppf->frames[0].frame_info.layer_info.ysize = ysize;
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
  const size_t alpha_channel_offset =
      ppf->info.num_color_channels * bytes_per_sample;

  size_t alpha_pixel_stride = 0;
  if (has_extra_channel_alpha) {
    const size_t alpha_bytes_per_sample =
        PackedImage::BitsPerChannel(
            ppf->frames[0].extra_channels[0].format.data_type) /
        8;
    alpha_pixel_stride = ppf->frames[0].extra_channels[0].format.num_channels *
                         alpha_bytes_per_sample;
  }

  std::vector<uint8_t> diff_mask(xsize * ysize);

  for (size_t i = 1; i < ppf->frames.size(); ++i) {
    PackedFrame& curr_frame = ppf->frames[i];

    size_t x_min = xsize;
    size_t y_min = ysize;
    size_t x_max = 0;
    size_t y_max = 0;
    bool found_diff = false;
    bool has_non_opaque_diff = false;
    size_t num_diff_pixels = 0;

    for (size_t y = 0; y < ysize; ++y) {
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

      for (size_t x = 0; x < xsize; ++x) {
        const bool diff = (memcmp(c_color_row + x * color_pixel_stride,
                                  f_color_row + x * color_pixel_stride,
                                  color_pixel_stride) != 0) ||
                          (has_extra_channel_alpha &&
                           memcmp(c_alpha_row + x * alpha_pixel_stride,
                                  f_alpha_row + x * alpha_pixel_stride,
                                  alpha_pixel_stride) != 0);
        diff_mask[y * xsize + x] = diff ? 1 : 0;
        if (diff) {
          found_diff = true;
          ++num_diff_pixels;
          x_min = std::min(x_min, x);
          x_max = std::max(x_max, x);
          y_min = std::min(y_min, y);
          y_max = std::max(y_max, y);
          if (has_extra_channel_alpha &&
              curr_frame.extra_channels[0].format.data_type == JXL_TYPE_UINT8 &&
              f_alpha_row[x * alpha_pixel_stride] != 255) {
            has_non_opaque_diff = true;
          } else if (has_interleaved_alpha &&
                     curr_frame.color.format.data_type == JXL_TYPE_UINT8 &&
                     f_color_row[x * color_pixel_stride +
                                 alpha_channel_offset] != 255) {
            has_non_opaque_diff = true;
          }
        }
      }
    }

    if (!found_diff) {
      x_min = 0;
      y_min = 0;
      x_max = 0;
      y_max = 0;
    }

    const size_t x0 = x_min;
    const size_t y0 = y_min;
    const size_t w = x_max - x_min + 1;
    const size_t h = y_max - y_min + 1;
    const size_t box_pixels = w * h;
    const size_t same_pixels =
        (box_pixels > num_diff_pixels) ? (box_pixels - num_diff_pixels) : 0;
    const double same_ratio = (box_pixels > 0)
                                  ? (static_cast<double>(same_pixels) /
                                     static_cast<double>(box_pixels))
                                  : 0.0;
    const bool bounding_box_is_large = (box_pixels >= 0.85 * xsize * ysize);
    bool can_use_blend = false;
    if ((has_extra_channel_alpha || has_interleaved_alpha) &&
        !has_non_opaque_diff) {
      if (!bounding_box_is_large || same_ratio >= 0.70) {
        can_use_blend = true;
      } else if (same_ratio >= 0.20 &&
                 ppf->frames[0].color.format.data_type == JXL_TYPE_UINT8 &&
                 ppf->info.num_color_channels == 3) {
        can_use_blend = DeltaEnablesPalette(
            static_cast<const uint8_t*>(canvas.color.pixels()),
            canvas.color.stride,
            static_cast<const uint8_t*>(curr_frame.color.pixels()),
            curr_frame.color.stride, xsize, ysize, color_pixel_stride);
      }
    }

    if (bounding_box_is_large && !can_use_blend) {
      curr_frame.frame_info.layer_info.have_crop = 0;
      curr_frame.frame_info.layer_info.crop_x0 = 0;
      curr_frame.frame_info.layer_info.crop_y0 = 0;
      curr_frame.frame_info.layer_info.xsize = xsize;
      curr_frame.frame_info.layer_info.ysize = ysize;
      curr_frame.frame_info.layer_info.blend_info.source = 1;
      curr_frame.frame_info.layer_info.blend_info.alpha = 0;
      curr_frame.frame_info.layer_info.blend_info.clamp = 1;
      curr_frame.frame_info.layer_info.save_as_reference = 1;

      if (curr_frame.frame_info.layer_info.blend_info.blendmode ==
          JXL_BLEND_BLEND) {
        // Preserve native blended overlays and composite onto canvas so
        // subsequent frames compare against the rendered canvas state.
        for (size_t y = 0; y < ysize; ++y) {
          uint8_t* c_color_row = static_cast<uint8_t*>(canvas.color.pixels()) +
                                 y * canvas.color.stride;
          const uint8_t* f_color_row =
              static_cast<const uint8_t*>(curr_frame.color.pixels()) +
              y * curr_frame.color.stride;
          uint8_t* c_alpha_row =
              has_extra_channel_alpha
                  ? (static_cast<uint8_t*>(canvas.extra_channels[0].pixels()) +
                     y * canvas.extra_channels[0].stride)
                  : nullptr;
          const uint8_t* f_alpha_row =
              has_extra_channel_alpha
                  ? (static_cast<const uint8_t*>(
                         curr_frame.extra_channels[0].pixels()) +
                     y * curr_frame.extra_channels[0].stride)
                  : nullptr;

          for (size_t x = 0; x < xsize; ++x) {
            uint8_t alpha = 255;
            if (has_extra_channel_alpha) {
              alpha = f_alpha_row[x * alpha_pixel_stride];
            } else if (has_interleaved_alpha) {
              alpha =
                  f_color_row[x * color_pixel_stride + alpha_channel_offset];
            }
            if (alpha > 0) {
              memcpy(c_color_row + x * color_pixel_stride,
                     f_color_row + x * color_pixel_stride, color_pixel_stride);
              if (has_extra_channel_alpha) {
                memcpy(c_alpha_row + x * alpha_pixel_stride,
                       f_alpha_row + x * alpha_pixel_stride,
                       alpha_pixel_stride);
              }
            }
          }
        }
      } else {
        curr_frame.frame_info.layer_info.blend_info.blendmode =
            JXL_BLEND_REPLACE;
        JXL_ASSIGN_OR_RETURN(canvas, curr_frame.Copy());
      }
      continue;
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
      const size_t cy = y0 + y;
      const uint8_t* f_color_row =
          static_cast<const uint8_t*>(curr_frame.color.pixels()) +
          cy * curr_frame.color.stride;
      uint8_t* dst_color_row = static_cast<uint8_t*>(cropped.color.pixels()) +
                               y * cropped.color.stride;

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
        const size_t cx = x0 + x;
        bool diff = can_use_blend ? (diff_mask[cy * xsize + cx] != 0) : true;
        if (!diff) {
          int matching_neighbors = 0;
          int total_neighbors = 0;
          if (x > 0) {
            ++total_neighbors;
            if (!diff_mask[cy * xsize + (cx - 1)]) ++matching_neighbors;
          }
          if (x + 1 < w) {
            ++total_neighbors;
            if (!diff_mask[cy * xsize + (cx + 1)]) ++matching_neighbors;
          }
          if (y > 0) {
            ++total_neighbors;
            if (!diff_mask[(cy - 1) * xsize + cx]) ++matching_neighbors;
          }
          if (y + 1 < h) {
            ++total_neighbors;
            if (!diff_mask[(cy + 1) * xsize + cx]) ++matching_neighbors;
          }
          if (total_neighbors - matching_neighbors > 1) {
            diff = true;
          }
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
        const size_t ec_stride_bytes =
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
        (x0 != 0 || y0 != 0 || w != xsize || h != ysize);
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
