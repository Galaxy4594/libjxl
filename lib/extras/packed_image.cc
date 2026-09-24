// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper class for storing external (int or float, interleaved) images. This is
// the common format used by other libraries and in the libjxl API.

#include "packed_image.h"

#include <jxl/codestream_header.h>
#include <jxl/encode.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

// Class representing an interleaved image with a bunch of channels.
StatusOr<PackedImage> PackedImage::Create(size_t xsize, size_t ysize,
                                          const JxlPixelFormat& format) {
  JXL_ASSIGN_OR_RETURN(size_t stride, CalcStride(format, xsize));
  size_t pixels_size = ysize * stride;
  if ((pixels_size / stride) != ysize) {
    return JXL_FAILURE("Image too big");
  }
  PackedImage image(xsize, ysize, format, stride);
  if (!image.pixels()) {
    // TODO(szabadka): use specialized OOM error code
    return JXL_FAILURE("Failed to allocate memory for image");
  }
  return image;
}

StatusOr<PackedImage> PackedImage::Copy() const {
  // Resulting copy_stride have to be less or equal to original -> always ok.
  JXL_ASSIGN_OR_RETURN(size_t copy_stride, CalcStride(format, xsize));
  PackedImage copy(xsize, ysize, format, copy_stride);
  const uint8_t* orig_pixels = reinterpret_cast<const uint8_t*>(pixels());
  uint8_t* copy_pixels = reinterpret_cast<uint8_t*>(copy.pixels());
  if (stride == copy_stride) {
    // Same stride -> copy in one go.
    memcpy(copy_pixels, orig_pixels, ysize * stride);
  } else {
    // Otherwise, copy row-wise.
    JXL_DASSERT(copy_stride < stride);
    for (size_t y = 0; y < ysize; ++y) {
      memcpy(copy_pixels + y * copy_stride, orig_pixels + y * stride,
             copy_stride);
    }
  }
  return copy;
}

Status PackedImage::ValidateDataType(JxlDataType data_type) {
  if ((data_type != JXL_TYPE_UINT8) && (data_type != JXL_TYPE_UINT16) &&
      (data_type != JXL_TYPE_FLOAT) && (data_type != JXL_TYPE_FLOAT16)) {
    return JXL_FAILURE("Unhandled data type: %d", static_cast<int>(data_type));
  }
  return true;
}

size_t PackedImage::BitsPerChannel(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_UINT8:
      return 8;
    case JXL_TYPE_UINT16:
      return 16;
    case JXL_TYPE_FLOAT:
      return 32;
    case JXL_TYPE_FLOAT16:
      return 16;
    default:
      JXL_DEBUG_ABORT("Unreachable");
      return 0;
  }
}

// Logical resize; use Copy() for storage reallocation, if necessary.
Status PackedImage::ShrinkTo(size_t new_xsize, size_t new_ysize) {
  if (new_xsize > xsize || new_ysize > ysize) {
    return JXL_FAILURE("Cannot shrink PackedImage to a larger size");
  }
  xsize = new_xsize;
  ysize = new_ysize;
  return true;
}

PackedImage::PackedImage(size_t xsize, size_t ysize,
                         const JxlPixelFormat& format, size_t stride)
    : xsize(xsize),
      ysize(ysize),
      stride(stride),
      format(format),
      pixels_size(ysize * stride),
      pixels_(malloc(std::max<size_t>(1, pixels_size)), free) {
  bytes_per_channel_ = BitsPerChannel(format.data_type) / jxl::kBitsPerByte;
  pixel_stride_ = format.num_channels * bytes_per_channel_;
  swap_endianness_ = SwapEndianness(format.endianness);
}

StatusOr<size_t> PackedImage::CalcStride(const JxlPixelFormat& format,
                                         size_t xsize) {
  size_t multiplier = (BitsPerChannel(format.data_type) * format.num_channels /
                       jxl::kBitsPerByte);
  size_t stride;
  if (!SafeMul(xsize, multiplier, stride)) {
    return JXL_FAILURE("Image too big");
  }
  if (!SafeRoundUpTo(stride, format.align, stride)) {
    return JXL_FAILURE("Image too big");
  }
  return stride;
}

PackedFrame::PackedFrame(PackedImage&& image) : color(std::move(image)) {}

PackedFrame::PackedFrame(PackedFrame&& other) = default;

PackedFrame& PackedFrame::operator=(PackedFrame&& other) = default;

PackedFrame::~PackedFrame() = default;

StatusOr<PackedFrame> PackedFrame::Create(size_t xsize, size_t ysize,
                                          const JxlPixelFormat& format) {
  JXL_ASSIGN_OR_RETURN(PackedImage image,
                       PackedImage::Create(xsize, ysize, format));
  PackedFrame frame(std::move(image));
  return frame;
}

StatusOr<PackedFrame> PackedFrame::Copy() const {
  JXL_ASSIGN_OR_RETURN(
      PackedFrame copy,
      PackedFrame::Create(color.xsize, color.ysize, color.format));
  copy.frame_info = frame_info;
  copy.name = name;
  JXL_ASSIGN_OR_RETURN(copy.color, color.Copy());
  for (const auto& ec : extra_channels) {
    JXL_ASSIGN_OR_RETURN(PackedImage ec_copy, ec.Copy());
    copy.extra_channels.emplace_back(std::move(ec_copy));
  }
  return copy;
}

// Logical resize; use Copy() for storage reallocation, if necessary.
Status PackedFrame::ShrinkTo(size_t new_xsize, size_t new_ysize) {
  JXL_RETURN_IF_ERROR(color.ShrinkTo(new_xsize, new_ysize));
  for (auto& ec : extra_channels) {
    JXL_RETURN_IF_ERROR(ec.ShrinkTo(new_xsize, new_ysize));
  }
  frame_info.layer_info.xsize = new_xsize;
  frame_info.layer_info.ysize = new_ysize;
  return true;
}

ChunkedPackedFrame::ChunkedPackedFrame(
    size_t xsize, size_t ysize,
    std::function<JxlChunkedFrameInputSource()> get_input_source)
    : xsize(xsize),
      ysize(ysize),
      get_input_source_(std::move(get_input_source)) {
  const auto input_source = get_input_source_();
  input_source.get_color_channels_pixel_format(input_source.opaque, &format);
}

PackedPixelFile::PackedPixelFile() { JxlEncoderInitBasicInfo(&info); };

Status PackedPixelFile::ShrinkTo(size_t new_xsize, size_t new_ysize) {
  for (auto& frame : frames) {
    JXL_RETURN_IF_ERROR(frame.ShrinkTo(new_xsize, new_ysize));
  }
  info.xsize = new_xsize;
  info.ysize = new_ysize;
  return true;
}

Status OptimizeAnimation(PackedPixelFile* ppf) {
  if (!ppf || !ppf->info.have_animation || ppf->frames.size() <= 1) {
    return true;
  }
  const size_t W = ppf->info.xsize;
  const size_t H = ppf->info.ysize;
  if (W == 0 || H == 0) return true;

  // Helper lambda to compare two frames for exact equality
  auto FramesEqual = [](const PackedFrame& f1, const PackedFrame& f2) -> bool {
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
  };

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

  // 1b. Deduplicate identical frame at looping animation boundary (last frame == first frame)
  if (ppf->frames.size() > 1 &&
      (ppf->info.animation.num_loops == 0 || ppf->info.animation.num_loops == 1 ||
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

  // 2. Check if all frames are full canvas
  bool all_full_canvas = true;
  for (const auto& frame : ppf->frames) {
    if (frame.frame_info.layer_info.have_crop != 0 ||
        frame.frame_info.layer_info.crop_x0 != 0 ||
        frame.frame_info.layer_info.crop_y0 != 0 ||
        frame.color.xsize != W || frame.color.ysize != H) {
      all_full_canvas = false;
      break;
    }
  }
  if (!all_full_canvas) {
    ppf->frames.back().frame_info.is_last = JXL_TRUE;
    return true;
  }

  const bool has_extra_channel_alpha_initial =
      !ppf->frames[0].extra_channels.empty();
  const bool has_interleaved_alpha_initial =
      (!has_extra_channel_alpha_initial &&
       ppf->frames[0].color.format.num_channels >
           ppf->info.num_color_channels);

  // If there is no alpha channel at all, add an alpha extra channel
  // so that delta frames can use JXL_BLEND_BLEND (delta zeroing) instead of
  // being forced to store unchanged pixels within the bounding box with JXL_BLEND_REPLACE.
  if (!has_extra_channel_alpha_initial && !has_interleaved_alpha_initial &&
      ppf->info.num_color_channels == 3 &&
      ppf->frames[0].color.format.data_type == JXL_TYPE_UINT8) {
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

    for (size_t f = 0; f < ppf->frames.size(); ++f) {
      JXL_ASSIGN_OR_RETURN(PackedImage f_alpha,
                           PackedImage::Create(W, H, alpha_format));
      memset(f_alpha.pixels(), 255, f_alpha.pixels_size);
      ppf->frames[f].extra_channels.emplace_back(std::move(f_alpha));
    }
  }

  // 2b. Clean dirty transparent pixels (where alpha == 0, zero out color channels).
  // Matches apngopt's optim_dirty() and libwebp's WebPCleanupTransparentArea().
  // When a pixel is 100% transparent, whatever RGB color is stored underneath is
  // completely invisible, but residual/palette color noise wastes entropy and
  // causes artificial size discrepancies between different input containers.
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

  // 3. Configure frame 0 to save as reference
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
       ppf->frames[0].color.format.num_channels >
           ppf->info.num_color_channels);

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
    alpha_pixel_stride =
        ppf->frames[0].extra_channels[0].format.num_channels *
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
          has_extra_channel_alpha
              ? (static_cast<const uint8_t*>(
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
              static_cast<const uint8_t*>(
                  canvas.extra_channels[0].pixels()) +
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

    JXL_ASSIGN_OR_RETURN(
        PackedFrame cropped,
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
      uint8_t* dst_color_row =
          static_cast<uint8_t*>(cropped.color.pixels()) +
          y * cropped.color.stride;

      const uint8_t* c_alpha_row =
          has_extra_channel_alpha
              ? (static_cast<const uint8_t*>(
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
              ? (static_cast<uint8_t*>(
                     cropped.extra_channels[0].pixels()) +
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
                   f_alpha_row + cx * alpha_pixel_stride,
                   alpha_pixel_stride);
          }
        }
      }
      for (size_t ec = 1; ec < curr_frame.extra_channels.size(); ++ec) {
        const auto& src_ec = curr_frame.extra_channels[ec];
        auto& dst_ec = cropped.extra_channels[ec];
        size_t ec_stride_bytes =
            src_ec.format.num_channels *
            (PackedImage::BitsPerChannel(src_ec.format.data_type) / 8);
        const uint8_t* src_row =
            static_cast<const uint8_t*>(src_ec.pixels()) +
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

