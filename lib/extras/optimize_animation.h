// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_EXTRAS_OPTIMIZE_ANIMATION_H_
#define LIB_EXTRAS_OPTIMIZE_ANIMATION_H_

#include "lib/extras/packed_image.h"
#include "lib/jxl/base/status.h"

namespace jxl {
namespace extras {

// Deduplicates identical frames, simplifies frame duration ticks, and crops
// frames to bounding-box deltas with reference blending where beneficial.
Status OptimizeAnimation(PackedPixelFile* ppf);

}  // namespace extras
}  // namespace jxl

#endif  // LIB_EXTRAS_OPTIMIZE_ANIMATION_H_
