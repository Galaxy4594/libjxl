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

// Performs delta optimization, temporal deduplication, and timebase
// normalization on animated frames in `ppf`. This includes:
// 1. Consecutive identical frame deduplication (folding durations).
// 2. Loop-boundary frame deduplication (last frame == first frame).
// 3. Timebase and frame duration GCD reduction (minimizing header varints).
// 4. Dirty transparent pixel clearing (optim_dirty).
// 5. Minimal bounding-box cropping with delta reference blending.
Status OptimizeAnimation(PackedPixelFile* ppf);

}  // namespace extras
}  // namespace jxl

#endif  // LIB_EXTRAS_OPTIMIZE_ANIMATION_H_
