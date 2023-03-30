/*----------------------------------------------------------------------------*
 * Copyright (c) 2023, Intel Corporation - All rights reserved.
 * Copyright (c) 2023, iherdgoats <goatherder@mailbox.org>.
 *
 * This file was originally part of FP8-Emulation-Toolkit. But it has been
 * heavily modified by iherdgoats in order to make it AVX2 compatible.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)
 * iherdgoats <goatherder@mailbox.org>
 *----------------------------------------------------------------------------*/

#ifndef FP8_QUANT_H
#define FP8_QUANT_H

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>

typedef union {
  uint16_t u;
  __fp16 f;
} __half_t;

typedef union {
  uint32_t u;
  float f;
} __float_t;

/**
 * @brief Computes an 8-bit floating point scale factor for the input array.
 *
 * @param in        Pointer to the input array of 32-bit floating point values.
 * @param size      Number of elements in the input array.
 */
static inline float fp8_scale(const float *__restrict__ in, const int size) {
  const int32_t fp8_exp_bias = 15; // (2^(5-1)) - 1
  const int32_t fp8_max_exp =
      30; // (2^5) - 2 (excluding subnormals and infinities)

  float max_val = 0.0;

#pragma omp parallel for reduction(max : max_val)
  for (int i = 0; i < size; i++) {
    max_val = (max_val < fabs(in[i])) ? fabs(in[i]) : max_val;
  }

  __float_t f;
  f.f = max_val;

  const int32_t max_exp = ((f.u >> 23) & 0xFF) - 127;

  const int32_t scale_exp = max_exp - (fp8_max_exp - fp8_exp_bias - 1);

  f.u = (127 - scale_exp) << 23;

  return f.f;
}

/**
 * @brief Quantizes 32-bit floating point values into 8-bit floats (5-bit
 * exponent, 2-bit mantissa).
 *
 * @param in        Pointer to the input array of 32-bit floating point values.
 * @param out       Pointer to the output array of 32-bit floating point values
 * representing quantized 8-bit floats.
 * @param size      Number of elements in the input array.
 * @param in_scale  Scale factor to apply to the input array before
 * quantization.
 * @param block_norm If true, normalizes the input data in blocks of size
 * 'block_size' before quantization.
 * @param block_size Size of the blocks for block normalization.
 */
void quantize_to_fp8(const float *__restrict__ in, float *__restrict__ out,
                     const int size, const float in_scale, bool block_norm,
                     int block_size);

/**
 * @brief Computes the 8-bit floating point scale factor for the input array.
 *
 * @param in        Pointer to the input array of 32-bit floating point values.
 * @param size      Number of elements in the input array.
 */
float fp8_scale(const float *__restrict__ in, const int size);

#endif // FP8_QUANT_H