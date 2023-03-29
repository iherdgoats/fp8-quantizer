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

#include <stdbool.h>

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

#endif // FP8_QUANT_H