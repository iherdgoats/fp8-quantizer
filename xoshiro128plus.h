/*----------------------------------------------------------------------------*
 * Copyright (c) 2023, Intel Corporation - All rights reserved.
 * Copyright (c) 2023, iherdgoats <goatherder@mailbox.org>.
 *
 * This file was originally part of FP8-Emulation-Toolkit. But it has been
 * heavily modified by iherdgoats in order to make it AVX2 compatible. The main
 * reason behind the choice to use the xoshiro128+ generator is performance.
 *
 * Written originally in 2018 by David Blackman and Sebastiano Vigna
 * (vigna@acm.org)
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)
 * iherdgoats <goatherder@mailbox.org>
 *----------------------------------------------------------------------------*/

#ifndef _XOSHIRO128PLUS_H_
#define _XOSHIRO128PLUS_H_

#include <immintrin.h>
#include <stdint.h>

/*
 * This is xoshiro128+ 1.0, our best and fastest 32-bit generator for 32-bit
 * floating-point numbers. We suggest to use its upper bits for
 * floating-point generation, as it is slightly faster than xoshiro128**.
 * It passes all tests we are aware of except for linearity tests, as the
 * lowest four bits have low linear complexity, so if low linear complexity is
 * not considered an issue (as it is usually the case) it can be used to
 * generate 32-bit outputs, too.
 *
 * We suggest to use a sign test to extract a random Boolean value, and
 * right shifts to extract subsets of bits.
 */

static inline uint32_t rotl(const uint32_t x, int k) {
  return (x << k) | (x >> (32 - k));
}

static inline uint32_t rand_xoshiro128plus_scalar(uint32_t *s) {
  const uint32_t result = s[0] + s[3];

  const uint32_t t = s[1] << 9;

  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];

  s[2] ^= t;

  s[3] = rotl(s[3], 11);

  return result;
}

// Effectively eight generators in parallel.
static inline __m256i _mm256_rand_xoshiro128plus_epi32(uint32_t *vs0,
                                                       uint32_t *vs1,
                                                       uint32_t *vs2,
                                                       uint32_t *vs3) {
  const __m256i vrplus = _mm256_add_epi32(_mm256_load_si256((__m256i *)vs0),
                                          _mm256_load_si256((__m256i *)vs3));
  const __m256i vt =
      _mm256_sll_epi32(_mm256_load_si256((__m256i *)vs1), _mm_cvtsi32_si128(9));

  _mm256_store_si256((__m256i *)vs2,
                     _mm256_xor_si256(_mm256_load_si256((__m256i *)vs2),
                                      _mm256_load_si256((__m256i *)vs0)));
  _mm256_store_si256((__m256i *)vs3,
                     _mm256_xor_si256(_mm256_load_si256((__m256i *)vs3),
                                      _mm256_load_si256((__m256i *)vs1)));
  _mm256_store_si256((__m256i *)vs1,
                     _mm256_xor_si256(_mm256_load_si256((__m256i *)vs1),
                                      _mm256_load_si256((__m256i *)vs2)));
  _mm256_store_si256((__m256i *)vs0,
                     _mm256_xor_si256(_mm256_load_si256((__m256i *)vs0),
                                      _mm256_load_si256((__m256i *)vs3)));
  _mm256_store_si256((__m256i *)vs2,
                     _mm256_xor_si256(_mm256_load_si256((__m256i *)vs2), vt));

  __m256i vl = _mm256_slli_epi32(_mm256_load_si256((__m256i *)vs3), 11);
  __m256i vr = _mm256_srli_epi32(_mm256_load_si256((__m256i *)vs3), 32 - 11);

  _mm256_store_si256((__m256i *)vs3, _mm256_or_si256(vl, vr));

  return vrplus;
}

#endif // _XOSHIRO128PLUS_H_