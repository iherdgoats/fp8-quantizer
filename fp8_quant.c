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

#include <immintrin.h>

#include "fp8_quant.h"
#include "xoshiro128plus.h"

static uint32_t s1_[4] = {1387366120, 2798441831, 888998500, 1099633400};
static uint32_t s2_[4] = {2034269327, 2125325156, 1209715489, 1931656721};
static uint32_t s3_[4] = {1555452618, 650181557, 883695203, 627677842};
static uint32_t s4_[4] = {4195248041, 2146478152, 480059239, 1468956197};
static uint32_t s5_[4] = {1252084877, 500390994, 977516591, 1950666000};
static uint32_t s6_[4] = {3936597502, 834151069, 1477014702, 734008143};
static uint32_t s7_[4] = {1983400973, 1164103095, 2110188261, 2019272068};
static uint32_t s8_[4] = {1877096364, 2833629967, 4196320416, 1774181187};
static uint32_t s9_[4] = {702309618, 4077815558, 1512057936, 1868769368};
static uint32_t s10_[4] = {510001215, 966559856, 776583255, 1475621065};
static uint32_t s11_[4] = {1271806057, 1881312534, 478635452, 814821902};
static uint32_t s12_[4] = {733990058, 1889991804, 1108257970, 1093480892};
static uint32_t s13_[4] = {4273743809, 4167473370, 558000409, 1594848927};
static uint32_t s14_[4] = {444870959, 1595722866, 1064124488, 3637102547};
static uint32_t s15_[4] = {703721499, 3896407831, 1002360059, 1427395742};
static uint32_t s16_[4] = {1295231497, 1254972431, 1423497865, 861918264};

/* seed pointer array */
static uint32_t *sptr_[16] = {s1_, s2_,  s3_,  s4_,  s5_,  s6_,  s7_,  s8_,
                              s9_, s10_, s11_, s12_, s13_, s14_, s15_, s16_};

float __half2float(uint16_t h_val) { return _cvtsh_ss(h_val); }

uint16_t __float2half_rn(float inval) {
  return _cvtss_sh(inval, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

void cvt_fp32_fp8_stochastic_intrinsic(const float *__restrict__ in, float *out,
                                       int size, float scale) {
  uint32_t vs0[8] __attribute__((aligned(32))) = {
      1387366120, 279844183, 888998500, 1099633400,
      1252084877, 500390994, 977516591, 1950666000};
  uint32_t vs1[8] __attribute__((aligned(32))) = {
      2034269327, 2125325156, 1209715489, 193165672,
      187709636,  28336299,   419632041,  1774181187};
  uint32_t vs2[8] __attribute__((aligned(32))) = {
      1555452618, 650181557,  883695203, 62767784,
      127180605,  1881312534, 478635452, 814821902};
  uint32_t vs3[8] __attribute__((aligned(32))) = {
      419524804, 2146478152, 480059239,  1468956197,
      444870959, 1595722866, 1064124488, 363710254};

#pragma omp parallel for firstprivate(vs0, vs1, vs2, vs3)
  for (int i = 0; i < size; i += 16) {
    const __m256i vnaninf = _mm256_set1_epi16(0x7c00);
    const __m256i vfixup = _mm256_set1_epi16(0x0001);
    const __m256i vfixupmask = _mm256_set1_epi16(0x0100);
    const __m256i vrneadd = _mm256_set1_epi16(0x007f);
    const __m256i vdenorm = _mm256_set1_epi16(0x03ff);
    const __m256i vexmant = _mm256_set1_epi16(0x7fff);

    __m256i rnd256 = _mm256_rand_xoshiro128plus_epi32(vs0, vs1, vs2, vs3);
    __m128i rnbits = _mm256_extracti32x4_epi32(rnd256, 0);

    __m256 s_ = _mm256_set1_ps(scale);
    __m256 sr_ = _mm256_set1_ps(1.0 / scale);

    __m256 b = _mm256_loadu_ps(&in[i]);
    __m256 a = _mm256_loadu_ps(&in[i + 8]);

    b = _mm256_mul_ps(b, s_);
    a = _mm256_mul_ps(a, s_);

    __m128i ah_ =
        _mm256_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m128i bh_ =
        _mm256_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

    const __m256i a_ =
        _mm256_insertf128_si256(_mm256_castsi128_si256(bh_), ah_, 1);

    const __m256i maska1_ =
        _mm256_cmpeq_epi16(_mm256_and_si256(a_, vnaninf), vnaninf);
    const __m256i maska2_ =
        _mm256_cmpeq_epi16(_mm256_and_si256(a_, vfixupmask), vfixupmask);
    const __m256i maska4_ =
        _mm256_cmpgt_epi16(vdenorm, _mm256_and_si256(a_, vexmant));

    __m256i a_sr_ = _mm256_blendv_epi8(
        a_, _mm256_add_epi16(a_, _mm256_cvtepu8_epi16(rnbits)),
        _mm256_andnot_si256(maska4_, maska1_));

    a_sr_ = _mm256_blendv_epi8(a_sr_, _mm256_add_epi16(a_sr_, vrneadd),
                               _mm256_and_si256(maska4_, maska2_));

    a_sr_ = _mm256_slli_epi16(_mm256_srli_epi16(a_sr_, 8), 8);

    bh_ = _mm256_extracti128_si256(a_sr_, 0);
    ah_ = _mm256_extracti128_si256(a_sr_, 1);

    b = _mm256_cvtph_ps(bh_);
    a = _mm256_cvtph_ps(ah_);

    _mm256_storeu_ps(&out[i], _mm256_mul_ps(b, sr_));
    _mm256_storeu_ps(&out[i + 8], _mm256_mul_ps(a, sr_));
  }
}

void cvt_fp32_fp8_stochastic_scalar(const float *__restrict__ in, float *out,
                                    int size, float scale) {
  int non_mant_bits = 5 /*exp_bits */ + 1; /* exponent + sign */
  int lshift = 10 - (8 /*mbits */ - non_mant_bits);

  uint16_t mask_mant = (uint16_t)(0xFFFF << lshift);
  uint16_t grs_bitmask = 0x00FF;
  uint16_t rne_tie = 0x0180;

  float scale_reciprocal = 1.0 / scale;

  for (int gid = 0; gid < size; gid++) {
    __half_t h;
    float inval = scale * in[gid];

    h.u = __float2half_rn(inval);

    uint16_t can_round = ((h.u & 0x7F00) <= 0x7B00) ? 1 : 0;
    uint16_t is_normal =
        (((h.u & 0x7C00) <= 0x7800) && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
    uint16_t is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
    uint16_t is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

    /* nearest rounding masks */
    uint16_t rnmask = (h.u & grs_bitmask);
    uint16_t rnmask_tie = (h.u & rne_tie);

    if (is_naninf == 0) {
      /* stochastic with 16 seeds */
      int seed_index = (gid / 16);
      uint16_t rand =
          (uint16_t)(rand_xoshiro128plus_scalar(sptr_[(seed_index % 16)]));
      /* apply stochastic rounding before truncation */
      h.u += can_round * is_normal * (rand & 0xFF);
      /* stochastic round:  denormals --> rne rounding */
      h.u += can_round * is_denorm *
             (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
    }
    /* truncation */
    h.u = (h.u & mask_mant);
    float f_ = __half2float(h.u);
    out[gid] = f_ * scale_reciprocal;
  }
}

void quantize_to_fp8(const float *__restrict__ in, float *__restrict__ out,
                     const int size, const float in_scale, bool block_norm,
                     int block_size) {
  float scale = in_scale;

  if (block_norm == true) {
    int nblocks = (size + (block_size - 1)) / block_size;

#pragma omp parallel for
    for (int b = 0; b < nblocks; b++) {
      int start_index = (b * block_size);
      /* handle the last block */
      if (start_index + block_size > size)
        block_size = (size - start_index);

      float scale = fp8_scale(&in[start_index], block_size);

      if ((block_size % 32) == 0) {
        cvt_fp32_fp8_stochastic_intrinsic(&in[start_index], &out[start_index],
                                          block_size, scale);
      } else {
        cvt_fp32_fp8_stochastic_scalar(&in[start_index], &out[start_index],
                                       block_size, scale);
      }
    }
  } else {
    if ((size % 32) == 0) {
      cvt_fp32_fp8_stochastic_intrinsic(in, out, size, scale);
    } else {
      int vec_size = ((int)(size / 32)) * 32;

      if (vec_size > 0) {
        cvt_fp32_fp8_stochastic_intrinsic(in, out, vec_size, scale);
      }

      cvt_fp32_fp8_stochastic_scalar(&in[vec_size], &out[vec_size],
                                     size - vec_size, scale);
    }
  }
}