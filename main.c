/*----------------------------------------------------------------------------*
 * Copyright (c) 2023, iherdgoats <goatherder@mailbox.org>.
 *
 * This file was originally part of FP8-Emulation-Toolkit. But it has been
 * heavily modified by iherdgoats in order to make it AVX2 compatible.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * iherdgoats <goatherder@mailbox.org>
 *----------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include "fp8_quant.h"

int main() {
  float in[100];
  float out[100];

  for (int i = 0; i < 100; i++) {
    in[i] = (float)rand() / (float)RAND_MAX;
    out[i] = 0.0;
  }

  quantize_to_fp8(in, out, 100, 1.0, false, 0);

  for (int i = 0; i < 100; i++) {
    printf("%f = ", in[i]);
    printf("%f\n", out[i]);
  }

  return 0;
}