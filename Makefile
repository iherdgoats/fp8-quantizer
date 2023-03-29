CC := clang
CFLAGS := -mavx2 -mf16c -fopenmp -O3

SRC := xoshiro128plus.h fp8_quant.c fp8_quant.h main.c

fp8_quant: $(SRC)
	$(CC) $(CFLAGS) -o fp8_quant fp8_quant.c main.c $(LDFLAGS)

format:
	clang-format -i $(SRC)

clean:
	-rm -f fp8_quant

.PHONY: format clean