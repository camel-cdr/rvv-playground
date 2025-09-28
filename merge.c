/*
 * Merge sortest arrays.
 * The code is optimized for similarly sized arrays with lots of overlap.
 */

#include <riscv_vector.h>
#include <string.h>
#include <stdio.h>

#if __GNUC__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) x
#define unlikely(x) x
#endif

void
merge_scalar(uint32_t *restrict dst, uint32_t const *restrict src1, uint32_t const *restrict src2, size_t n1, size_t n2)
{
	uint32_t const *end1 = src1+n1, *end2 = src2+n2;
	for (; src1 != end1 && src2 != end2; ) {
		uint32_t s1 = *src1, s2 = *src2;
		int lt = s1 < s2;
		*dst++ = lt ? s1 : s2;
		src1 += lt;
		src2 += !lt;
	}
	memcpy(dst, src1 != end1 ? src1 : src2, ((end1-src1) + (end2-src2)) * sizeof *dst);
}

static inline vuint32m8_t
rvv_bitonic_merge_u32m4(vuint32m4_t v0 /* ascending */, vuint32m4_t v1 /* descending */)
{
	size_t VL1 = __riscv_vsetvlmax_e32m1();
	size_t VL4 = __riscv_vsetvlmax_e32m4();
	vuint32m1_t l0, l1, l2, l3, h0, h1, h2, h3;
	vuint32m1_t L0, L1, L2, L3, H0, H1, H2, H3;
	l0 = __riscv_vget_u32m1(v0, 0); l1 = __riscv_vget_u32m1(v0, 1);
	l2 = __riscv_vget_u32m1(v0, 2); l3 = __riscv_vget_u32m1(v0, 3);
	h0 = __riscv_vget_u32m1(v1, 0); h1 = __riscv_vget_u32m1(v1, 1);
	h2 = __riscv_vget_u32m1(v1, 2); h3 = __riscv_vget_u32m1(v1, 3);

	L0 = __riscv_vminu(l0, h0, VL1); L1 = __riscv_vminu(l1, h1, VL1);
	L2 = __riscv_vminu(l2, h2, VL1); L3 = __riscv_vminu(l3, h3, VL1);
	H0 = __riscv_vmaxu(l0, h0, VL1); H1 = __riscv_vmaxu(l1, h1, VL1);
	H2 = __riscv_vmaxu(l2, h2, VL1); H3 = __riscv_vmaxu(l3, h3, VL1);
	l0 = L0; l1 = L1; l2 = L2; l3 = L3;
	h0 = H0; h1 = H1; h2 = H2; h3 = H3;

	L0 = __riscv_vminu(l0, h0, VL1); L1 = __riscv_vminu(l1, h1, VL1);
	L2 = __riscv_vmaxu(l0, h0, VL1); L3 = __riscv_vmaxu(l1, h1, VL1);
	H0 = __riscv_vminu(l2, h2, VL1); H1 = __riscv_vminu(l3, h3, VL1);
	H2 = __riscv_vmaxu(l2, h2, VL1); H3 = __riscv_vmaxu(l3, h3, VL1);
	l0 = L0; l1 = L1; l2 = L2; l3 = L3;
	h0 = H0; h1 = H1; h2 = H2; h3 = H3;

	L0 = __riscv_vminu(l0, h0, VL1); L1 = __riscv_vmaxu(l0, h0, VL1);
	L2 = __riscv_vminu(l2, h2, VL1); L3 = __riscv_vmaxu(l2, h2, VL1);
	H0 = __riscv_vminu(l1, h1, VL1); H1 = __riscv_vmaxu(l1, h1, VL1);
	H2 = __riscv_vminu(l3, h3, VL1); H3 = __riscv_vmaxu(l3, h3, VL1);
	l0 = L0; l1 = L1; l2 = L2; l3 = L3;
	h0 = H0; h1 = H1; h2 = H2; h3 = H3;

#if 1
	vuint32m1_t vid = __riscv_vid_v_u32m1(VL1);
	for (size_t p2 = VL1>>1; p2; p2 >>= 1) {
		vbool32_t m = __riscv_vmsne(__riscv_vand(vid, p2, VL1), 0, VL1);

		L0 = __riscv_vminu(l0, h0, VL1); L1 = __riscv_vminu(l1, h1, VL1);
		L2 = __riscv_vminu(l2, h2, VL1); L3 = __riscv_vminu(l3, h3, VL1);
		H0 = __riscv_vmaxu(l0, h0, VL1); H1 = __riscv_vmaxu(l1, h1, VL1);
		H2 = __riscv_vmaxu(l2, h2, VL1); H3 = __riscv_vmaxu(l3, h3, VL1);
		l0 = L0; l1 = L1; l2 = L2; l3 = L3;
		h0 = H0; h1 = H1; h2 = H2; h3 = H3;

		l0 = __riscv_vslideup_mu(  m, l0, H0, p2, VL1);
		l1 = __riscv_vslideup_mu(  m, l1, H1, p2, VL1);
		l2 = __riscv_vslideup_mu(  m, l2, H2, p2, VL1);
		l3 = __riscv_vslideup_mu(  m, l3, H3, p2, VL1);
		m = __riscv_vmnot(m, VL1);
		h0 = __riscv_vslidedown_mu(m, h0, L0, p2, VL1);
		h1 = __riscv_vslidedown_mu(m, h1, L1, p2, VL1);
		h2 = __riscv_vslidedown_mu(m, h2, L2, p2, VL1);
		h3 = __riscv_vslidedown_mu(m, h3, L3, p2, VL1);
	}

	v0 = __riscv_vcreate_v_u32m1_u32m4(l0, l1, l2, l3);
	v1 = __riscv_vcreate_v_u32m1_u32m4(h0, h1, h2, h3);
#elif 0
	/* LMUL=4 version, sadly needs to use LMUL=4 to compute the mask */
	v0 = __riscv_vcreate_v_u32m1_u32m4(l0, l1, l2, l3);
	v1 = __riscv_vcreate_v_u32m1_u32m4(h0, h1, h2, h3);
	vuint16m2_t vid = __riscv_vid_v_u16m2(VL4);
	vuint32m4_t vmin, vmax;
	vbool8_t m;

	for (size_t p2 = VL1>>1; p2; p2 >>= 1) {
		vmin = __riscv_vminu(v0, v1, VL4);
		vmax = __riscv_vmaxu(v0, v1, VL4);
		v0 = vmin; v1 = vmax;
		m = __riscv_vmsne(__riscv_vand(vid, p2, VL4), 0, VL4);
		v0 = __riscv_vslideup_mu(  m, v0, vmax, p2, VL4);
		m = __riscv_vmnot(m, VL4);
		v1 = __riscv_vslidedown_mu(m, v1, vmin, p2, VL4);
	}
#elif 0
	/* optimized mask computation for LMUL=4 version */
	v0 = __riscv_vcreate_v_u32m1_u32m4(l0, l1, l2, l3);
	v1 = __riscv_vcreate_v_u32m1_u32m4(h0, h1, h2, h3);
	vuint16m2_t vid = __riscv_vid_v_u16m2(VL4);
	vuint32m4_t vmin, vmax;
	vbool8_t m;

	size_t p2 = VL1>>1;
	for (; unlikely(p2 > 32); p2 >>= 1) {
		/* slow path, only entered by VLEN >= 4096 */
		vmin = __riscv_vminu(v0, v1, VL4);
		vmax = __riscv_vmaxu(v0, v1, VL4);
		v0 = vmin; v1 = vmax;
		m = __riscv_vmsne(__riscv_vand(vid, p2, VL4), 0, VL4);
		v0 = __riscv_vslideup_mu(  m, v0, vmax, p2, VL4);
		m = __riscv_vmnot(m, VL4);
		v1 = __riscv_vslidedown_mu(m, v1, vmin, p2, VL4);
	}

	static const uint64_t masks[6] = {
		0xaaaaaaaaaaaaaaaa,
		0xcccccccccccccccc,
		0xf0f0f0f0f0f0f0f0,
		0xff00ff00ff00ff00,
		0xffff0000ffff0000,
		0xffffffff00000000,
	};
	size_t midx = __builtin_ctz(VL1);
	size_t VLM = __riscv_vsetvlmax_e64m1();

	for (; p2; p2 >>= 1) {
		vmin = __riscv_vminu(v0, v1, VL4);
		vmax = __riscv_vmaxu(v0, v1, VL4);
		v0 = vmin; v1 = vmax;
		m = __riscv_vreinterpret_b8(__riscv_vmv_v_x_u64m1(masks[--midx], VLM));
		v0 = __riscv_vslideup_mu(  m, v0, vmax, p2, VL4);
		m = __riscv_vmnot(m, VL4);
		v1 = __riscv_vslidedown_mu(m, v1, vmin, p2, VL4);
	}

#endif
	vuint32m4_t t = v0;
	v0 = __riscv_vminu(t, v1, VL4);
	v1 = __riscv_vmaxu(t, v1, VL4);

	/* zip */
	return __riscv_vreinterpret_u32m8(__riscv_vwmaccu(__riscv_vwaddu_vv(v0, v1, VL4), -1, v1, VL4));
}


void
merge_rvv(uint32_t *restrict dst, uint32_t const *restrict src1, uint32_t const *restrict src2, size_t n1, size_t n2)
{
	const size_t VL1 = __riscv_vsetvlmax_e32m1();
	const size_t VL4 = __riscv_vsetvlmax_e32m4();
	uint32_t *end0 = dst+n1+n2;
	uint32_t const *end1 = src1+n1, *end2 = src2+n2;

	if (n1 < VL4+1 || n2 < VL4+1)  {
		merge_scalar(dst, src1, src2, end1-src1, end2-src2);
		return;
	}
	uint32_t const *stop1 = src1 + ((n1-1)&~(VL4-1)), *stop2 = src2 + ((n2-1)&~(VL4-1));

	vuint32m4_t v0, v1;
	vuint32m1_t vrev = __riscv_vrsub(__riscv_vid_v_u32m1(VL1), VL1-1, VL1);
	v0 = __riscv_vle32_v_u32m4(src1, VL4); src1 += VL4;
	int lt = 0;
	do {
		v1 = __riscv_vle32_v_u32m4(lt ? src1 : src2, VL4);
		src1 += lt?VL4:0; src2 += lt?0:VL4;
		lt = src1[0] < src2[0];
		v1 = __riscv_vcreate_v_u32m1_u32m4(
				__riscv_vrgather(__riscv_vget_u32m1(v1, 3), vrev, VL1),
				__riscv_vrgather(__riscv_vget_u32m1(v1, 2), vrev, VL1),
				__riscv_vrgather(__riscv_vget_u32m1(v1, 1), vrev, VL1),
				__riscv_vrgather(__riscv_vget_u32m1(v1, 0), vrev, VL1));
		vuint32m8_t v =  rvv_bitonic_merge_u32m4(v0, v1);
		v0 = __riscv_vget_u32m4(v, 0);
		v1 = __riscv_vget_u32m4(v, 1);
		__riscv_vse32(dst, v0, VL4); dst += VL4;
		v0 = v1;
	} while (src1 < stop1 && src2 < stop2);

	/* three-way merge */
	uint32_t *src0 = end0-VL4;
	__riscv_vse32(src0, v0, VL4);
	for (; src0 != end0 && src1 != end1 && src2 != end2; ) {
		uint32_t s0 = *src0, s1 = *src1, s2 = *src2;
		if (s0 <= s1) {
			if (s0 <= s2) *dst++ = s0, ++src0;
			else          *dst++ = s2, ++src2;
		} else {
			if (s1 <= s2) *dst++ = s1, ++src1;
			else          *dst++ = s2, ++src2;
		}
	}

	/* two-way merge */
	/**/ if (src1 < end1 && src2 < end2) merge_scalar(dst, src1, src2, end1-src1, end2-src2);
	else if (src0 < end0 && src2 < end2) merge_scalar(dst, src0, src2, end0-src0, end2-src2);
	else if (src0 < end0 && src1 < end1) merge_scalar(dst, src0, src1, end0-src0, end1-src1);
}

#if defined(FUZZ)

void
merge_fuzz(uint32_t *restrict dst, uint32_t const *restrict src1, uint32_t const *restrict src2, size_t n1, size_t n2)
{
	uint32_t const *end1 = src1+n1, *end2 = src2+n2;
	for (; src1 != end1 && src2 != end2; ) {
		if (*src1 < *src2) *dst++ = *src1++;
		else               *dst++ = *src2++;
	}
	memcpy(dst, src1 != end1 ? src1 : src2, ((end1-src1) + (end2-src2)) * sizeof *dst);
}

#include <stdio.h>
#include <assert.h>

#define N (65536/32*4*5)
static uint32_t tgt[N*2], dst[N*2], src1[N], src2[N];

int
LLVMFuzzerTestOneInput(const uint8_t *src, size_t size)
{
	memset(dst, 0, sizeof dst);
	memset(tgt, 0, sizeof tgt);
	size_t max = __riscv_vsetvlmax_e32m4()*5;
	if (size < 3 || size > max) return -1;
	size_t n1 = src[0] << 16 | src[1] << 8 | src[2]; src += 3; size -= 3;
	n1 = size ? n1 % (size+1) : 0;
	size_t n2 = size - n1;
	for (size_t i = 0; i < n1; ++i) src1[i] = i ? src1[i-1] + *src++ : *src++;
	for (size_t i = 0; i < n2; ++i) src2[i] = i ? src2[i-1] + *src++ : *src++;
	size_t off = N*2-n1-n2;
	merge_fuzz(tgt+off, src1, src2, n1, n2);
	merge_rvv(dst+off, src1, src2, n1, n2);
	for (size_t i = off; i < off+n1+n2; ++i) {
		if (tgt[i] != dst[i]) {
			printf("src1: "); for (size_t j = off; j < n1; ++j) printf("%u, ", src1[j]); puts("");
			printf("src2: "); for (size_t j = off; j < n1; ++j) printf("%u, ", src2[j]); puts("");
			printf("tgt: "); for (size_t j = off; j < off+n1+n2; ++j) printf("%u ", tgt[j]); puts("");
			printf("dst: "); for (size_t j = off; j < off+n1+n2; ++j) printf("%u ", dst[j]); puts("");
			assert(tgt[i] == dst[i]);
		}
	}
	return 0;
}

#elif defined(TEST)
#include <stdio.h>

int
main(void)
{
	uint32_t src1[] = { 48, 222, 224, 224, 224, 224, 224, 224, 284, 284, 284, 284, 284, 284, 284, 521, 758, 995, 1232, 1469, 1706, 1943, 2180, 2417, 2654, 2891, 3128, 3365, 3602, 3839, 4076, 4313, 4550, 4787, 5024, 5261, 5498, 5735, 5972, 6209, 6446 };
	uint32_t src2[] = {  237, 474, 711, 948, 1185, 1422, 1659, 1896, 1896, 1896, 1896, 1896, 1925, 1925, 1925, 1925, 1925, 1925, 1925, 1925, 1925, 1927, 2180, 2180, 2180, 2180, 2180, 2180, 2410, 2410, 2410, 2477, 2720, 183, 183, 183, 218, 253, 288, 291, 398 };
	size_t n1 = sizeof src1 / sizeof *src1;
	size_t n2 = sizeof src2 / sizeof *src2;
	uint32_t res[(sizeof src1) / sizeof *src1 + (sizeof src2) / sizeof *src2];
	memset(res, 0xFF, sizeof res);
	merge_rvv(res, src1, src2, n1, n2);
	for (size_t i = 0; i < n1+n2; ++i) printf("%u ", res[i]); puts("");
	return 0;
}

#elif defined(BENCH)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

#define SIZE (1<<22)
#define RUNS 32

static int
compare_u32(void const *a, void const *b)
{
	uint32_t A = *(uint32_t*)a, B = *(uint32_t*)b;
	return A < B ? -1 : A > B ? 1 : 0;
}

int
main(void)
{
	RandU64 prng = {123^clock(), 456^(uintptr_t)&prng, 789};

	uint32_t *src1 = malloc(SIZE * sizeof *src1);
	uint32_t *src2 = malloc(SIZE * sizeof *src2);
	uint32_t *dst = malloc(SIZE*2 * sizeof *dst);

	for (size_t i = 0; i < RUNS; ++i) {
		for (size_t j = 0; j < SIZE; ++j) src1[j] = randu64(&prng);
		for (size_t j = 0; j < SIZE; ++j) src2[j] = randu64(&prng);
		qsort(src1, SIZE, sizeof *src1, compare_u32);
		qsort(src2, SIZE, sizeof *src2, compare_u32);

		clock_t beg = clock();
		merge_rvv(dst, src1, src2, SIZE, SIZE);
		double secs = (clock() - beg) * 1.0/CLOCKS_PER_SEC;
		printf("%f secs, %f GiB/s checksum: %u\n", secs, SIZE/2.0/(1024*1024*1024)/secs, dst[randu64(&prng)%(SIZE*2)]);
	}
}

#endif
