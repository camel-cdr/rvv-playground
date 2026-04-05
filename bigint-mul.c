/* bigint-mul.c -- RVV implementation of arbitrary-precision multiplication
 * Olaf Bernstein <camel-cdr@protonmail.com>
 * Distributed under the MIT license, see license at the end of the file.
 * New versions available at https://github.com/camel-cdr/rvv-playground
 *
 * Benchmarks:
 *
 * SpacemiT X100 N=1024*32 REP=1024*32:
 *     bits:    256      512      1024     2048     4096     8192
 *     ref:     0.0022   0.0069   0.0260   0.1055   0.4464   1.6406 secs
 *     sketch1: 0.0029   0.0087   0.0318   0.1242   0.4951   1.9522 secs
 *     sketch2: 0.0025   0.0071   0.0259   0.0977   0.3871   1.4938 secs
 *     sketch3: 0.0020   0.0066   0.0242   0.0961   0.3898   1.4831 secs
 *     rvv:     0.0021   0.0057   0.0190   0.0892   0.4014   1.5958 secs
 *     rbr:     0.0141   0.0359   0.1165   0.4159   1.5283   5.8680 secs
 *     vmac52:  0.0032   0.0097   0.0225   0.0665   0.2579   1.0000 secs
 *     vmac54:  0.0026   0.0092   0.0225   0.0648   0.2515   0.9003 secs
 *     vmac56:  0.0026   0.0092   0.0213   0.0614   0.2419   0.8712 secs
 * Note: CPU has 1/2 SEW=64 multiplication throughput compared to SEW=32
 *
 * SpacemiT A100 N=1024*32 REP=1024*32:
 *     bits:    256      512      1024     2048     4096     8192
 *     ref:     0.0340   0.1026   0.3250   1.1367   4.2319   6.3660 secs
 *     sketch1: 0.0528   0.1756   0.5758   2.0685   7.7863   0.2770 secs
 *     sketch2: 0.0336   0.1017   0.3300   1.1777   4.4566   7.4480 secs
 *     sketch3: 0.0383   0.1204   0.4390   1.6825   6.6157   6.2890 secs
 *     rvv:     0.0128   0.0223   0.0372   0.1076   0.3103   2.8954 secs
 *     rbr:     0.0581   0.1092   0.2915   0.7674   2.4982   9.1551 secs
 *     vmac52:  0.0173   0.0305   0.0464   0.1663   0.4524   1.5449 secs
 *     vmac54:  0.0155   0.0283   0.0466   0.1624   0.4419   1.4858 secs
 *     vmac56:  0.0166   0.0252   0.0427   0.1550   0.4238   1.4373 secs
 * Note: CPU has 1/4 SEW=64 multiplication throughput compared to SEW=16
 *
 * SpacemiT X60 N=1024*32 REP=1024*32:
 *     bits:    256      512      1024     2048     4096     8192
 *     ref:     0.0081   0.0266   0.0993   0.3862   1.5274   6.1021 secs
 *     sketch1: 0.0125   0.0440   0.1685   0.6611   2.6230   0.4743 secs
 *     sketch2: 0.0102   0.0348   0.1315   0.5141   2.0340   8.1214 secs
 *     sketch3: 0.0094   0.0326   0.1246   0.4893   1.9434   7.7695 secs
 *     rvv:     0.0040   0.0098   0.0309   0.1368   0.7360   2.9165 secs
 *     rbr:     0.0218   0.0570   0.1741   0.6136   2.2659   8.7132 secs
 *     vmac52:  0.0047   0.0149   0.0337   0.0945   0.3639   1.4056 secs
 *     vmac54:  0.0041   0.0139   0.0338   0.0920   0.3549   1.2616 secs
 *     vmac56:  0.0041   0.0139   0.0320   0.0875   0.3415   1.2202 secs
 * Note: CPU has 1/2 SEW=64 multiplication throughput compared to SEW=32
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__)
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x)   (x)
#define unlikely(x) (x)
#endif

#define ARGS \
	uint64_t *restrict dst, \
	const uint64_t *restrict lhs, size_t nl, \
	const uint64_t *restrict rhs, size_t nr

void
mul_ref(ARGS)
{
	memset(dst, 0, (nl + nr) * sizeof(uint64_t));
	for (size_t i = 0; i < nl; ++i) {
		__uint128_t carry = 0, prod;
		for (size_t j = 0; j < nr; ++j) {
			prod = (__uint128_t)lhs[i] * rhs[j] + dst[i + j] + carry;
			dst[i + j] = prod;
			carry = prod >> 64;
		}
		dst[i + nr] += carry;
	}
}

void
mul_sketch1(ARGS)
{
	memset(dst, 0, (nl + nr) * sizeof(uint64_t));
	for (size_t i = 0; i < nl; ++i) {
		uint64_t carry = 0;
		for (size_t j = 0; j < nr; ++j) {
			uint64_t prod = lhs[i] * rhs[j];
			uint64_t s1 = dst[i+j] + prod, s2 = s1 + carry;
			dst[i+j] = s2;
			carry = (s1 < prod) | (s2 < s1);
		}
		dst[i + nr] += carry;

		carry = 0;
		for (size_t j = 0; j < nr; ++j) {
			uint64_t prod = ((__uint128_t)lhs[i] * rhs[j]) >> 64;
			uint64_t s1 = dst[i+j+1] + prod, s2 = s1 + carry;
			dst[i+j+1] = s2;
			carry = (s1 < prod) | (s2 < s1);
		}
	}
}

void
mul_sketch2(ARGS)
{
	memset(dst, 0, (nl + nr) * sizeof(uint64_t));
	for (size_t i = 0; i < nl; ++i) {
		uint64_t c1 = 0, c2 = 0, prod, s1, s2;
		for (size_t j = 0; j < nr; ++j) {
			prod = lhs[i] * rhs[j];
			s1 = dst[i+j] + prod, s2 = s1 + c1;
			dst[i+j] = s2;
			c1 = (s1 < prod) | (s2 < s1);

			prod = ((__uint128_t)lhs[i] * rhs[j]) >> 64;
			s1 = dst[i+j+1] + prod, s2 = s1 + c2;
			dst[i+j+1] = s2;
			c2 = (s1 < prod) | (s2 < s1);
		}
		dst[i + nr] += c1;
	}
}

void
mul_sketch3(ARGS)
{
	memset(dst, 0, (nl + nr) * sizeof(uint64_t));
	for (size_t i = 0; i < nl; ++i) {
		uint64_t carry = 0, prod, sum;
		for (size_t j = 0; j < nr; ++j) {
			prod = lhs[i] * rhs[j];
			dst[i+j] = sum = dst[i+j] + prod;
			prod = (sum < prod) + carry;
			prod += ((__uint128_t)lhs[i] * rhs[j]) >> 64;
			dst[i+j+1] = sum = dst[i+j+1] + prod;
			carry = sum < prod;
		}
	}
}

#if __riscv_vector

#include <riscv_vector.h>

#include <assert.h>
#include "utils.h"

void
mul_rvv(ARGS)
{
	size_t VL1 = __riscv_vsetvlmax_e64m1(), VL2 = VL1*2, VL4 = VL1*4, VL8 = VL1*8;

#define SMALL(mx,bx) \
	if (nr <= VL##mx) { \
		const size_t vl = nr; \
		vuint64m##mx##_t vrhs = __riscv_vle64_v_u64m##mx(rhs, vl); \
		vuint64m##mx##_t vdst = __riscv_vmv_v_x_u64m##mx(0, vl); \
		vbool##bx##_t all = __riscv_vmset_m_b##bx(vl); \
		for (size_t i = 0; i < nl; ++i) { \
			vuint64m##mx##_t vlo = __riscv_vmul(vrhs, lhs[i], vl); \
			vuint64m##mx##_t vhi = __riscv_vmulhu(vrhs, lhs[i], vl); \
			vbool##bx##_t inc = __riscv_vmadc(vdst, vlo, vl); \
			vhi = __riscv_vadc(vhi, 0, inc, vl); /* can't carry */ \
\
			vdst = __riscv_vadd(vdst, vlo, vl); \
			dst[i] = __riscv_vmv_x(vdst); \
			vdst = __riscv_vslide1down(vdst, 0, vl); \
\
			vuint64m1_t gen = __riscv_vreinterpret_u64m1(__riscv_vmadc(vdst, vhi, vl)); \
			vuint64m1_t prop = __riscv_vreinterpret_u64m1(__riscv_vmadc(vdst, vhi, all, vl)); \
			vbool##bx##_t carry = __riscv_vreinterpret_b##bx( \
				__riscv_vxor(__riscv_vadd(gen, prop, 1), \
				             __riscv_vxor(gen, prop, 1), 1)); \
			vdst = __riscv_vadc(vdst, vhi, carry, vl); \
		} \
		__riscv_vse64(dst + nl, vdst, vl); \
		return; \
	}
	if (nr-1 < 64) {
		SMALL(1,64);
		SMALL(2,32);
		SMALL(4,16);
		SMALL(8,8);
	}
#undef SMALL

	uint64_t *d = dst;
	vuint64m8_t zero = __riscv_vmv_v_x_u64m8(0, VL8);
	for (size_t vl, n = nl + nr; n > 0; n -= vl, d += vl)
		__riscv_vse64(d, zero, vl = __riscv_vsetvl_e64m8(n));
	if (nr == 0 || nl == 0) return;

	vbool8_t all = __riscv_vmset_m_b8(VL8);
	if (VL8 > 64) goto large;

	uint64_t masklo = ((((uint64_t)1 << (VL8-1)) - 1) << 1) + 1;
	uint64_t maskhi = ~masklo;
	for (size_t i = 0; i < nl; ++i) {
		vbool64_t carryin = __riscv_vmclr_m_b64(VL1);
		d = dst+i;
		uint64_t const *r = rhs;
		for (size_t vl, n = nr; n > 0; n -= vl, d += vl, r += vl) {
			vl = __riscv_vsetvl_e64m8(n < VL8 ? n : VL8);

			vuint64m8_t vdst = __riscv_vle64_v_u64m8(d, vl);
			vuint64m8_t vrhs = __riscv_vle64_v_u64m8(r, vl);
			vuint64m8_t vlo = __riscv_vmul(vrhs, lhs[i], vl);
			vuint64m8_t vhi = __riscv_vmulhu(vrhs, lhs[i], vl);
			vbool8_t inc = __riscv_vmadc(vdst, vlo, vl);
			vhi = __riscv_vadc(vhi, 0, inc, vl); /* can't carry */

			vdst = __riscv_vadd(vdst, vlo, vl);
			d[0] = __riscv_vmv_x(vdst);
			vdst = __riscv_vslide1down(vdst, d[vl], vl);

			vuint64m1_t gen = __riscv_vreinterpret_u64m1(__riscv_vmadc(vdst, vhi, vl));
			vuint64m1_t prop = __riscv_vreinterpret_u64m1(__riscv_vmadc(vdst, vhi, all, vl));
			vbool8_t carry = __riscv_vreinterpret_b8(
					__riscv_vxor(__riscv_vadc(gen, prop, carryin, 1),
			                             __riscv_vxor(gen, prop, 1), 1));
			gen = __riscv_vand(gen, masklo, 1);
			prop = __riscv_vor(prop, maskhi, 1);
			carryin = __riscv_vmadc(gen, prop, carryin, 1);
			vdst = __riscv_vadc(vdst, vhi, carry, vl);
			__riscv_vse64(d+1, vdst, vl);
		}
	}
	return;

large:
	/* For VLEN=2^16 (the maximum), we only use half and don't bother with
	 * LMUL=4, because any sensible VLEN=2^16 implementation will have Ovlt
	 * behavior. */
	VL8 = VL8 < 64*64 ? VL8 : 64*64;
	VL1 = VL8/8;

	size_t VLM = VL8 / 64;
	masklo = ((((uint64_t)1 << (VLM-1)) - 1) << 1) + 1;
	maskhi = ~masklo;
	vbool64_t all2 = __riscv_vmset_m_b64(VL1);
	for (size_t i = 0; i < nl; ++i) {
		vbool64_t carryin = __riscv_vmclr_m_b64(VL1);
		d = dst+i;
		uint64_t const *r = rhs;
		for (size_t vl, n = nr; n > 0; n -= vl, d += vl, r += vl) {
			vl = __riscv_vsetvl_e64m8(n < VL8 ? n : VL8);

			vuint64m8_t vdst = __riscv_vle64_v_u64m8(d, vl);
			vuint64m8_t vrhs = __riscv_vle64_v_u64m8(r, vl);
			vuint64m8_t vlo = __riscv_vmul(vrhs, lhs[i], vl);
			vuint64m8_t vhi = __riscv_vmulhu(vrhs, lhs[i], vl);
			vbool8_t inc = __riscv_vmadc(vdst, vlo, vl);
			vhi = __riscv_vadc(vhi, 0, inc, vl); /* can't carry */

			vdst = __riscv_vadd(vdst, vlo, vl);
			d[0] = __riscv_vmv_x(vdst);
			vdst = __riscv_vslide1down(vdst, d[vl], vl);

			vuint64m1_t gen = __riscv_vreinterpret_u64m1(__riscv_vmadc(vdst, vhi, vl));
			vuint64m1_t prop = __riscv_vreinterpret_u64m1(__riscv_vmadc(vdst, vhi, all, vl));
			vuint64m1_t gen2 = __riscv_vreinterpret_u64m1(__riscv_vmadc(gen, prop, VLM));
			vuint64m1_t prop2 = __riscv_vreinterpret_u64m1(__riscv_vmadc(gen, prop, all2, VLM));
			vbool64_t carry2 = __riscv_vreinterpret_b64(
				__riscv_vxor(__riscv_vadc(gen2, prop2, carryin, 1), __riscv_vxor(gen2, prop2, 1), 1)
			);
			gen2 = __riscv_vand(gen2, masklo, 1);
			prop2 = __riscv_vor(prop2, maskhi, 1);
			carryin = __riscv_vmadc(gen2, prop2, carryin, 1);
			vbool8_t carry = __riscv_vreinterpret_b8(
					__riscv_vxor(__riscv_vadc(gen, prop, carry2, VLM),
			                             __riscv_vxor(gen, prop, VLM), VLM));
			vdst = __riscv_vadc(vdst, vhi, carry, vl);
			__riscv_vse64(d+1, vdst, vl);
		}
	}
}

void
mul_vmaccN(ARGS)
{
	size_t VL1 = __riscv_vsetvlmax_e64m1(), VL2 = VL1*2, VL4 = VL1*4, VL8 = VL1*8;
	vuint64m8_t zero = __riscv_vmv_v_x_u64m8(0, VL8);
	uint64_t *d = dst;
	for (size_t vl, n = nl + nr; n > 0; n -= vl, d += vl)
		__riscv_vse64(d, zero, vl = __riscv_vsetvl_e64m8(n));

#define IMPL(mx, vl_expr, stop_expr) \
	for (size_t vl; stop_expr; nr -= vl, rhs += vl, dst += vl) { \
		vl = vl_expr; \
		vuint64m##mx##_t vrhs = __riscv_vle64_v_u64m##mx(rhs, vl); \
		vuint64m##mx##_t vdst = __riscv_vle64_v_u64m##mx(dst, vl); \
		for (size_t i = 0; i < nl; ++i) { \
			uint64_t l = lhs[i]; \
			vdst = __riscv_vmacc(vdst, l, vrhs, vl); /* vmaccu<N>l.vx */ \
			dst[i] = __riscv_vmv_x(vdst); \
			vdst = __riscv_vslide1down(vdst, 0, vl); \
			vdst = __riscv_vmacc(vdst, l, vrhs, vl); /* vmaccu<N>h.vx */ \
		} \
		__riscv_vse64(dst + nl, vdst, vl); \
	}
	IMPL(8, VL8, nr >= VL8);
	IMPL(4, VL4, nr >= VL4);
	IMPL(2, __riscv_vsetvl_e64m2(nr < VL2 ? nr : VL2), nr > 0);
#undef IMPL
}

#endif

#if defined(FUZZ) || defined(BENCH)
#if __riscv_vector
#define XMACRO_RVV(f) f(mul_rvv)
#else
#define XMACRO_RVV(f)
#endif
#define XMACRO(f) \
	f(mul_ref) \
	f(mul_sketch1) \
	f(mul_sketch2) \
	f(mul_sketch3) \
	XMACRO_RVV(f)
#endif

#if defined(FUZZ)

void
mul_fuzz(ARGS)
{
	memset(dst, 0, (nl + nr) * sizeof(uint64_t));
	for (size_t i = 0; i < nl; ++i) {
		uint64_t carry = 0;
		for (size_t j = 0; j < nr; ++j) {
			__uint128_t prod = (__uint128_t)lhs[i] * rhs[j];
			uint64_t lo = prod, hi = prod >> 64;
			uint64_t sum = dst[i + j] + lo;
			if (sum < lo) ++hi;
			if (sum + carry < sum) ++hi;
			dst[i + j] = sum + carry;
			carry = hi;
		}
		dst[i + nr] += carry;
	}
}

#include <assert.h>
#include <string.h>
#include <stdio.h>

#define N (1024*32)

int
LLVMFuzzerTestOneInput(const uint8_t *src, size_t size)
{
	if (((uintptr_t)src & 7) != 0) return -1;
	if (size < 2) return -1;
	size_t nl = src[size-2] << 8 | src[size-1]; size -= 2;
	size_t n = size / 8;
	if (n > N) return -1;
	nl %= n+1; /* in range [0,n] */
	size_t nr = n - nl;

	uint64_t *lhs = (uint64_t*)src, *rhs = lhs+nl;
	static uint64_t ref[N*2], tst[N*2];
	mul_fuzz(ref, lhs, nl, rhs, nr);

	#define XCHECK(name) \
		name(tst, lhs, nl, rhs, nr); \
		if (memcmp(ref, tst, nl+nr) != 0) { \
			printf("ERROR: in " #name ", total=%zux%zu\n", nl, nr); \
			for (size_t i = 0; i < nl+nr; ++i) \
				printf("%016zx, got %016zx %s\n", \
					ref[i], tst[i], \
					ref[i] != tst[i] ? "!!!" : ""); \
			assert(0); \
		} \

	//XCHECK(mul_rvv)
	XMACRO(XCHECK)

	return 0;
}

#if 0
int
main(void) /* for builds without libfuzzer */
{
	static const struct { uint8_t src[N*8*2]; size_t size; } arr[] = {
	};
	for (size_t i = 0; i < sizeof arr / sizeof *arr; ++i)
		LLVMFuzzerTestOneInput(arr[i].src, arr[i].size);
}
#endif

#elif defined(BENCH)

#include <stdio.h>
#include <time.h>
#include "utils.h"

#define N (1024)
#define REP (1024*32)
void
bench(
	const char *name,
	void (*mul)(
		uint64_t *restrict dst,
		const uint64_t *restrict lhs, size_t rl,
		const uint64_t *restrict rhs, size_t rn),
	size_t n)
{
	RandU64 rng = { (uint64_t)&bench, 1234, 678 };
	uint64_t dst[N], lhs[N], rhs[N];
	for (size_t i = 0; i < n; ++i) {
		lhs[i] = randu64(&rng);
		rhs[i] = randu64(&rng);
	}

	time_t beg, end;
	beg = clock();
	for (size_t i = 0; i < REP; ++i)
		mul(dst, lhs, n, rhs, n);
	end = clock();

	uint64_t hash = 0;
	for (size_t i = 0; i < n*2; ++i)
		hash ^= hash64(dst[i]);
	printf("%30s time: %9.6f secs hash: %lx\n", name, (end-beg)*1.0/CLOCKS_PER_SEC, hash);
}

extern void /* https://lists.riscv.org/g/tech-crypto-ext/message/1112 */
rbr_mul_56_rvv_carryi(
		uint64_t *dst56,
		const uint64_t *restrict lhs56, size_t nl56,
		const uint64_t *restrict rhs56, size_t lr56);

int
main(void)
{
	size_t cases[] = { 256, 512, 1024, 2048, 4096, 8192 };
	for (size_t i = 0; i < sizeof(cases) / sizeof *cases; ++i) {
		size_t n = cases[i]/64;
		printf("n=%zu:\n", n);
		#define XBENCH(name) bench(#name, &name, n);
		XMACRO(XBENCH)
#if __riscv_vector
		bench("rbr", rbr_mul_56_rvv_carryi, (n*64+55)/56);
		bench("vmac52", mul_vmaccN, (n*64+63)/52);
		bench("vmac54", mul_vmaccN, (n*64+63)/54);
		bench("vmac56", mul_vmaccN, (n*64+63)/56);
#endif
	}
}
#endif
