/* bigint-add.c -- RVV implementation of arbitrary-precision addition
 * Olaf Bernstein <camel-cdr@protonmail.com>
 * Distributed under the MIT license, see license at the end of the file.
 * New versions available at https://github.com/camel-cdr/rvv-playground
 *
 * Benchmarks:
 *
 * SpacemiT X100 N=1024*32 WORK=1ull<<32:
 *     gmp:            6.583392 secs
 *     simple:         7.047197 secs
 *     speculative:    5.874165 secs
 *     speculative_x4: 5.438176 secs
 *     sorear:         7.421120 secs
 *     rvv:            4.223685 secs
 *
 * SpacemiT A100 N=1024*32 WORK=1ull<<32:
 *     gmp:            201.278858 secs
 *     simple:         138.622004 secs
 *     speculative:    208.885835 secs
 *     speculative_x4: 121.358876 secs
 *     sorear:         121.111467 secs
 *     rvv:             13.446655 secs
 */

#include <stddef.h>
#include <stdint.h>

#define STR(x) #x
#if defined(__clang__)
#define UNROLL(n) _Pragma(STR(clang loop unroll_count(n)))
#elif defined(__GNUC__)
#define UNROLL(n) _Pragma(STR(gcc unroll n))
#else
#define UNROLL(n) _Pragma(STR(omp unroll partial(n)))
#endif

#if defined(__GNUC__)
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x)   (x)
#define unlikely(x) (x)
#endif


#define ARGS \
	uint64_t *restrict dst, \
	uint64_t *restrict lhs, \
	uint64_t *restrict rhs, \
	size_t n

uint64_t
adc_simple(ARGS, uint64_t carry)
{
	UNROLL(4)
	for (size_t i = 0; i < n; ++i) {
		uint64_t l = lhs[i], r = rhs[i], lr, d;
		lr = l + r;
		dst[i] = d = lr + carry;
		carry = (lr < l) | (d < lr);
	}
	return carry;
}

void add_simple(ARGS) { adc_simple(dst, lhs, rhs, n, 0); }

void
add_speculative(ARGS)
{
	uint64_t d, s, carry = 0;
	UNROLL(32)
	for (size_t i = 0; i < n; ++i) {
		d = lhs[i], s = rhs[i];
		d += s;
		dst[i] = d + carry;
		if (unlikely(d == (uint64_t)-1)) { __asm(""); continue; }
		carry = d < s;
	}
}

void
add_speculative_x4(ARGS)
{
	uint64_t d, s, carry = 0;
	UNROLL(16)
	for (; n >= 4; n -= 4) {
		uint64_t saved = carry;

		#define IMPL(f) \
			d = *lhs++ + (s = *rhs++); \
			*dst++ = d + carry; \
			const uint64_t f = d == (uint64_t)-1; \
			carry = d < s;

		IMPL(f1); IMPL(f2); IMPL(f3); IMPL(f4);

		if (unlikely((f1 | f2) | (f3 | f4))) {
			carry = adc_simple(dst-4, lhs-4, rhs-4, 4, saved);
		}
	}

	adc_simple(dst, lhs, rhs, n, carry);
}

/* https://www.reddit.com/r/RISCV/comments/1jsnbdr/comment/mn6mjoe */
void
add_sorear(ARGS)
{
	uint64_t l, r, sum, t1, t2, new_early_carry, early_carry, late_carry;
	late_carry = early_carry = 0;
	UNROLL(4)
	while (n--) {
		l = *lhs++;
		r = *rhs++;
		t1 = l + r;
		new_early_carry = t1 < l;
		t2 = t1 + early_carry;
		early_carry = new_early_carry; /* renamed out */
		*dst++ = sum = t2 + late_carry;
		late_carry = sum < t1;
	}
}

#if __riscv_vector

#include <riscv_vector.h>

void
add_rvv(uint64_t *restrict dst, uint64_t *restrict lhs, uint64_t *restrict rhs, size_t n)
{
	size_t VL1 = __riscv_vsetvlmax_e64m1(), VL2 = VL1*2, VL4 = VL1*4, VL8 = VL1*8;
	vuint64m1_t v1 = __riscv_vmv_v_x_u64m1(-1, VL1);
	vbool64_t carryin = __riscv_vmclr_m_b64(VL1);

#define IMPL1(mx, bx, vl_expr, stop_expr) do { \
	vbool##bx##_t all = __riscv_vreinterpret_b##bx(v1); \
	vuint64m1_t vlo = __riscv_vmv_v_x_u64m1(((((uint64_t)1 << (VL##mx-1)) - 1) << 1) + 1, VL1); \
	vuint64m1_t vhi = __riscv_vnot(vlo, VL1); \
	for (size_t vl; stop_expr; n -= vl, lhs += vl, rhs += vl, dst += vl) { \
		vl = vl_expr; \
		vuint64m##mx##_t L = __riscv_vle64_v_u64m##mx(lhs, vl); \
		vuint64m##mx##_t R = __riscv_vle64_v_u64m##mx(rhs, vl); \
		vuint64m1_t gen = __riscv_vreinterpret_u64m1(__riscv_vmadc(L, R, vl)); \
		vuint64m1_t prop = __riscv_vreinterpret_u64m1(__riscv_vmadc(L, R, all, vl)); \
		vbool##bx##_t carry = __riscv_vreinterpret_b##bx( \
			__riscv_vxor(__riscv_vadc(gen, prop, carryin, 1), __riscv_vxor(gen, prop, 1), 1) \
		); \
		gen = __riscv_vand(gen, vlo, 1); \
		prop = __riscv_vor(prop, vhi, 1); \
		carryin = __riscv_vmadc(gen, prop, carryin, 1); \
		vuint64m##mx##_t vsum = __riscv_vadc(L, R, carry, vl); \
		__riscv_vse64(dst, vsum, vl); \
	} \
} while (0)

#define IMPL2(mx, bx, vl_expr, stop_expr) do { \
	vbool##bx##_t all = __riscv_vreinterpret_b##bx(v1); \
	vbool64_t all2 = __riscv_vreinterpret_b64(v1); \
	size_t vl2 = VL##mx / 64;\
	vuint64m1_t vlo = __riscv_vmv_v_x_u64m1(((((uint64_t)1 << (VL##mx/64-1)) - 1) << 1) + 1, VL1); \
	vuint64m1_t vhi = __riscv_vnot(vlo, VL1); \
	for (size_t vl; stop_expr; n -= vl, lhs += vl, rhs += vl, dst += vl) { \
		vl = vl_expr; \
		vuint64m##mx##_t L = __riscv_vle64_v_u64m##mx(lhs, vl); \
		vuint64m##mx##_t R = __riscv_vle64_v_u64m##mx(rhs, vl); \
		vuint64m1_t gen = __riscv_vreinterpret_u64m1(__riscv_vmadc(L, R, vl)); \
		vuint64m1_t prop = __riscv_vreinterpret_u64m1(__riscv_vmadc(L, R, all, vl)); \
		vuint64m1_t gen2 = __riscv_vreinterpret_u64m1(__riscv_vmadc(gen, prop, vl2)); \
		vuint64m1_t prop2 = __riscv_vreinterpret_u64m1(__riscv_vmadc(gen, prop, all2, vl2)); \
		vbool64_t carry2 = __riscv_vreinterpret_b64( \
			__riscv_vxor(__riscv_vadc(gen2, prop2, carryin, 1), __riscv_vxor(gen2, prop2, 1), 1) \
		); \
		gen2 = __riscv_vand(gen2, vlo, 1); \
		prop2 = __riscv_vor(prop2, vhi, 1); \
		carryin = __riscv_vmadc(gen2, prop2, carryin, 1); \
		\
		vbool##bx##_t carry = __riscv_vreinterpret_b##bx( \
			__riscv_vxor(__riscv_vadc(gen, prop, carry2, vl2), __riscv_vxor(gen, prop, vl2), vl2) \
		); \
		vuint64m##mx##_t vsum = __riscv_vadc(L, R, carry, vl); \
		__riscv_vse64(dst, vsum, vl); \
	} \
} while (0)


	if (n <= 64) { IMPL1(2, 32, __riscv_vsetvl_e64m2(n < VL2 ? n : VL2), n > 0); return; }
	if (VL8 <= 64) IMPL1(8, 8, VL8, n >= VL8);
	else           IMPL2(8, 8, VL8, n >= VL8);
	if (VL4 <= 64) IMPL1(4, 16, VL4, n >= VL4);
	else           IMPL2(4, 16, VL4, n >= VL4);
	if (VL2 <= 64) IMPL1(2, 32, __riscv_vsetvl_e64m2(n < VL2 ? n : VL2), n > 0);
	else           IMPL2(2, 32, __riscv_vsetvl_e64m2(n < VL2 ? n : VL2), n > 0);

	#undef IMPL1
	#undef IMPL2
}
#endif

#if defined(FUZZ) || defined(BENCH)
#if __riscv_vector
#define XMACRO_RVV(f) f(add_rvv)
#else
#define XMACRO_RVV(f)
#endif
#define XMACRO(f) \
	f(add_simple) \
	f(add_speculative) \
	f(add_speculative_x4) \
	f(add_sorear) \
	XMACRO_RVV(f)
#endif

#if defined(FUZZ)

void
add_fuzzing(
	uint64_t *restrict dst,
	uint64_t *restrict lhs,
	uint64_t *restrict rhs,
	size_t n)
{
	uint64_t carry = 0;
	for (size_t i = 0; i < n; ++i) {
		uint64_t l = lhs[i], r = rhs[i], lr;
		lr = l + r;
		if (carry) {
			dst[i] = lr + 1;
			if (lr < l) carry = 1;
			else if (lr + 1 < lr) carry = 1; else carry = 0;
		} else {
			dst[i] = lr;
			if (lr < l) carry = 1; else carry = 0;
		}
	}
}

#include <assert.h>
#include <string.h>
#include <stdio.h>

#define N (1024*16)

int
LLVMFuzzerTestOneInput(const uint8_t *src, size_t size)
{
	size_t n = size / 2 / 8;
	if (n > N) return -1;
	if (((uintptr_t)src & 7) != 0) return -1;

	uint64_t *lhs = (uint64_t*)src, *rhs = lhs+n;

	static uint64_t ref[N], tst[N];
	add_fuzzing(ref, lhs, rhs, n);

	#define XCHECK(name) \
		name(tst, lhs, rhs, n); \
		if (memcmp(ref, tst, n) != 0) { \
			printf("ERROR: in " #name ", total=%zu\n", n); \
			for (size_t i = 0; i < n; ++i) \
				printf("%016zx + %016zx = %016zx, got %016zx %s\n", \
					lhs[i], rhs[i], ref[i], tst[i], \
					ref[i] != tst[i] ? "!!!" : ""); \
			assert(0); \
		} \

	XCHECK(add_rvv)
	//XMACRO(XCHECK)

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

#include <gmp.h>
void
add_gmp(
	uint64_t *restrict dst,
	uint64_t *restrict lhs,
	uint64_t *restrict rhs,
	size_t n)
{
	mpn_add_n(dst, lhs, rhs, n);
}

#define N (1024*32)
#define WORK (1ull << 32)

void
bench(
	const char *name,
	void (*add)(
		uint64_t *restrict dst,
		uint64_t *restrict lhs,
		uint64_t *restrict rhs,
		size_t n),
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
	for (size_t i = 0; i < WORK/n; ++i)
		add(dst, lhs, rhs, n);
	end = clock();

	uint64_t hash = 0;
	for (size_t i = 0; i < n; ++i)
		hash ^= hash64(dst[i]);
	printf("%30s time: %f secs hash: %lx\n", name, (end-beg)*1.0/CLOCKS_PER_SEC, hash);
}

int
main(void)
{
	#define XBENCH(name) bench(#name, &name, N);
	XBENCH(add_gmp)
	XMACRO(XBENCH)
}
#endif

