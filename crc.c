#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include <riscv_vector.h>
#include <riscv_bitmanip.h>

#if VARIANT == 0
#define REFLECT 1
#define POLY 0xEDB88320 // iSCSI
#elif VARIANT == 1
#define REFLECT 0
#define POLY 0x04C11DB7 // IEEE
#endif

#if defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)
typedef uint64_t __attribute__((__may_alias__)) u64a;
#else
typedef uint64_t u64a;
#endif

/* reference implementation */
static uint32_t
crc32_ref(uint8_t *src, size_t n, uint32_t crc)
{
#if REFLECT
	for (size_t i = 0; i < n; ++i) {
		crc = crc ^ src[i];
		for (size_t j = 0; j < 8; ++j)
			crc = (crc & 1) ? (crc >> 1) ^ POLY : (crc >> 1);
	}
	return crc;
#else
	for (size_t i = 0; i < n; ++i) {
		crc = crc ^ ((uint64_t)src[i] << 24);
		for (size_t j = 0; j < 8; ++j)
			crc = (crc & 1u<<31) ? (crc << 1) ^ POLY : (crc << 1);
	}
	return crc;
#endif
}

/* GF(2) helper functions */
static uint64_t
xnmodp(uint64_t n, uint64_t poly, uint64_t deg)
{ /* source: https://stackoverflow.com/a/21201497 public domain by Mark Adler */
	uint64_t mod, mask;
	if (n < deg) return poly;
	mod = poly &= mask = ((((uint64_t)1 << (--deg)) - 1) << 1) + 1;
	for (; --n > deg;) mod = (mod << 1) ^ ((mod >> deg) & 1 ? poly : 0);
	return mod & mask;
}
static uint64_t
xndivp(uint64_t n, uint64_t poly, uint64_t deg)
{
	__uint128_t d = 0, o = 1;
	__uint128_t x = o << n;
	__uint128_t p = poly ^ (o << deg);
	while (x >= (o << deg)) {
		size_t l = 128 - __riscv_clz_64(x >> 64);
		if (l == 64) l -= __riscv_clz_64(x);
		d ^= o << (l - deg - 1);
		x ^= p << (l - deg - 1);
	}
	return d;
}
static uint64_t
bitrev(uint64_t x)
{
#if  __riscv_zbkb
	x = __riscv_brev8_64(x);
#else
	x = ((x >>  1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1);
	x = ((x >>  2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2);
	x = ((x >>  4) & 0x0f0f0f0f0f0f0f0f) | ((x & 0x0f0f0f0f0f0f0f0f) << 4);
#endif
#if  __riscv_zbb
	x = __riscv_rev8_64(x);
#else
	x = ((x >>  8) & 0x00ff00ff00ff00ff) | ((x & 0x00ff00ff00ff00ff) <<  8);
	x = ((x >> 16) & 0x0000ffff0000ffff) | ((x & 0x0000ffff0000ffff) << 16);
	x = ((x >> 32) & 0x00000000ffffffff) | ((x & 0x00000000ffffffff) << 32);
#endif
	return x;
}
static uint64_t
fold_poly(uint64_t N)
{
#if REFLECT
	return bitrev(xnmodp(N, bitrev(POLY) >> 32, 32)) >> 31;
#else
	return xnmodp(N, POLY, 32) << 32;
#endif
}

static uint64_t barrettPoly = 0;
static uint64_t foldPoly0[16] = {0};
static uint64_t foldPoly1[16] = {0};

static void
init_poly(void)
{
	for (size_t i = 0; i < 16; ++i) {
		foldPoly0[i] = fold_poly(((size_t)1<<i)*2*64+32);
		foldPoly1[i] = fold_poly(((size_t)1<<i)*2*64-32);
	}

#if REFLECT
	barrettPoly = bitrev(xndivp(63+32, bitrev(POLY) >> 32, 32));
#else
	barrettPoly = xndivp(63+32, POLY, 32);
#endif
}

/*
 * Scalar implementation using clmul for short inputs via barrett reduction.
 *
 * A scalar clmul implementation for bigger inputs, should use folding with
 * multiple accumulators instead.
 */
static uint32_t
crc32_short(uint8_t *src, size_t n, uint64_t crc)
{
	if (n < 1) return crc;
	u64a *aligned = (u64a*)((uintptr_t)src & ~(uintptr_t)7);
#if REFLECT
	if ((uintptr_t)aligned != (uintptr_t)src) {
		size_t align = (uintptr_t)src - (uintptr_t)aligned;
		size_t N = 8-align < n ? 8-align : n;
		uint64_t tmp = ((*aligned++ >> (align*8)) ^ crc) << (64-N*8);
		crc = __riscv_clmulr_64(__riscv_clmul_64(tmp, barrettPoly), POLY)
		      ^ (crc >> (N*8));
		n -= N;
	}
	for (; n >= 8; n -= 8) {
		uint64_t tmp = *aligned++ ^ crc;
		crc = __riscv_clmulr_64(__riscv_clmul_64(tmp, barrettPoly), POLY);
	}
	if (n > 0) {
		uint64_t tmp = (*aligned ^ crc) << (64-n*8);
		crc = __riscv_clmulr_64(__riscv_clmul_64(tmp, barrettPoly), POLY)
		      ^ (crc >> (n*8));
	}
#else
	if ((uintptr_t)aligned != (uintptr_t)src) {
		size_t align = (uintptr_t)src - (uintptr_t)aligned;
		size_t N = 8-align < n ? 8-align : n;
		uint64_t tmp = __riscv_rev8_64(*aligned++) << (align*8) >> (64-N*8);
		tmp ^= N >= 4 ? crc << (N*8-32) : crc >> (32-N*8);
		crc = __riscv_clmul_64(__riscv_clmulr_64(tmp, barrettPoly), POLY) ^ (crc << (N*8));
		n -= N;
	}
	for (; n >= 8; n -= 8, src += 8) {
		uint64_t tmp = __riscv_rev8_64(*aligned++) ^ crc << 32;
		crc = __riscv_clmul_64(__riscv_clmulr_64(tmp, barrettPoly), POLY);
	}
	if (n > 0) {
		uint64_t tmp = __riscv_rev8_64(*aligned++) >> (64-n*8);
		crc = (uint32_t)crc;
		tmp ^= n >= 4 ? crc << (n*8-32) : crc >> (32-n*8);
		crc = __riscv_clmul_64(__riscv_clmulr_64(tmp, barrettPoly), POLY) ^ (crc << (n*8));
	}
#endif
	return crc;
}

/*
 * Recommended RVV reference implementation.
 */
uint32_t
crc32_rvv_seg(uint8_t *src, size_t n, uint32_t crc)
{
	if (n < 16*2+8) return crc32_short(src, n, crc);

	/* align to 16-bytes */
	size_t align = (uintptr_t)src & 15;
	if (align) {
		size_t N = 16-align < n ? 16-align : n;
		crc = crc32_short(src, N, crc);
		n -= N;
		src += N;
	}

	vuint64m1_t vacc0, vacc1;

#define CRC_VFOLD(T, vacc0, vacc1, v0, v1, fold0, fold1, vl) do { \
		T vlo0 = __riscv_vclmul( vacc0, fold0, vl); \
		T vlo1 = __riscv_vclmul( vacc1, fold1, vl); \
		T vhi0 = __riscv_vclmulh(vacc0, fold0, vl); \
		T vhi1 = __riscv_vclmulh(vacc1, fold1, vl); \
		T vlo = __riscv_vxor(vlo0, vlo1, vl); \
		T vhi = __riscv_vxor(vhi0, vhi1, vl); \
		vacc0 = __riscv_vxor(v0, REFLECT ? vlo : vhi, vl); \
		vacc1 = __riscv_vxor(v1, REFLECT ? vhi : vlo, vl); \
	} while(0)

	size_t vl = 0;
	const size_t VL4 = __riscv_vsetvlmax_e64m4(), VLb = VL4*2*8;
	if (n > VLb) {
		vl = VL4;
		size_t log2vl = __riscv_ctz_64(vl);
		vuint64m4_t vfold0 = __riscv_vmv_v_x_u64m4(foldPoly0[log2vl], vl);
		vuint64m4_t vfold1 = __riscv_vmv_v_x_u64m4(foldPoly1[log2vl], vl);
		vuint64m4x2_t vacc = __riscv_vlseg2e64_v_u64m4x2((uint64_t*)src, vl);
		vuint64m4_t vacc40 = __riscv_vget_u64m4(vacc, 0);
		vuint64m4_t vacc41 = __riscv_vget_u64m4(vacc, 1);
#if REFLECT
		vacc40 = __riscv_vxor_tu(vacc40, vacc40, crc, 1);
#else
		vacc40 = __riscv_vrev8(vacc40, vl);
		vacc41 = __riscv_vrev8(vacc41, vl);
		vacc40 = __riscv_vxor_tu(vacc40, vacc40, ((uint64_t)crc)<<32, 1);
#endif
		n -= VLb, src += VLb;
		for (; n > VLb; n -= VLb, src += VLb) {
			vuint64m4x2_t v = __riscv_vlseg2e64_v_u64m4x2((uint64_t*)src, vl);
			vuint64m4_t v0 = __riscv_vget_u64m4(v, 0);
			vuint64m4_t v1 = __riscv_vget_u64m4(v, 1);
#if !REFLECT
			v0 = __riscv_vrev8(v0, vl);
			v1 = __riscv_vrev8(v1, vl);
#endif
			CRC_VFOLD(vuint64m4_t, vacc40, vacc41, v0, v1, vfold0, vfold1, vl);
		}

		vuint64m2_t vacc20 = __riscv_vget_u64m2(vacc40, 0);
		vuint64m2_t vacc21 = __riscv_vget_u64m2(vacc41, 0);
		{
			vl >>= 1; --log2vl;
			uint64_t fold0 = foldPoly0[log2vl];
			uint64_t fold1 = foldPoly1[log2vl];
			vuint64m2_t v0 = __riscv_vget_u64m2(vacc40, 1);
			vuint64m2_t v1 = __riscv_vget_u64m2(vacc41, 1);
			CRC_VFOLD(vuint64m2_t, vacc20, vacc21, v0, v1, fold0, fold1, vl);
		}

		vacc0 = __riscv_vget_u64m1(vacc20, 0);
		vacc1 = __riscv_vget_u64m1(vacc21, 0);
		{
			vl >>= 1; --log2vl;
			uint64_t fold0 = foldPoly0[log2vl];
			uint64_t fold1 = foldPoly1[log2vl];
			vuint64m1_t v0 = __riscv_vget_u64m1(vacc20, 1);
			vuint64m1_t v1 = __riscv_vget_u64m1(vacc21, 1);
			CRC_VFOLD(vuint64m1_t, vacc0, vacc1, v0, v1, fold0, fold1, vl);
		}
	} else {
		const size_t VL = __riscv_vsetvlmax_e64m1();
		vl = VL < n/16 ? VL : n/16;
		vuint64m1x2_t vacc = __riscv_vlseg2e64_v_u64m1x2((uint64_t*)src, vl);
		vacc0 = __riscv_vget_u64m1(vacc, 0);
		vacc1 = __riscv_vget_u64m1(vacc, 1);
#if REFLECT
		vacc0 = __riscv_vxor_tu(vacc0, vacc0, crc, 1);
#else
		vacc0 = __riscv_vrev8(vacc0, vl);
		vacc1 = __riscv_vrev8(vacc1, vl);
		vacc0 = __riscv_vxor_tu(vacc0, vacc0, ((uint64_t)crc)<<32, 1);
#endif
		vuint64m1_t vzero = __riscv_vmv_v_x_u64m1(0, VL);
		size_t vlin = vl;
		vl = (size_t)1 << (-__riscv_clz_64(vl-1) & 63);
		vacc0 = __riscv_vslideup(vzero, vacc0, vl-vlin, vl);
		vacc1 = __riscv_vslideup(vzero, vacc1, vl-vlin, vl);
		n -= vlin*16, src += vlin*16;
	}

	size_t log2vl = __riscv_ctz_64(vl);
	uint64_t fold0 = foldPoly0[log2vl];
	uint64_t fold1 = foldPoly1[log2vl];
	while (vl > 1) {
		while ((vl*16) <= n) {
			vuint64m1x2_t v = __riscv_vlseg2e64_v_u64m1x2((uint64_t*)src, vl);
			vuint64m1_t v0 = __riscv_vget_u64m1(v, 0);
			vuint64m1_t v1 = __riscv_vget_u64m1(v, 1);
#if !REFLECT
			v0 = __riscv_vrev8(v0, vl);
			v1 = __riscv_vrev8(v1, vl);
#endif
			CRC_VFOLD(vuint64m1_t, vacc0, vacc1, v0, v1, fold0, fold1, vl);
			n -= vl*16, src += vl*16;
		}
		vl >>= 1; --log2vl;
		fold0 = foldPoly0[log2vl];
		fold1 = foldPoly1[log2vl];
		vuint64m1_t v0 = __riscv_vslidedown(vacc0, vl, vl);
		vuint64m1_t v1 = __riscv_vslidedown(vacc1, vl, vl);
		CRC_VFOLD(vuint64m1_t, vacc0, vacc1, v0, v1, fold0, fold1, vl);
	}

	uint64_t u0 = __riscv_vmv_x(vacc0);
	uint64_t u1 = __riscv_vmv_x(vacc1);

#if REFLECT
	crc = __riscv_clmulr_64(__riscv_clmul_64(u0, barrettPoly), POLY);
	crc = __riscv_clmulr_64(__riscv_clmul_64(u1 ^ crc, barrettPoly), POLY);
#else
	crc = __riscv_clmul_64(__riscv_clmulr_64(u0, barrettPoly), POLY);
	crc = __riscv_clmul_64(__riscv_clmulr_64(u1 ^ (uint64_t)crc << 32, barrettPoly), POLY);
#endif

	return crc32_short(src, n, crc);
}

/*
 * Simple RVV reference implementation of the main loop.
 */
uint32_t
crc32_rvv_poc_seg(uint8_t *src, size_t n, uint32_t crc)
{
	size_t VL = __riscv_vsetvlmax_e64m4();
	size_t VLb = VL*2*8;
	if (n > VLb) {
		vuint64m4_t vfold0 = __riscv_vmv_v_x_u64m4(foldPoly0[__riscv_ctz_64(VL)], VL);
		vuint64m4_t vfold1 = __riscv_vmv_v_x_u64m4(foldPoly1[__riscv_ctz_64(VL)], VL);
		vuint64m4x2_t vacc = __riscv_vlseg2e64_v_u64m4x2((uint64_t*)src, VL);
		vuint64m4_t vacc0 = __riscv_vget_u64m4(vacc, 0);
		vuint64m4_t vacc1 = __riscv_vget_u64m4(vacc, 1);
#if REFLECT
		vacc0 = __riscv_vxor_tu(vacc0, vacc0, crc, 1);
#else
		vacc0 = __riscv_vrev8(vacc0, VL);
		vacc1 = __riscv_vrev8(vacc1, VL);
		vacc0 = __riscv_vxor_tu(vacc0, vacc0, (uint64_t)crc<<32, 1);
#endif
		n -= VLb, src += VLb;
		for (; n > VLb; n -= VLb, src += VLb) {
			vuint64m4x2_t v = __riscv_vlseg2e64_v_u64m4x2((uint64_t*)src, VL);
			vuint64m4_t v0 = __riscv_vget_u64m4(v, 0);
			vuint64m4_t v1 = __riscv_vget_u64m4(v, 1);
#if !REFLECT
			v0 = __riscv_vrev8(v0, VL);
			v1 = __riscv_vrev8(v1, VL);
#endif
			vuint64m4_t vlo0 = __riscv_vclmul( vacc0, vfold0, VL);
			vuint64m4_t vlo1 = __riscv_vclmul( vacc1, vfold1, VL);
			vuint64m4_t vhi0 = __riscv_vclmulh(vacc0, vfold0, VL);
			vuint64m4_t vhi1 = __riscv_vclmulh(vacc1, vfold1, VL);
			vuint64m4_t vlo = __riscv_vxor(vlo0, vlo1, VL);
			vuint64m4_t vhi = __riscv_vxor(vhi0, vhi1, VL);
			vacc0 = __riscv_vxor(v0, REFLECT ? vlo : vhi, VL);
			vacc1 = __riscv_vxor(v1, REFLECT ? vhi : vlo, VL);
		}
#if !REFLECT
		vacc0 = __riscv_vrev8(vacc0, VL);
		vacc1 = __riscv_vrev8(vacc1, VL);
#endif
		vuint64m4x2_t va = __riscv_vcreate_v_u64m4x2(vacc0, vacc1);
		uint64_t buf[VL*2];
		__riscv_vsseg2e64(buf, va, VL);
		crc = crc32_short((void*)buf, sizeof buf, 0);
	}
	return crc32_short(src, n, crc);
}

#if REFLECT
/*
 * Alternative RVV implementation POC without seg2 loads.
 */
uint32_t
crc32_rvv_poc_pair(uint8_t *src, size_t n, uint32_t crc)
{
	// assumes src is 64-bit aligned
	size_t VL = __riscv_vsetvlmax_e64m4();
	size_t VLb = VL*8;
	if (n > VLb) {
		vbool16_t modd = __riscv_vreinterpret_b16(__riscv_vmv_v_x_u8m1(0b10101010, __riscv_vsetvlmax_e8m1()));
		vbool16_t meven = __riscv_vmnot(modd, VL);
		vuint64m4_t vfold;
		// can be pre-computed in a LUT for log2(VL), as above in *_seg
		uint64_t rpoly = bitrev(POLY) >> 32;
		vfold = __riscv_vmv_v_x_u64m4(bitrev(xnmodp(VL*64+32, rpoly, 32)) >> 31, VL);
		vfold = __riscv_vmerge(vfold, bitrev(xnmodp(VL*64-32, rpoly, 32)) >> 31, modd, VL);
		vuint64m4_t vacc = __riscv_vle64_v_u64m4((uint64_t*)src, VL);
		vacc = __riscv_vxor_tu(vacc, vacc, crc, 1);
		n -= VLb, src += VLb;
		for (; n > VLb; n -= VLb, src += VLb) {
			vuint64m4_t v = __riscv_vle64_v_u64m4((uint64_t*)src, VL);
			vuint64m4_t vlolo = __riscv_vclmul(vacc,  vfold, VL);
			vuint64m4_t vhihi = __riscv_vclmulh(vacc, vfold, VL);
			vuint64m4_t vlohi = __riscv_vslide1up_mu(  modd,  vlolo, vhihi, 0, VL);
			vuint64m4_t vhilo = __riscv_vslide1down_mu(meven, vhihi, vlolo, 0, VL);
			vacc = __riscv_vxor(v, __riscv_vxor(vlohi, vhilo, VL), VL);
		}
		uint64_t buf[VL];
		__riscv_vse64(buf, vacc, VL);
		crc = crc32_short((void*)buf, sizeof buf, 0);
	}
	return crc32_short(src, n, crc);
}
#endif

#if defined(TEST)

#include <stdio.h>
#include "utils.h"

int
main(void)
{
	init_poly();

	RandU64 r = {123, 4324, (uintptr_t)&r};
	uint32_t crc = 1235678;
	_Alignas(8) uint8_t src[1024*4];
	for (size_t i = 0; i < sizeof src; ++i) src[i] = randu64(&r);

	printf("ref:          %u\n", crc32_ref(src, sizeof src, crc));
	printf("short:        %u\n", crc32_short(src, sizeof src, crc));
	printf("rvv_seg:      %u\n", crc32_rvv_seg(src, sizeof src, crc));
	printf("rvv_poc_seg:  %u\n", crc32_rvv_poc_seg(src, sizeof src, crc));
#if REFLECT
	printf("ref_poc_pair: %u\n", crc32_rvv_poc_pair(src, sizeof src, crc));
#endif
	fflush(stdout);

	for (size_t i = 0; i < (sizeof src)-16; ++i) {
		for (size_t j = 0; j < 16; ++j) {
			uint32_t ref = crc32_ref(src+j, i, crc);
			if (crc32_short(src+j, i, crc) != ref)        printf("ERR short: %zu %zu\n", i, j);
			if (crc32_rvv_seg(src+j, i, crc) != ref)      printf("ERR rvv_seg: %zu %zu\n", i, j);
			if (crc32_rvv_poc_seg(src+j, i, crc) != ref)  printf("ERR rvv_poc_seg: %zu %zu\n", i, j);
#if REFLECT
			if (crc32_rvv_poc_pair(src+j, i, crc) != ref) printf("ERR rvv_poc_pair: %zu %zu\n", i, j);
#endif
		}
	}

}

#endif

