#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>
#include <riscv_bitmanip.h>

static uint32_t
crc32_iscsi_ref(uint8_t *src, size_t n, uint32_t crc)
{
	const uint32_t poly = 0x82F63B78;
	for (size_t i = 0; i < n; i++) {
		crc = crc ^ src[i];
		for (size_t j = 0; j < 8; j++)
			crc = (crc & 0x1ULL) ? (crc >> 1) ^ poly : (crc >> 1);
	}
	return crc;
}

/* source: https://stackoverflow.com/a/21201497 public domain by Mark Adler */
static uint64_t
xnmodp(uint64_t n, uint64_t poly, uint64_t deg)
{
	uint64_t mod, mask;
	if (n < deg) return poly;
	mod = poly &= mask = ((((uint64_t)1 << (--deg)) - 1) << 1) + 1;
	for (; --n > deg;) mod = (mod << 1) ^ ((mod >> deg) & 1 ? poly : 0);
	return mod & mask;
}
static uint64_t bitrev(uint64_t x)
{
#if  __riscv_zbb && __riscv_zbkb
	return __riscv_brev8_64(__riscv_rev8_64(x));
#else
	x = ((x >>  1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1);
	x = ((x >>  2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2);
	x = ((x >>  4) & 0x0f0f0f0f0f0f0f0f) | ((x & 0x0f0f0f0f0f0f0f0f) << 4);
#if  __riscv_zbb
	x = __riscv_rev8_64(x);
#else
	x = ((x >>  8) & 0x00ff00ff00ff00ff) | ((x & 0x00ff00ff00ff00ff) << 8);
	x = ((x >> 16) & 0x0000ffff0000ffff) | ((x & 0x0000ffff0000ffff) << 16);
	x = ((x >> 32) & 0x00000000ffffffff) | ((x & 0x00000000ffffffff) << 32);
#endif
	return x;
#endif
}

uint32_t
crc32_iscsi_rvv_seg(uint8_t *src, size_t n, uint32_t crc)
{
	// assumes src is 64-bit aligned
	uint64_t poly = 0x82f63b78;
	size_t VL = __riscv_vsetvlmax_e64m4();
	size_t VLb = VL*2*8;
	if (n > VLb) {
		// can be pre-computed in a LUT for log2(VL*2)
		uint64_t rpoly = bitrev(poly)>>32;
		vuint64m4_t vfold0 = __riscv_vmv_v_x_u64m4(bitrev(xnmodp(VL*2*64+32, rpoly, 32)) >> 31, VL);
		vuint64m4_t vfold1 = __riscv_vmv_v_x_u64m4(bitrev(xnmodp(VL*2*64-32, rpoly, 32)) >> 31, VL);
		vuint64m4x2_t vacc = __riscv_vlseg2e64_v_u64m4x2((uint64_t*)src, VL);
		vuint64m4_t vacc0 = __riscv_vget_u64m4(vacc, 0);
		vuint64m4_t vacc1 = __riscv_vget_u64m4(vacc, 1);
		vacc0 = __riscv_vxor_tu(vacc0, vacc0, crc, 1);
		n -= VLb, src += VLb;
		for (; n > VLb; n -= VLb, src += VLb) {
			vuint64m4x2_t v = __riscv_vlseg2e64_v_u64m4x2((uint64_t*)src, VL);
			vuint64m4_t vlo0 = __riscv_vclmul( vacc0, vfold0, VL);
			vuint64m4_t vlo1 = __riscv_vclmul( vacc1, vfold1, VL);
			vuint64m4_t vhi0 = __riscv_vclmulh(vacc0, vfold0, VL);
			vuint64m4_t vhi1 = __riscv_vclmulh(vacc1, vfold1, VL);
			vacc0 = __riscv_vxor(__riscv_vget_u64m4(v, 0), __riscv_vxor(vlo0, vlo1, VL), VL);
			vacc1 = __riscv_vxor(__riscv_vget_u64m4(v, 1), __riscv_vxor(vhi0, vhi1, VL), VL);
		}
		vacc = __riscv_vcreate_v_u64m4x2(vacc0, vacc1);
		// you should fold to LMUL=1 and loop until vl<2, but for POC this is simpler
		uint64_t buf[VL*2];
		__riscv_vsseg2e64(buf, vacc, VL);
		crc = crc32_iscsi_ref((void*)buf, sizeof buf, 0);
	}
	return crc32_iscsi_ref(src, n, crc);
}

uint32_t
crc32_iscsi_rvv_pair(uint8_t *src, size_t n, uint32_t crc)
{
	// assumes src is 64-bit aligned
	uint64_t poly = 0x82f63b78;
	size_t VL = __riscv_vsetvlmax_e64m4();
	size_t VLb = VL*8;
	if (n > VLb) {
		vbool16_t modd = __riscv_vreinterpret_b16(__riscv_vmv_v_x_u8m1(0b10101010, __riscv_vsetvlmax_e8m1()));
		vbool16_t meven = __riscv_vmnot(modd, VL);
		vuint64m4_t vfold;
		// can be pre-computed in a LUT for log2(VL)
		uint64_t rpoly = bitrev(poly)>>32;
		vfold = __riscv_vmv_v_x_u64m4(   bitrev(xnmodp(VL*64+32, rpoly, 32)) >> 31, VL);
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
		// you should fold to LMUL=1 and loop until vl<2, but for POC this is simpler
		uint64_t buf[VL];
		__riscv_vse64(buf, vacc, VL);
		crc = crc32_iscsi_ref((void*)buf, sizeof buf, 0);
	}
	return crc32_iscsi_ref(src, n, crc);
}

#if defined(TEST)

#include <stdio.h>
int
main(void)
{
	uint32_t crc = 123;
	_Alignas(8) uint8_t src[1024*4];
	for (size_t i = 0; i < sizeof src; ++i) src[i] = i*i^(i*7) + i*69 + 123;

	printf("ref:      %u\n", crc32_iscsi_ref(src, sizeof src, crc));
	printf("rvv_seg:  %u\n", crc32_iscsi_rvv_seg(src, sizeof src, crc));
	printf("ref_pair: %u\n", crc32_iscsi_rvv_pair(src, sizeof src, crc));
}

#endif
