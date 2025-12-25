/*
 * base64-decode
 */

#include <riscv_vector.h>
#include <assert.h>

static const uint8_t LUTenc[64] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	"abcdefghijklmnopqrstuvwxyz"
	"0123456789"
	"+/";

static uint8_t LUT256[256];

void init_lut()
{
	for (size_t i = 0; i < 256; ++i) LUT256[i] = 0xFF;
	for (size_t i = 0; i <  64; ++i) LUT256[LUTenc[i]] = i;
}

static size_t
b64_decode_scalar(uint8_t *dst, const uint8_t *src, size_t length)
{
	uint8_t *dstBeg = dst;
	assert(length % 4 == 0);
	for (; length > 4; length -= 4, src += 4, dst += 3) {
		uint8_t l0 = LUT256[src[0]];
		uint8_t l1 = LUT256[src[1]];
		uint8_t l2 = LUT256[src[2]];
		uint8_t l3 = LUT256[src[3]];
		if (l0 == 0xFF || l1 == 0xFF || l2 == 0xFF || l3 == 0xFF)
			return 0;
		dst[0] = l0 << 2 | l1 >> 4;
		dst[1] = l1 << 4 | l2 >> 2;
		dst[2] = l2 << 6 | l3;
	}
	if (length > 0) {
		uint8_t s0 = src[0], l0 = LUT256[s0];
		uint8_t s1 = src[1], l1 = LUT256[s1];
		uint8_t s2 = src[2], l2 = LUT256[s2];
		uint8_t s3 = src[3], l3 = LUT256[s3];
		if (l0 == 0xFF || l1 == 0xFF) return 0;
		if (l2 == 0xFF && s2 != '=') return 0;
		if (l3 == 0xFF && s3 != '=') return 0;
		if (s2 == '=' && s3 != '=') return 0;
		*dst++ = l0 << 2 | l1 >> 4;
		if (s2 != '=') *dst++ = l1 << 4 | l2 >> 2;
		if (s3 != '=') *dst++ = l2 << 6 | l3;
	}
	return dst - dstBeg;
}

#define vrgather_u8m1x4vl(tbl, idx, vl) \
	__riscv_vcreate_v_u8m1_u8m4( \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 0), vl), \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 1), vl), \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 2), vl), \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 3), vl))

#define vrgather_u8m2x2(tbl, idx) \
	__riscv_vcreate_v_u8m2_u8m4( \
		__riscv_vrgather_vv_u8m2(tbl, __riscv_vget_v_u8m4_u8m2(idx, 0), \
		                              __riscv_vsetvlmax_e8m2()),        \
		__riscv_vrgather_vv_u8m2(tbl, __riscv_vget_v_u8m4_u8m2(idx, 1), \
		                              __riscv_vsetvlmax_e8m2()))

#define vrgather_u8m1x4(tbl, idx) vrgather_u8m1x4vl(tbl, idx, __riscv_vsetvlmax_e8m1())

size_t
b64_decode_rvv_3xLUT16(uint8_t *dst, const uint8_t *src, size_t len)
{
	uint8_t *dstBeg = dst;

	const size_t VL = __riscv_vsetvlmax_e8m1(), VL3 = VL*3, VL4 = VL*4;
	static const uint8_t err_lo[16] = { 21, 17, 17, 17, 17,  17,  17,  17,  17, 17, 19, 26, 27, 27, 27, 26 };
	static const uint8_t err_hi[16] = { 16, 16, 1,  2,  4,   8,   4,   8,   16, 16, 16, 16, 16, 16, 16, 16 };
	static const uint8_t off[16]    = { 0,  16, 19, 4,  -65, -65, -71, -71, 0,  0,  0,  0,  0,  0,  0,  0 };

	const vuint8m1_t vErrLo = __riscv_vle8_v_u8m1(err_lo, sizeof err_lo);
	const vuint8m1_t vErrHi = __riscv_vle8_v_u8m1(err_hi, sizeof err_hi);
	const vuint8m1_t vOff   = __riscv_vle8_v_u8m1(off, sizeof off);

	len = len/4;
	size_t tail = len == 0 ? 0 : 1;
	len -= tail;
	for (; len > VL; len -= VL, dst += VL3, src += VL4) {
		vuint8m1x4_t vseg = __riscv_vlseg4e8_v_u8m1x4(src, VL);
		vuint8m1_t v0 = __riscv_vget_u8m1(vseg, 0), v1 = __riscv_vget_u8m1(vseg, 1);
		vuint8m1_t v2 = __riscv_vget_u8m1(vseg, 2), v3 = __riscv_vget_u8m1(vseg, 3);

		vuint8m4_t v = __riscv_vcreate_v_u8m1_u8m4(v0, v1, v2, v3);
		vuint8m4_t vHi = __riscv_vsrl(v, 4, VL4);
		vuint8m4_t vLo = __riscv_vand(v, 0xF, VL4);

		/* only works for vl=VLMAX, se we need a seperate RVV tail */
		vuint8m4_t err = __riscv_vand(
				vrgather_u8m1x4(vErrLo, vLo),
				vrgather_u8m1x4(vErrHi, vHi), VL4);
		if (__riscv_vfirst(__riscv_vmsne(err, 0, VL4), VL4) >= 0) {
			return 0;
		}

		v = __riscv_vmerge(v, '/'-3, __riscv_vmseq(v, '/', VL4), VL4);
		vuint8m4_t bits = __riscv_vadd(v, vrgather_u8m1x4vl(vOff, vHi, VL), VL4);
		vuint8m1_t b0 = __riscv_vget_u8m1(bits, 0), b1 = __riscv_vget_u8m1(bits, 1);
		vuint8m1_t b2 = __riscv_vget_u8m1(bits, 2), b3 = __riscv_vget_u8m1(bits, 3);

		vuint8m1_t o0 = __riscv_vmacc(__riscv_vsrl(b1, 4, VL), 1<<2, b0, VL);
		vuint8m1_t o1 = __riscv_vmacc(__riscv_vsrl(b2, 2, VL), 1<<4, b1, VL);
		vuint8m1_t o2 = __riscv_vmacc(b3, 1<<6, b2, VL);
		__riscv_vsseg3e8_v_u8m1x3(dst, __riscv_vcreate_v_u8m1x3(o0, o1, o2),  VL);
	}

	if (len > 0) {
		size_t vl = __riscv_vsetvl_e8m1(len);
		vuint8m1x4_t vseg = __riscv_vlseg4e8_v_u8m1x4(src, vl);
		vuint8m1_t v0 = __riscv_vget_u8m1(vseg, 0), v1 = __riscv_vget_u8m1(vseg, 1);
		vuint8m1_t v2 = __riscv_vget_u8m1(vseg, 2), v3 = __riscv_vget_u8m1(vseg, 3);

		vuint8m4_t v = __riscv_vcreate_v_u8m1_u8m4(v0, v1, v2, v3);
		vuint8m4_t vHi = __riscv_vsrl(v, 4, VL4);
		vuint8m4_t vLo = __riscv_vand(v, 0xF, VL4);

		#define LUT_ERR(lo,hi,i,vl) __riscv_vand( \
			__riscv_vrgather(vErrLo, __riscv_vget_u8m1(lo, i), vl), \
			__riscv_vrgather(vErrHi, __riscv_vget_u8m1(hi, i), vl), vl)

		vuint8m1_t err0 = LUT_ERR(vLo, vHi, 0, vl), err1 = LUT_ERR(vLo, vHi, 1, vl);
		vuint8m1_t err2 = LUT_ERR(vLo, vHi, 2, vl), err3 = LUT_ERR(vLo, vHi, 3, vl);
		if (__riscv_vfirst(__riscv_vmsne(err0, 0, vl), vl) >= 0) return 0;
		if (__riscv_vfirst(__riscv_vmsne(err1, 0, vl), vl) >= 0) return 0;
		if (__riscv_vfirst(__riscv_vmsne(err2, 0, vl), vl) >= 0) return 0;
		if (__riscv_vfirst(__riscv_vmsne(err3, 0, vl), vl) >= 0) return 0;

		v = __riscv_vmerge(v, '/'-3, __riscv_vmseq(v, '/', VL4), VL4);
		vuint8m4_t bits = __riscv_vadd(v, vrgather_u8m1x4vl(vOff, vHi, vl), VL4);
		vuint8m1_t b0 = __riscv_vget_u8m1(bits, 0), b1 = __riscv_vget_u8m1(bits, 1);
		vuint8m1_t b2 = __riscv_vget_u8m1(bits, 2), b3 = __riscv_vget_u8m1(bits, 3);

		vuint8m1_t o0 = __riscv_vmacc(__riscv_vsrl(b1, 4, vl), 1<<2, b0, vl);
		vuint8m1_t o1 = __riscv_vmacc(__riscv_vsrl(b2, 2, vl), 1<<4, b1, vl);
		vuint8m1_t o2 = __riscv_vmacc(b3, 1<<6, b2, vl);
		__riscv_vsseg3e8_v_u8m1x3(dst, __riscv_vcreate_v_u8m1x3(o0, o1, o2),  vl);
		dst += vl*3;
		src += vl*4;
	}

	/* TODO: dedicated tail */
	if (tail > 0) {
		uint8_t s0 = src[0], l0 = LUT256[s0];
		uint8_t s1 = src[1], l1 = LUT256[s1];
		uint8_t s2 = src[2], l2 = LUT256[s2];
		uint8_t s3 = src[3], l3 = LUT256[s3];
		if (l0 == 0xFF || l1 == 0xFF) return 0;
		if (l2 == 0xFF && s2 != '=') return 0;
		if (l3 == 0xFF && s3 != '=') return 0;
		if (s2 == '=' && s3 != '=') return 0;
		*dst++ = l0 << 2 | l1 >> 4;
		if (s2 != '=') *dst++ = l1 << 4 | l2 >> 2;
		if (s3 != '=') *dst++ = l2 << 6 | l3;
	}
	return dst - dstBeg;
}

size_t
b64_decode_rvv_LUT128(uint8_t *dst, const uint8_t *src, size_t len)
{
	if (__riscv_vlenb() <= 128/8)
		return b64_decode_rvv_3xLUT16(dst, src, len);
	uint8_t *dstBeg = dst;

	const vuint8m4_t vlut = __riscv_vle8_v_u8m4(LUT256, 128);
	const size_t VL = __riscv_vsetvlmax_e8m1(), VL2 = VL*2, VL3 = VL*3, VL4 = VL*4;

	len = len/4;
	size_t tail = len == 0 ? 0 : 1;
	len -= tail;
	for (; len > VL; len -= VL, dst += VL3, src += VL4) {
		vuint8m1x4_t vseg = __riscv_vlseg4e8_v_u8m1x4(src, VL);
		vuint8m1_t v0 = __riscv_vget_u8m1(vseg, 0);
		vuint8m1_t v1 = __riscv_vget_u8m1(vseg, 1);
		vuint8m1_t v2 = __riscv_vget_u8m1(vseg, 2);
		vuint8m1_t v3 = __riscv_vget_u8m1(vseg, 3);

		vuint8m4_t v = __riscv_vcreate_v_u8m1_u8m4(v0, v1, v2, v3), vb;

		/**/ if (VL >= 128)
			vb = vrgather_u8m1x4vl(__riscv_vlmul_trunc_u8m1(vlut), v, VL);
		else if (VL2 >= 128)
			vb = vrgather_u8m2x2(__riscv_vlmul_trunc_u8m2(vlut), v);
		else
			vb = __riscv_vrgather(vlut, v, VL4);
		vint8m4_t ve = __riscv_vreinterpret_i8m4(__riscv_vor(vb, v, VL4));

		if (__riscv_vfirst(__riscv_vmslt(ve, 0, VL4), VL4) >= 0) {
			return 0;
		}

		vuint8m1_t b0 = __riscv_vget_u8m1(vb, 0);
		vuint8m1_t b1 = __riscv_vget_u8m1(vb, 1);
		vuint8m1_t b2 = __riscv_vget_u8m1(vb, 2);
		vuint8m1_t b3 = __riscv_vget_u8m1(vb, 3);

		vuint8m1_t o0 = __riscv_vmacc(__riscv_vsrl(b1, 4, VL), 1<<2, b0, VL);
		vuint8m1_t o1 = __riscv_vmacc(__riscv_vsrl(b2, 2, VL), 1<<4, b1, VL);
		vuint8m1_t o2 = __riscv_vmacc(b3, 1<<6, b2, VL);
		__riscv_vsseg3e8_v_u8m1x3(dst, __riscv_vcreate_v_u8m1x3(o0, o1, o2), VL);
	}

	if (len > 0) {
		size_t vl = __riscv_vsetvl_e8m1(len);
		vuint8m1x4_t vseg = __riscv_vlseg4e8_v_u8m1x4(src, vl);
		vuint8m1_t v0 = __riscv_vget_u8m1(vseg, 0);
		vuint8m1_t v1 = __riscv_vget_u8m1(vseg, 1);
		vuint8m1_t v2 = __riscv_vget_u8m1(vseg, 2);
		vuint8m1_t v3 = __riscv_vget_u8m1(vseg, 3);

		vuint8m4_t v = __riscv_vcreate_v_u8m1_u8m4(v0, v1, v2, v3), vb;

		/**/ if (VL >= 128)
			vb = vrgather_u8m1x4vl(__riscv_vlmul_trunc_u8m1(vlut), v, vl);
		else if (VL2 >= 128)
			vb = vrgather_u8m2x2(__riscv_vlmul_trunc_u8m2(vlut), v);
		else
			vb = __riscv_vrgather(vlut, v, VL4);
		vint8m4_t ve = __riscv_vreinterpret_i8m4(__riscv_vor(vb, v, VL4));

		if (__riscv_vfirst(__riscv_vmslt(__riscv_vget_i8m1(ve, 0), 0, vl), vl) >= 0) return 0;
		if (__riscv_vfirst(__riscv_vmslt(__riscv_vget_i8m1(ve, 1), 0, vl), vl) >= 0) return 0;
		if (__riscv_vfirst(__riscv_vmslt(__riscv_vget_i8m1(ve, 2), 0, vl), vl) >= 0) return 0;
		if (__riscv_vfirst(__riscv_vmslt(__riscv_vget_i8m1(ve, 3), 0, vl), vl) >= 0) return 0;

		vuint8m1_t b0 = __riscv_vget_u8m1(vb, 0);
		vuint8m1_t b1 = __riscv_vget_u8m1(vb, 1);
		vuint8m1_t b2 = __riscv_vget_u8m1(vb, 2);
		vuint8m1_t b3 = __riscv_vget_u8m1(vb, 3);

		vuint8m1_t o0 = __riscv_vmacc(__riscv_vsrl(b1, 4, vl), 1<<2, b0, vl);
		vuint8m1_t o1 = __riscv_vmacc(__riscv_vsrl(b2, 2, vl), 1<<4, b1, vl);
		vuint8m1_t o2 = __riscv_vmacc(b3, 1<<6, b2, vl);
		__riscv_vsseg3e8_v_u8m1x3(dst, __riscv_vcreate_v_u8m1x3(o0, o1, o2),  vl);
		dst += vl*3;
		src += vl*4;
	}

	if (tail > 0) {
		uint8_t s0 = src[0], l0 = LUT256[s0];
		uint8_t s1 = src[1], l1 = LUT256[s1];
		uint8_t s2 = src[2], l2 = LUT256[s2];
		uint8_t s3 = src[3], l3 = LUT256[s3];
		if (l0 == 0xFF || l1 == 0xFF) return 0;
		if (l2 == 0xFF && s2 != '=') return 0;
		if (l3 == 0xFF && s3 != '=') return 0;
		if (s2 == '=' && s3 != '=') return 0;
		*dst++ = l0 << 2 | l1 >> 4;
		if (s2 != '=') *dst++ = l1 << 4 | l2 >> 2;
		if (s3 != '=') *dst++ = l2 << 6 | l3;
	}
	return dst - dstBeg;
}


#if defined(FUZZ)

#include <stdio.h>
#include <assert.h>
#include <string.h>

#if 0
#define ENCODE(i) LUTenc[i]
#else
/* this produces a larger fuzzign corpus */
#define ENCODE(i) (i < 26 ? i+'A' : \
                   i < 52 ? i+'a'-26 : \
                   i < 62 ? i+'0'-52 : \
                   i < 63 ? '+' : '/')
#endif

static size_t
b64_encode_fuzz(uint8_t *dst, const uint8_t *src, size_t length)
{
	uint8_t *dstBeg = dst;
	for (; length >= 3; length -= 3, src += 3, dst += 4) {
		uint32_t u32 = src[0] << 16 | src[1] << 8 | src[2];
		uint8_t i0 = (u32 >> 18) & 63; dst[0] = ENCODE(i0);
		uint8_t i1 = (u32 >> 12) & 63; dst[1] = ENCODE(i1);
		uint8_t i2 = (u32 >>  6) & 63; dst[2] = ENCODE(i2);
		uint8_t i3 = (u32 >>  0) & 63; dst[3] = ENCODE(i3);
	}
	if (length > 0) {
		uint32_t u32 = src[0] << 8 | (length > 1 ? src[1] : 0);
		uint8_t i0 = (u32 >> 10) & 63;
		uint8_t i1 = (u32 >> 4) & 63;
		uint8_t i2 = (u32 << 2) & 63;
		*dst++ =              ENCODE(i0);
		*dst++ =              ENCODE(i1);
		*dst++ = length > 1 ? ENCODE(i2) : '=';
		*dst++ =                           '=';
	}
	return dst - dstBeg;
}

#define N (1024*4)

int
LLVMFuzzerTestOneInput(const uint8_t *src, size_t size)
{
	init_lut();
	if (size > N) return -1;

	/* round trip */
	{
#if __STDC_NO_VLA__
		static uint8_t enc[N*2], ref[N*2], rvv_3xLUT16[N*2], rvv_LUT128[N*2];
#else
		size_t nsrc = (size*4/3+3)/4*4; nsrc = nsrc ? nsrc : 1;
		size_t ndst = nsrc/4*3; ndst = ndst ? ndst : 1;
		uint8_t enc[nsrc], ref[ndst], rvv_3xLUT16[ndst], rvv_LUT128[ndst];
#endif
		size_t nenc = b64_encode_fuzz(enc, src, size);
		size_t nref = b64_decode_scalar(ref, enc, nenc);
		size_t nrvv_3xLUT16 = b64_decode_rvv_3xLUT16(rvv_3xLUT16, enc, nenc);
		size_t nrvv_LUT128 = b64_decode_rvv_LUT128(rvv_LUT128, enc, nenc);

		assert(size == nref);
		assert(size == nrvv_3xLUT16);
		assert(size == nrvv_LUT128);
		assert(memcmp(src, ref, nref) == 0);
		assert(memcmp(ref, rvv_3xLUT16, nref) == 0);
		assert(memcmp(ref, rvv_LUT128, nref) == 0);
	}

	/* direct input */
	{
		size = size/4*4;
#if __STDC_NO_VLA__
		static uint8_t ref[N], rvv_3xLUT16[N], rvv_LUT128[N];
#else
		size_t ndst = size/4*3; ndst = ndst ? ndst : 1;
		uint8_t ref[ndst], rvv_3xLUT16[ndst], rvv_LUT128[ndst];
#endif
		size_t nref = b64_decode_scalar(ref, src, size);
		size_t nrvv_3xLUT16 = b64_decode_rvv_3xLUT16(rvv_3xLUT16, src, size);
		size_t nrvv_LUT128 = b64_decode_rvv_LUT128(rvv_LUT128, src, size);

		assert(nref == nrvv_3xLUT16);
		assert(nref == nrvv_LUT128);
		assert(memcmp(ref, rvv_3xLUT16, nref) == 0);
		assert(memcmp(ref, rvv_LUT128, nref) == 0);
	}

	return 0;
}

#endif
