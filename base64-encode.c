#include <riscv_vector.h>

static size_t
b64_encode_scalar(uint8_t *dst, const uint8_t *src, size_t length, const uint8_t lut[16])
{
	uint8_t *dstBeg = dst;
	for (; length >= 3; length -= 3, src += 3, dst += 4) {
		uint32_t u32 = src[0] << 16 | src[1] << 8 | src[2];
		dst[0] = lut[(u32 >> 18) & 63];
		dst[1] = lut[(u32 >> 12) & 63];
		dst[2] = lut[(u32 >>  6) & 63];
		dst[3] = lut[(u32 >>  0) & 63];
	}
	if (length > 0) {
		uint32_t u32 = src[0] << 8 | (length > 1 ? src[1] : 0);
		*dst++ =              lut[(u32 >> 10) & 63];
		*dst++ =              lut[(u32 >> 4) & 63];
		*dst++ = length > 1 ? lut[(u32 << 2) & 63] : '=';
		*dst++ =                                     '=';
	}
	return dst - dstBeg;
}

#define vrgather_u8m1x4(tbl, idx)                                               \
	__riscv_vcreate_v_u8m1_u8m4(                                            \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 0), \
		                              __riscv_vsetvlmax_e8m1()),        \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 1), \
		                              __riscv_vsetvlmax_e8m1()),        \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 2), \
		                              __riscv_vsetvlmax_e8m1()),        \
		__riscv_vrgather_vv_u8m1(tbl, __riscv_vget_v_u8m4_u8m1(idx, 3), \
		                              __riscv_vsetvlmax_e8m1()))

#define vrgather_u8m2x2(tbl, idx)                                               \
	__riscv_vcreate_v_u8m2_u8m4(                                            \
		__riscv_vrgather_vv_u8m2(tbl, __riscv_vget_v_u8m4_u8m2(idx, 0), \
		                              __riscv_vsetvlmax_e8m2()),        \
		__riscv_vrgather_vv_u8m2(tbl, __riscv_vget_v_u8m4_u8m2(idx, 1), \
		                              __riscv_vsetvlmax_e8m2()))

static size_t
b64_encode_rvv(uint8_t *dst, const uint8_t *src, size_t length, const uint8_t lut[16])
{
	uint8_t *dstBeg = dst;
	const size_t VL = __riscv_vsetvlmax_e8m1(), VL2 = VL*2, VL4 = VL*4;

	// 0 0 0 0 | 3 3 3 3 | 6 6 6 6 | ... | vshuf = (vid >> 2) * 3
	// 1 0 2 1 | 4 3 5 4 | 7 6 8 7 | ... | ((vid >> 2) * 3) + on every u32 (1,0,2,1)
	vuint8m1_t vshuf = __riscv_vmul(__riscv_vsrl(__riscv_vid_v_u8m1(VL), 2, VL), 3, VL);
	vshuf = __riscv_vreinterpret_u8m1(__riscv_vadd(__riscv_vreinterpret_u32m1(vshuf), 0x01020001, VL));

	vuint16m2_t vshuf16 = __riscv_vmul(__riscv_vsrl(__riscv_vid_v_u16m2(VL), 2, VL), 3, VL);
	vshuf16 = __riscv_vreinterpret_u16m2(__riscv_vadd(__riscv_vreinterpret_u64m2(vshuf16), 0x0001000200000001, VL));

	vuint16m4_t va6 =__riscv_vreinterpret_u16m4(__riscv_vmv_v_x_u32m4(0x0006000a, VL));
	vuint16m4_t v48 =__riscv_vreinterpret_u16m4(__riscv_vmv_v_x_u32m4(0x00080004, VL));
	vbool2_t modd = __riscv_vmsne(__riscv_vand(__riscv_vid_v_u8m4(VL4), 1, VL4), 0, VL4);
	vuint8m4_t v63 = __riscv_vmv_v_x_u8m4(63, VL4);

	vuint8m4_t vlut = __riscv_vle8_v_u8m4(lut, 64);
	for (; length >= VL4; length -= VL*3, src += VL*3, dst += VL4) {
		vuint8m1_t v0 = __riscv_vle8_v_u8m1(src + (VL/4)*0, VL);
		vuint8m1_t v1 = __riscv_vle8_v_u8m1(src + (VL/4)*3, VL);
		vuint8m1_t v2 = __riscv_vle8_v_u8m1(src + (VL/4)*6, VL);
		vuint8m1_t v3 = __riscv_vle8_v_u8m1(src + (VL/4)*9, VL);

		if (VL <= 256) {
			v0 = __riscv_vrgather(v0, vshuf, VL);
			v1 = __riscv_vrgather(v1, vshuf, VL);
			v2 = __riscv_vrgather(v2, vshuf, VL);
			v3 = __riscv_vrgather(v3, vshuf, VL);
		} else {
			v0 = __riscv_vrgatherei16(v0, vshuf16, VL);
			v1 = __riscv_vrgatherei16(v1, vshuf16, VL);
			v2 = __riscv_vrgatherei16(v2, vshuf16, VL);
			v3 = __riscv_vrgatherei16(v3, vshuf16, VL);
		}

		// v =    bbbbcccc,aaaaaabb,  ccdddddd,bbbbcccc, ...
		// v16 = [aaaaaabb|bbbbcccc],[bbbbcccc|ccdddddd]
		vuint8m4_t v = __riscv_vcreate_v_u8m1_u8m4(v0, v1, v2, v3);
		vuint16m4_t v16 = __riscv_vreinterpret_u16m4(v);

		// ac =  [00000000|00aaaaaa],[000000bb|bbcccccc]
		// ac =   00aaaaaa,00000000,  bbcccccc,000000bb
		vuint8m4_t ac = __riscv_vreinterpret_u8m4(__riscv_vsrl(v16, va6, VL4));
		// bd =  [aabbbbbb|cccc0000],[ccdddddd|00000000]
		// bd =   cccc0000,aabbbbbb,  00000000,ccdddddd
		vuint8m4_t bd = __riscv_vreinterpret_u8m4(__riscv_vsll(v16, v48, VL4));

		vuint8m4_t abcd = __riscv_vmerge(ac, bd, modd, VL4);
		abcd = __riscv_vand(abcd, v63, VL4);

		vuint8m4_t vout;
		/**/ if (VL >= 64)
			vout = vrgather_u8m1x4(__riscv_vlmul_trunc_u8m1(vlut), abcd);
		else if (VL2 >= 64)
			vout = vrgather_u8m2x2(__riscv_vlmul_trunc_u8m2(vlut), abcd);
		else /* VL4 is always >= 64, because V mandates VLEN >= 128 */
			vout = __riscv_vrgather(vlut, abcd, VL4);

		__riscv_vse8(dst, vout, VL4);
	}
	return (dst - dstBeg) + b64_encode_scalar(dst, src, length, lut);
}

static size_t
b64_encode_rvvseg(uint8_t *dst, const uint8_t *src, size_t length, const uint8_t lut[16])
{
	uint8_t *dstBeg = dst;
	const size_t VL = __riscv_vsetvlmax_e8m1(), VL2 = VL*2, VL4 = VL*4;

	vuint8m1_t v63 = __riscv_vmv_v_x_u8m1(63, VL);

	vuint8m4_t vlut = __riscv_vle8_v_u8m4(lut, 64);
	for (; length >= VL4; length -= VL*3, src += VL*3, dst += VL4) {
		vuint8m1x3_t vseg = __riscv_vlseg3e8_v_u8m1x3(src, VL);
		vuint8m1_t v0 = __riscv_vget_u8m1(vseg, 0);
		vuint8m1_t v1 = __riscv_vget_u8m1(vseg, 1);
		vuint8m1_t v2 = __riscv_vget_u8m1(vseg, 2);

		vuint8m1_t vd = __riscv_vand(v2, v63, VL);
		vuint8m1_t vc = __riscv_vand(__riscv_vmacc(__riscv_vsrl(v2, 6, VL), 1<<2, v1, VL), v63, VL);
		vuint8m1_t vb = __riscv_vand(__riscv_vmacc(__riscv_vsrl(v1, 4, VL), 1<<4, v0, VL), v63, VL);
		vuint8m1_t va = __riscv_vsrl(v0, 2, VL);

		vuint8m4_t abcd = __riscv_vcreate_v_u8m1_u8m4(va, vb, vc, vd);

		/**/ if (VL >= 64)
			abcd = vrgather_u8m1x4(__riscv_vlmul_trunc_u8m1(vlut), abcd);
		else if (VL2 >= 64)
			abcd = vrgather_u8m2x2(__riscv_vlmul_trunc_u8m2(vlut), abcd);
		else /* VL4 is always >= 64, because V mandates VLEN >= 128 */
			abcd = __riscv_vrgather(vlut, abcd, VL4);
		va = __riscv_vget_u8m1(abcd, 0);
		vb = __riscv_vget_u8m1(abcd, 1);
		vc = __riscv_vget_u8m1(abcd, 2);
		vd = __riscv_vget_u8m1(abcd, 3);
		__riscv_vsseg4e8_v_u8m1x4(dst, __riscv_vcreate_v_u8m1x4(va, vb, vc, vd), VL);
	}
	return (dst - dstBeg) + b64_encode_scalar(dst, src, length, lut);
}

static uint8_t base64LUT[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	"abcdefghijklmnopqrstuvwxyz"
	"0123456789"
	"+/";

#if defined(FUZZ)

#include <assert.h>
#include <string.h>

#define N 1024
static uint8_t ref[N*2], rvv[N*2], seg[N*2];

int
LLVMFuzzerTestOneInput(const uint8_t *src, size_t size)
{
	if (size > N) return -1;
	size_t nref = b64_encode_scalar(ref, src, size, (uint8_t*)base64LUT);
	size_t nrvv = b64_encode_rvv(rvv, src, size, (uint8_t*)base64LUT);
	size_t nseg = b64_encode_rvvseg(seg, src, size, (uint8_t*)base64LUT);
	assert(nref == nrvv);
	assert(nref == nseg);
	assert(memcmp(ref, rvv, nref) == 0);
	assert(memcmp(ref, seg, nref) == 0);
	return 0;
}

#elif defined(BENCH)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * C908 VLEN=128     X60 VLEN=256:
 * ref: 1.000000 T   ref: 1.000000 T
 * rvv: 0.626652 T   rvv: 0.089777 T
 * seg: 0.864453 T   seg: 0.082051 T
 */

int
main(int argc, char **argv)
{
	if (argc < 2) return 1;
	FILE *f = fopen(argv[1], "r");
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);

	uint8_t *src = malloc(size);
	uint8_t *dst = malloc(size * 2);
	fread(src, 1, size, f);
	fclose(f);

	clock_t beg;
	for (size_t i = 0; i < 5; ++i) {
		beg = clock();
		b64_encode_scalar(dst, src, size, base64LUT);
		printf("ref: %f secs\n", (clock() - beg) * 1.0/CLOCKS_PER_SEC);
		beg = clock();
		b64_encode_rvv(dst, src, size, base64LUT);
		printf("rvv: %f secs\n", (clock() - beg) * 1.0/CLOCKS_PER_SEC);
		beg = clock();
		b64_encode_rvvseg(dst, src, size, base64LUT);
		printf("seg: %f secs\n", (clock() - beg) * 1.0/CLOCKS_PER_SEC);
	}
}

#elif defined(TEST)

#include <stdio.h>

int
main(void)
{
	uint8_t src[] =
		"This is a test 1 + 2 / 3 wow!! dalisjd lkaskd jljasd"
		"This is a test 1 + 2 / 3 wow!! dalisjd lkaskd jljasd";
	uint8_t ref[2*sizeof src];
	uint8_t rvv[2*sizeof src];
	uint8_t seg[2*sizeof src];
	size_t nref = b64_encode_scalar(ref, src, sizeof src - 1, base64LUT);
	size_t nrvv = b64_encode_rvv(rvv, src, sizeof src - 1, base64LUT);
	size_t nseg = b64_encode_rvvseg(seg, src, sizeof src - 1, base64LUT);
	printf("in:  '%.*s' %zu\n", (int)sizeof src, src, sizeof src - 1);
	printf("ref: '%.*s' %zu\n", (int)nref, ref, nref);
	printf("rvv: '%.*s' %zu\n", (int)nrvv, rvv, nrvv);
	printf("seg: '%.*s' %zu\n", (int)nseg, seg, nseg);
	return 0;
}

#endif
