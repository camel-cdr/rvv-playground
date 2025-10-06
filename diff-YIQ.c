#include <stdint.h>
#include <stddef.h>

/* RVV port of odiff (https://github.com/dmtrKovalenko/odiff)
 * which uses the YIQ NTSC difference equation.
 *
 * See also: "Measuring perceived color difference using YIQ NTSC
 *            transmission color space in mobile applications"
 */
#define YIQ_Y_R_COEFF 0.29889531
#define YIQ_Y_G_COEFF 0.58662247
#define YIQ_Y_B_COEFF 0.11448223

#define YIQ_I_R_COEFF 0.59597799
#define YIQ_I_G_COEFF -0.27417610
#define YIQ_I_B_COEFF -0.32180189

#define YIQ_Q_R_COEFF 0.21147017
#define YIQ_Q_G_COEFF -0.52261711
#define YIQ_Q_B_COEFF 0.31114694

#define YIQ_Y_WEIGHT 0.5053
#define YIQ_I_WEIGHT 0.299
#define YIQ_Q_WEIGHT 0.1957

#define RGB2YIQ(r,g,b,N,T) (r*(T)YIQ_##N##_R_COEFF + g*(T)YIQ_##N##_G_COEFF + b*(T)YIQ_##N##_B_COEFF)

static void
diff_fp64(uint32_t *src1, uint32_t *src2, size_t n, float *diff)
{
	uint8_t *s1 = (uint8_t*)src1, *s2 = (uint8_t*)src2;
	for (size_t i = 0; i < n; ++i, s1 += 4, s2 += 4, ++diff) {
		double vr = s1[0], yr = s2[0];
		double vg = s1[1], yg = s2[1];
		double vb = s1[2], yb = s2[2];
		double va = s1[3], ya = s2[3];

		va *= 1.0/255;
		vg = 255 + (vg - 255) * va;
		vr = 255 + (vr - 255) * va;
		vb = 255 + (vb - 255) * va;

		ya *= 1.0/255;
		yg = 255 + (yg - 255) * ya;
		yr = 255 + (yr - 255) * ya;
		yb = 255 + (yb - 255) * ya;

		double y = RGB2YIQ(vr,vg,vb, Y, double) - RGB2YIQ(yr,yg,yb, Y, double);
		double i = RGB2YIQ(vr,vg,vb, I, double) - RGB2YIQ(yr,yg,yb, I, double);
		double q = RGB2YIQ(vr,vg,vb, Q, double) - RGB2YIQ(yr,yg,yb, Q, double);

		*diff = YIQ_Y_WEIGHT*y*y + YIQ_I_WEIGHT*i*i + YIQ_Q_WEIGHT*q*q;
	}
}

#if __GNUC__
__attribute__((noinline))
#endif
size_t
odiff_fp32(uint32_t *src1, uint32_t *src2, size_t n, float max_delta, uint32_t *diff, uint32_t diffcol)
{
	/* NOTE: for benchmarking, doesn't write to diff on purpose */
	(void)diff;
	(void)diffcol;
	size_t count = 0;
	uint8_t *s1 = (uint8_t*)src1, *s2 = (uint8_t*)src2;
	for (size_t i = 0; i < n; ++i, s1 += 4, s2 += 4) {
		float vr = s1[0], yr = s2[0];
		float vg = s1[1], yg = s2[1];
		float vb = s1[2], yb = s2[2];
		float va = s1[3], ya = s2[3];

#if __GNUC__
		__asm("":::"memory"); /* stop autovec */
#endif

		va *= 1.0f/255;
		vg = 255 + (vg - 255) * va;
		vr = 255 + (vr - 255) * va;
		vb = 255 + (vb - 255) * va;

		ya *= 1.0f/255;
		yg = 255 + (yg - 255) * ya;
		yr = 255 + (yr - 255) * ya;
		yb = 255 + (yb - 255) * ya;

		float y = RGB2YIQ(vr,vg,vb, Y, float) - RGB2YIQ(yr,yg,yb, Y, float);
		float i = RGB2YIQ(vr,vg,vb, I, float) - RGB2YIQ(yr,yg,yb, I, float);
		float q = RGB2YIQ(vr,vg,vb, Q, float) - RGB2YIQ(yr,yg,yb, Q, float);
		float delta = YIQ_Y_WEIGHT*y*y + YIQ_I_WEIGHT*i*i + YIQ_Q_WEIGHT*q*q;
		count += delta > max_delta;
	}
	return count;
}


#if 0
/* simplifying the original equation  */
y = rgb2y(a) - rgb2y(b) = rgb2y(a - b);
i = rgb2i(a) - rgb2i(b) = rgb2i(a - b);
q = rgb2q(a) - rgb2q(b) = rgb2q(a - b);
return (YIQ_Y_WEIGHT * y * y) + (YIQ_I_WEIGHT * i * i) + (YIQ_Q_WEIGHT * q * q);

[r,g,b] = a-b;
y = ((r * YIQ_Y_R_COEFF) + (g * YIQ_Y_G_COEFF) + (b * YIQ_Y_B_COEFF)) * YIQ_Y_WEIGHT_SQRT;
i = ((r * YIQ_I_R_COEFF) + (g * YIQ_I_G_COEFF) + (b * YIQ_I_B_COEFF)) * YIQ_I_WEIGHT_SQRT;
q = ((r * YIQ_Q_R_COEFF) + (g * YIQ_Q_G_COEFF) + (b * YIQ_Q_B_COEFF)) * YIQ_Q_WEIGHT_SQRT;
return y*y + i*i + q*q;

return (r*C1 + g*C2 + b*C3)**2
      +(r*C4 + g*C5 + b*C6)**2
      +(r*C7 + g*C8 + b*C9)**2;

return r*C1*(r*C1 + g*2*C2 + b*2*C3) + (g*C2)**2 + b*C3*(g*2*C2 + b*C3)
     + r*C4*(r*C4 + g*2*C5 + b*2*C6) + (g*C5)**2 + b*C6*(g*2*C5 + b*C6)
     + r*C7*(r*C7 + g*2*C8 + b*2*C9) + (g*C8)**2 + b*C9*(g*2*C8 + b*C9);

return r*r*C1*C1 + r*g*2*C2*C1 + r*b*2*C3*C1 + (g*C2)**2 + b*g*2*C2*C3 + b*b*C3*C3
     + r*r*C4*C4 + r*g*2*C5*C4 + r*b*2*C6*C4 + (g*C5)**2 + b*g*2*C5*C6 + b*b*C6*C6
     + r*r*C7*C7 + r*g*2*C8*C7 + r*b*2*C9*C7 + (g*C8)**2 + b*g*2*C8*C9 + b*b*C9*C9;

return r*r * (C1*C1+C4*C4+C7*C7)
     + r*g*2*(C2*C1+C5*C4+C8*C7)
     + r*b*2*(C3*C1+C6*C4+C9*C7)
     + g*g * (C2*C2+C5*C5+C8*C8)
     + b*g*2*(C2*C3+C5*C6+C8*C9)
     + b*b * (C3*C3+C6*C6+C9*C9);

// 24 -> 18 -> 15 -> 12 instructions
return r*(r*Y1 + g*Y2 + b*Y3) + g*(g*Y4 + b*Y5) + b*b*Y6;
#endif

#define YIQ_Y_WEIGHT_SQRT 0.7108445681019163
#define YIQ_I_WEIGHT_SQRT 0.5468089245796927
#define YIQ_Q_WEIGHT_SQRT 0.4423799272118933

#define C1 (YIQ_Y_R_COEFF*YIQ_Y_WEIGHT_SQRT)
#define C2 (YIQ_Y_G_COEFF*YIQ_Y_WEIGHT_SQRT)
#define C3 (YIQ_Y_B_COEFF*YIQ_Y_WEIGHT_SQRT)
#define C4 (YIQ_I_R_COEFF*YIQ_I_WEIGHT_SQRT)
#define C5 (YIQ_I_G_COEFF*YIQ_I_WEIGHT_SQRT)
#define C6 (YIQ_I_B_COEFF*YIQ_I_WEIGHT_SQRT)
#define C7 (YIQ_Q_R_COEFF*YIQ_Q_WEIGHT_SQRT)
#define C8 (YIQ_Q_G_COEFF*YIQ_Q_WEIGHT_SQRT)
#define C9 (YIQ_Q_B_COEFF*YIQ_Q_WEIGHT_SQRT)

#define Y1 (  (C1*C1+C4*C4+C7*C7)) /*  0.160096 */
#define Y2 (2*(C2*C1+C5*C4+C8*C7)) /*  0.036226 */
#define Y3 (2*(C3*C1+C6*C4+C9*C7)) /* -0.054354 */
#define Y4 (  (C2*C2+C5*C5+C8*C8)) /*  0.249815 */
#define Y5 (2*(C2*C3+C5*C6+C8*C9)) /*  0.056986 */
#define Y6 (  (C3*C3+C6*C6+C9*C9)) /*  0.056532 */

#if __riscv_xtheadvector
/* xtheadvector fixes */
#define __riscv_vzext_vf2(v, vl) __riscv_vwaddu_vx(v, 0, vl)
#endif

#include <riscv_vector.h>

#if __riscv_zvfh
static inline vfloat16m2_t
#else
static inline vfloat32m4_t
#endif
rvv_yiq_diff(vuint8m1x4_t v4, vuint8m1x4_t y4, size_t vl)
{
#if __riscv_zvfh
	/* use fp16, not sure if the error is in acceptable range */
	vfloat16m2_t vr = __riscv_vfwcvt_f(__riscv_vget_u8m1(v4, 0), vl);
	vfloat16m2_t vg = __riscv_vfwcvt_f(__riscv_vget_u8m1(v4, 1), vl);
	vfloat16m2_t vb = __riscv_vfwcvt_f(__riscv_vget_u8m1(v4, 2), vl);
	vfloat16m2_t va = __riscv_vfwcvt_f(__riscv_vget_u8m1(v4, 3), vl);
	vfloat16m2_t yr = __riscv_vfwcvt_f(__riscv_vget_u8m1(y4, 0), vl);
	vfloat16m2_t yg = __riscv_vfwcvt_f(__riscv_vget_u8m1(y4, 1), vl);
	vfloat16m2_t yb = __riscv_vfwcvt_f(__riscv_vget_u8m1(y4, 2), vl);
	vfloat16m2_t ya = __riscv_vfwcvt_f(__riscv_vget_u8m1(y4, 3), vl);
	vfloat16m2_t vy, dr, dg, db, v;
#else
	vfloat32m4_t vr = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(v4, 0), vl), vl);
	vfloat32m4_t vg = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(v4, 1), vl), vl);
	vfloat32m4_t vb = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(v4, 2), vl), vl);
	vfloat32m4_t va = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(v4, 3), vl), vl);
	vfloat32m4_t yr = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(y4, 0), vl), vl);
	vfloat32m4_t yg = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(y4, 1), vl), vl);
	vfloat32m4_t yb = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(y4, 2), vl), vl);
	vfloat32m4_t ya = __riscv_vfwcvt_f(__riscv_vzext_vf2(__riscv_vget_u8m1(y4, 3), vl), vl);
	vfloat32m4_t vy, dr, dg, db, v;
#endif

	/* ((v-255)*(va/255)+255) - ((y-255)*(ya/255)+255)
		* = (v-255)*(va/255)+255  -  (y-255)*(ya/255)-255
		* = (v-255)*(va/255)      -  (y-255)*(ya/255)
		* = (v-255)*(va/255)      -  (y-255)*(ya/255)
		* = (v*va/255-255*va/255) -  (y*ya/255-255*ya/255)
		* = (v*va/255-    va    ) -  (y*ya/255-    ya    )
		* =  v*va/255-    va      -   y*ya/255+    ya
		* =  v*va/255-  y*ya/255  -     va    +    ya
		* =  v*va/255- (y*ya/255  +    (va    -    ya)) */
	vy = __riscv_vfsub(va, ya, vl);
	va = __riscv_vfmul(va, 1/255.0f, vl);
	ya = __riscv_vfmul(ya, 1/255.0f, vl);
	dr = __riscv_vfmsub(vr, va, __riscv_vfmadd(yr, ya, vy, vl), vl);
	dg = __riscv_vfmsub(vg, va, __riscv_vfmadd(yg, ya, vy, vl), vl);
	db = __riscv_vfmsub(vb, va, __riscv_vfmadd(yb, ya, vy, vl), vl);

	/* r*(r*Y1 + g*Y2 + b*Y3) + g*(g*Y4 + b*Y5) + b*b*Y6 (see top of file) */
	v = __riscv_vfmul(__riscv_vfmul(db, Y6, vl), db, vl);
	v = __riscv_vfmacc(v, dg, __riscv_vfmacc(__riscv_vfmul(dg, Y4, vl), Y5, db, vl), vl);
	v = __riscv_vfmacc(v, dr, __riscv_vfmacc(__riscv_vfmacc(__riscv_vfmul(dr, Y1, vl), Y2, dg, vl), Y3, db, vl), vl);
	return v;
}

static size_t
diff_rvv(uint32_t *src1, uint32_t *src2, size_t n, float *diff)
{
	size_t count = 0;
	for (size_t vl; n > 0; n -= vl, src1 += vl, src2 += vl, diff += vl) {
		vl = __riscv_vsetvl_e8m1(n);
		vuint8m1x4_t v4 = __riscv_vlseg4e8_v_u8m1x4((uint8_t*)src1, vl);
		vuint8m1x4_t y4 = __riscv_vlseg4e8_v_u8m1x4((uint8_t*)src2, vl);
#if __riscv_zvfh
		__riscv_vse32(diff, __riscv_vfwcvt_f(rvv_yiq_diff(v4, y4, vl), vl), vl);
#else
		__riscv_vse32(diff, rvv_yiq_diff(v4, y4, vl), vl);
#endif
	}
	return count;
}

#if __GNUC__
__attribute__((noinline))
#endif
size_t
odiff_rvv(uint32_t *src1, uint32_t *src2, size_t n, float max_delta, uint32_t *diff, uint32_t diffcol)
{
	size_t count = 0;
	for (size_t vl; n > 0; n -= vl, src1 += vl, src2 += vl) {

#if __riscv_xtheadvector
		/* segmented load are very slow xtheadvector hardware (C910v1) */
		vl = __riscv_vsetvl_e32m4(n);
		vuint32m4_t v = __riscv_vle32_v_u32m4(src1, vl);
		vuint32m4_t y = __riscv_vle32_v_u32m4(src2, vl);
		long idx = __riscv_vfirst(__riscv_vmsne(v, y, vl), vl);
		if (idx < 0) continue;

		vuint16m2_t v1 = __riscv_vnsrl(v, 0, vl), v2 = __riscv_vnsrl(v, 16, vl);
		vuint16m2_t y1 = __riscv_vnsrl(y, 0, vl), y2 = __riscv_vnsrl(y, 16, vl);
		vuint8m1_t v11 = __riscv_vnsrl(v1, 0, vl), v12 = __riscv_vnsrl(v1, 8, vl);
		vuint8m1_t v21 = __riscv_vnsrl(v2, 0, vl), v22 = __riscv_vnsrl(v2, 8, vl);
		vuint8m1_t y11 = __riscv_vnsrl(y1, 0, vl), y12 = __riscv_vnsrl(y1, 8, vl);
		vuint8m1_t y21 = __riscv_vnsrl(y2, 0, vl), y22 = __riscv_vnsrl(y2, 8, vl);
		vuint8m1x4_t v4 = __riscv_vcreate_v_u8m1x4(v11, v12, v21, v22);
		vuint8m1x4_t y4 = __riscv_vcreate_v_u8m1x4(y11, y12, y21, y22);
#else
		vl = __riscv_vsetvl_e32m2(n);
		vuint32m2_t va = __riscv_vle32_v_u32m2(src1, vl);
		vuint32m2_t vb = __riscv_vle32_v_u32m2(src2, vl);
		long idx = __riscv_vfirst(__riscv_vmsne(va, vb, vl), vl);
		if (idx < 0) continue;
		src1 += idx;
		src2 += idx;
		n -= idx;

		vl = __riscv_vsetvl_e8m1(n);
		vuint8m1x4_t v4 = __riscv_vlseg4e8_v_u8m1x4((uint8_t*)src1, vl);
		vuint8m1x4_t y4 = __riscv_vlseg4e8_v_u8m1x4((uint8_t*)src2, vl);
#endif
		vbool8_t m = __riscv_vmfgt(rvv_yiq_diff(v4, y4, vl), max_delta, vl);
		count += __riscv_vcpop(m, vl);

		if (diff) {
			__riscv_vse32(m, diff, __riscv_vmv_v_x_u32m4(diffcol, vl), vl);
			diff += vl;
		}
	}
	return count;
}

#if defined(TEST)
#include <math.h>
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

#define N (1024*1024)
#define RUNS 100

int
main(void)
{
	uint32_t *src1 = malloc(N*4), *src2 = malloc(N*4);
	float *diff1 = malloc(N*4), *diff2 = malloc(N*4);
	RandU64 prng = {123^(uintptr_t)src2, 456^(uintptr_t)&prng, 789};

	for (size_t i = 0; i < RUNS; ++i) {
		for (size_t j = 0; j < N; ++j) {
			uint64_t u = randu64(&prng);
			uint64_t b1 = (u >> 32) & 1;
			uint64_t s1 = (u >> 33) & 31;
			uint64_t b2 = (u >> 37) & 1;
			uint64_t s2 = (u >> 38) & 31;
			src1[j] = u;
			if (j & 1) src2[j] = u >> 32;
			else       src2[j] = u ^ (b1<<s1) ^ (b2<<s2);
		}
		diff_fp64(src1, src2, N, diff1);
		diff_rvv( src1, src2, N, diff2);

		float threshold = (i+1)*1.0/(RUNS+1);
		float max_delta = 35215. * threshold*threshold;

		float max = 0;
		size_t count1 = 0, count2 = 0, wrong = 0;
		for (size_t j = 0; j < N; ++j) {
			float diff = fabsf(diff1[j] - diff2[j]);
			max = max > diff ? max : diff;
			int gt1 = diff1[j] > max_delta;
			int gt2 = diff2[j] > max_delta;
			count1 += gt1;
			count2 += gt1;
			wrong += gt1 != gt2;
		}
		printf("threshold: %f max_delta: %f max_diff: %f count: %zu/%zu wrong: %zu\n", threshold, max_delta, max, count1, count2, wrong);
	}

	free(src1); free(src2);
	free(diff1); free(diff2);
}
#elif defined(BENCH)

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1024*1024*32)

int
main(void)
{
	uint32_t *src1 = malloc(N*4), *src2 = malloc(N*4);
	RandU64 prng = {123^(uintptr_t)src2, 456^(uintptr_t)&prng, 789};

	float max_delta = 35215*0.1*0.1;

	for (size_t i = 0; i < 20; ++i) {
		for (size_t j = 0; j < N; ++j) {
			uint64_t u = randu64(&prng);
			src1[j] = u;
			src2[j] = u >> 32;
		}
		clock_t beg;

		beg = clock();
		size_t cnt1 = odiff_rvv(src1, src2, N, max_delta, (void*)0, 0);
		printf("rvv: %f secs ", (clock()-beg)*1.0/CLOCKS_PER_SEC);
		beg = clock();
		size_t cnt2 = odiff_fp32(src1, src2, N, max_delta, (void*)0, 0);
		printf("scalar: %f secs, got %zu %s %zu\n", (clock()-beg)*1.0/CLOCKS_PER_SEC, cnt1, cnt1==cnt2 ? "==" : "!=", cnt2);
	}

	free(src1); free(src2);
}

/*
 * sg2042 (XuanTie C910v1) with N 1024*1024*32:
 *     rvv:    0.151 secs
 *     scalar: 0.890 secs (5.9x speedup)
 *
 * bpi-f3 (SpacemiT X60) with N 1024*1024*32:
 *     rvv:    0.160 secs
 *     scalar: 3.058 secs (5.9x speedup)
 */
#endif
