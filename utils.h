#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#if !defined(NOLOG) && !defined(FUZZ)

#include <string.h>
#include <stdio.h>

#ifndef LOG_INDENT
#define LOG_INDENT 8
#endif

#define logf printf
#define log puts

#define LOG_NAME(name) do { \
		static char str[LOG_INDENT < sizeof name ? sizeof name : LOG_INDENT]; \
		memcpy(str, name, sizeof name); \
		memset(str + sizeof name, ' ', sizeof str - sizeof name); \
		fwrite(str, 1, sizeof str, stdout); \
	} while (0)

#define LOGC(name, v, vl) do { \
	LOG_NAME(name); \
	for (size_t i = vl; i--; ) { \
		char c = __riscv_vmv_x(__riscv_vslidedown(v, i, vl)); \
		if (c == '\0') c = '@';\
		if (c == '\n') c = '|';\
		printf("%c", c); \
	} \
	puts(""); \
} while (0)

#define LOGV(name, fmt, v, vl) do { \
	LOG_NAME(name); \
	for (size_t i = vl; i--; ) \
		printf(fmt, __riscv_vmv_x(__riscv_vslidedown(v, i, vl))); \
	puts(""); \
} while (0)

#define LOGVf(name, v, vl) do { \
	LOG_NAME(name); \
	for (size_t i = vl; i--; ) \
		printf("%g ", __riscv_vfmv_f(__riscv_vslidedown(v, i, vl))); \
	puts(""); \
} while (0)

#define LOGM(name, M, m, vl) LOGMf(name, M, "%1x", m, vl)

#define LOGMf(name, M, fmt, m, vl) do { \
	vuint8m##M##_t v = __riscv_vmv_v_x_u8m##M(0, vl); \
	v = __riscv_vmerge(v, 1, m, vl); \
	LOGV(name, fmt, v, vl); \
} while (0)

#else
#define logf(...)
#define log(...)
#define LOGC(...)
#define LOGV(...)
#define LOGM(...)
#define LOGMf(...)
#endif

typedef struct { uint64_t x, y, z; } RandU64;

#define RANDU64_ROTL(x,n) (((x) << (n)) | ((x) >> (8*sizeof(x) - (n))))

/* RomuTrio, see https://romu-random.org/ */
static inline uint64_t
randu64(RandU64 *r)
{
	uint64_t xp = r->x, yp = r->y, zp = r->z;
	r->x = 15241094284759029579u * zp;
	r->y = RANDU64_ROTL(yp - xp, 12);
	r->z = RANDU64_ROTL(zp - yp, 44);
	return xp;
}

#endif
