#include <stddef.h>
#include <stdint.h>

/* Freestanding PTX has no C runtime. Overlap occurs in the CELT history buffer. */
void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = dst;
    const unsigned char *s = src;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
    return dst;
}
void *memset(void *dst, int value, size_t n) {
    unsigned char *d = dst;
    for (size_t i = 0; i < n; i++) d[i] = (unsigned char)value;
    return dst;
}
void *memmove(void *dst, const void *src, size_t n) {
    if ((uintptr_t)dst <= (uintptr_t)src) return memcpy(dst, src, n);
    unsigned char *d = dst;
    const unsigned char *s = src;
    while (n > 0) { n--; d[n] = s[n]; }
    return dst;
}
