#pragma once
#include <stddef.h>

void *malloc(size_t);
void *realloc(void *, size_t);
void free(void *);
#define abs(x) __builtin_abs(x)
