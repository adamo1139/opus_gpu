#pragma once

/* One cursor per lane in block-local shared memory. The backing storage for
 * the actual samples is part of each stream's decoder workspace. */
#ifdef __NVPTX__
extern char * __attribute__((address_space(3))) lane_cursors[32];
#define global_stack lane_cursors[__nvvm_read_ptx_sreg_tid_x()]
#else
extern _Thread_local char *global_stack;
#endif
