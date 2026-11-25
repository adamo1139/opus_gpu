#pragma once

/* One cursor per lane in block-local shared memory. The backing storage for
 * the actual samples is part of each stream's decoder workspace. */
#ifdef __NVPTX__
extern char * __attribute__((address_space(3))) lane_cursors[32];
#define global_stack lane_cursors[__nvvm_read_ptx_sreg_tid_x()]
#else
extern _Thread_local char *global_stack;
#endif

#ifdef OPUS_KERNEL_ENTRY
/* Compiling this file as ordinary C also exercises the same batch kernel on CPU. */
#ifdef __NVPTX__
char * __attribute__((address_space(3))) lane_cursors[32];
static inline int stream_index(void) {
    return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x()
        + __nvvm_read_ptx_sreg_tid_x();
}
#else
_Thread_local char *global_stack;
static inline int stream_index(void) { return 0; }
#endif

#endif
