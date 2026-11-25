#include "opus.h"
#include "arch.h"
#include "lane_stack.h"
#include <stdint.h>

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

void decoder_size(int *size) {
    *size = ((opus_decoder_get_size(2) + 15) & ~15) + GLOBAL_STACK_SIZE;
}

void decode_streams(
    int count, const uint8_t *data, const int64_t *packet_offsets,
    const int *first_packet, const int *channels, const int *gain,
    const int64_t *pcm_offsets, int16_t *pcm, uint8_t *states,
    int stride, int *status
) {
    int i = stream_index();
    if (i >= count) return;
    OpusDecoder *decoder = (OpusDecoder *)(states + (int64_t)i * stride);
    global_stack = (char *)decoder + stride - GLOBAL_STACK_SIZE;
    int result = opus_decoder_init(decoder, 48000, channels[i]);
    if (result == OPUS_OK)
        result = opus_decoder_ctl(decoder, OPUS_SET_GAIN(gain[i]));
    int64_t position = pcm_offsets[i];
    for (int p = first_packet[i]; result == OPUS_OK && p < first_packet[i + 1]; p++) {
        int64_t remaining = (pcm_offsets[i + 1] - position) / channels[i];
        int capacity = remaining > 5760 ? 5760 : (int)remaining;
        int length = (int)(packet_offsets[p + 1] - packet_offsets[p]);
        int frames = opus_decode(decoder, data + packet_offsets[p], length,
                                 pcm + position, capacity, 0);
        if (frames < 0)
            result = frames;
        else
            position += frames * channels[i];
    }
    status[i] = result ? result : (position == pcm_offsets[i + 1] ? OPUS_OK : OPUS_INVALID_PACKET);
}
