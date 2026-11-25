#define OPUS_KERNEL_ENTRY
#include "opus.h"
#include "arch.h"
#include "lane_stack.h"
#include <stdint.h>

void encoder_size(int *size) {
    *size = ((opus_encoder_get_size(2) + 15) & ~15) + GLOBAL_STACK_SIZE;
}

/* One stateful encoder per file; PCM is padded to complete 20 ms frames. */
void encode_streams(int count, const int16_t *pcm, const int64_t *pcm_offsets,
                    const int *first_packet, const int *channels, int bitrate,
                    int complexity, int capacity, unsigned char *packets, int *sizes,
                    unsigned char *states, int stride, int *status) {
    int i = stream_index();
    if (i >= count) return;
    OpusEncoder *encoder = (OpusEncoder *)(states + (int64_t)i * stride);
    global_stack = (char *)encoder + stride - GLOBAL_STACK_SIZE;
    int result = opus_encoder_init(encoder, 48000, channels[i], OPUS_APPLICATION_AUDIO);
    if (result == OPUS_OK) result = opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
    if (result == OPUS_OK) result = opus_encoder_ctl(encoder, OPUS_SET_COMPLEXITY(complexity));
    if (result == OPUS_OK) result = opus_encoder_ctl(encoder, OPUS_SET_VBR(0));
    int delay = 0;
    if (result == OPUS_OK) result = opus_encoder_ctl(encoder, OPUS_GET_LOOKAHEAD(&delay));
    if (result == OPUS_OK && delay != 312) result = OPUS_INTERNAL_ERROR;
    const int16_t *samples = pcm + pcm_offsets[i];
    for (int p = first_packet[i]; result == OPUS_OK && p < first_packet[i+1]; p++) {
        int size = opus_encode(encoder, samples, 960, packets + (int64_t)p * capacity, capacity);
        if (size < 0) result = size;
        else sizes[p] = size;
        samples += 960 * channels[i];
    }
    status[i] = result;
}
