/*
 * Fast bit-packing for AAC bitstream construction.
 *
 * Exposes a single function via cdecl ABI:
 *   int bitwriter_pack(codes, lengths, n, output, capacity)
 *
 * Packs `n` variable-length codewords (MSB-first) into a contiguous byte
 * buffer. Returns the total number of bits written. The caller is
 * responsible for ensuring `output` has enough room.
 *
 * Loaded via ctypes at import time; falls back to the pure-Python
 * BitWriter if compilation fails.
 */

#include <stdint.h>
#include <string.h>

/*
 * BitWriter state — mirrors the Python class but with zero interpreter
 * overhead per write_bits call.
 */
typedef struct {
    uint8_t *buf;
    int      capacity;
    int      byte_pos;
    uint64_t accum;
    int      nbits;
    int      total_bits;
} BitWriter;

static void bw_init(BitWriter *bw, uint8_t *buf, int capacity) {
    bw->buf       = buf;
    bw->capacity  = capacity;
    bw->byte_pos  = 0;
    bw->accum     = 0;
    bw->nbits     = 0;
    bw->total_bits = 0;
}

static inline void bw_write(BitWriter *bw, uint32_t value, int num_bits) {
    uint64_t mask = ((uint64_t)1 << num_bits) - 1;
    bw->accum = (bw->accum << num_bits) | (value & mask);
    bw->nbits += num_bits;
    bw->total_bits += num_bits;

    while (bw->nbits >= 8) {
        bw->nbits -= 8;
        uint8_t byte = (uint8_t)((bw->accum >> bw->nbits) & 0xFF);
        if (bw->byte_pos < bw->capacity) {
            bw->buf[bw->byte_pos++] = byte;
        }
        if (bw->nbits > 0)
            bw->accum &= ((uint64_t)1 << bw->nbits) - 1;
        else
            bw->accum = 0;
    }
}

static int bw_flush(BitWriter *bw) {
    /* Flush partial byte (MSB-aligned). Returns total bytes. */
    if (bw->nbits > 0) {
        uint8_t byte = (uint8_t)((bw->accum << (8 - bw->nbits)) & 0xFF);
        if (bw->byte_pos < bw->capacity)
            bw->buf[bw->byte_pos++] = byte;
    }
    return bw->byte_pos;
}

/* ------------------------------------------------------------------ */
/* Public API — called from Python via ctypes.                        */
/* ------------------------------------------------------------------ */

/*
 * Pack an array of (code, length) pairs into a byte buffer.
 *
 * Parameters:
 *   codes    – uint32 array of codewords
 *   lengths  – uint8 array of bit lengths per codeword
 *   n        – number of codewords
 *   output   – pre-allocated output byte buffer
 *   capacity – size of output buffer in bytes
 *
 * Returns the total number of BITS written.
 */
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int bitwriter_pack(
    const uint32_t *codes,
    const uint8_t  *lengths,
    int             n,
    uint8_t        *output,
    int             capacity
) {
    BitWriter bw;
    bw_init(&bw, output, capacity);
    for (int i = 0; i < n; i++) {
        bw_write(&bw, codes[i], lengths[i]);
    }
    bw_flush(&bw);
    return bw.total_bits;
}

/*
 * Full ICS-level bitstream assembly: writes ADTS-payload bits for one
 * channel element given pre-computed arrays of (code, length) pairs for
 * each ICS section (header, section_data, scalefactor_data, spectral_data).
 *
 * This avoids round-tripping back to Python between sections.
 *
 * Parameters:
 *   header_codes / header_lengths / header_n       – ICS header bits
 *   section_codes / section_lengths / section_n     – section_data bits
 *   sf_codes / sf_lengths / sf_n                    – scalefactor_data bits
 *   spectral_codes / spectral_lengths / spectral_n  – spectral_data bits
 *   output / capacity                               – output buffer
 *
 * Returns total BITS written.
 */
EXPORT int bitwriter_pack_ics(
    const uint32_t *header_codes,   const uint8_t *header_lengths,   int header_n,
    const uint32_t *section_codes,  const uint8_t *section_lengths,  int section_n,
    const uint32_t *sf_codes,       const uint8_t *sf_lengths,       int sf_n,
    const uint32_t *spectral_codes, const uint8_t *spectral_lengths, int spectral_n,
    uint8_t *output, int capacity
) {
    BitWriter bw;
    bw_init(&bw, output, capacity);
    for (int i = 0; i < header_n; i++)
        bw_write(&bw, header_codes[i], header_lengths[i]);
    for (int i = 0; i < section_n; i++)
        bw_write(&bw, section_codes[i], section_lengths[i]);
    for (int i = 0; i < sf_n; i++)
        bw_write(&bw, sf_codes[i], sf_lengths[i]);
    for (int i = 0; i < spectral_n; i++)
        bw_write(&bw, spectral_codes[i], spectral_lengths[i]);
    bw_flush(&bw);
    return bw.total_bits;
}
