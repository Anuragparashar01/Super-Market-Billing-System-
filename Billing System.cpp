// cnt1s_stream_bytes_outputdat.c
// Diehard Count-the-1's test on a STREAM OF BYTES
// Reads exactly 256004 bytes from file: output.dat

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static inline int popcount8(uint8_t x) {
    int c = 0;
    for (int i = 0; i < 8; ++i) c += (x >> i) & 1u;
    return c;
}

// Return digit 0..4 corresponding to letters A..E
// A:0-2 ones, B:3, C:4, D:5, E:6-8
static inline uint8_t byte_to_letter(uint8_t b) {
    int w = popcount8(b);
    if (w <= 2) return 0;   // A
    if (w == 3) return 1;   // B
    if (w == 4) return 2;   // C
    if (w == 5) return 3;   // D
    return 4;               // E
}

static inline double Phi(double z) {
    // Standard normal CDF
    return 0.5 * (1.0 + erf(z / M_SQRT2));
}

// Expected count for a cell index idx representing an L-letter base-5 word
static double expected_count(int idx, int L, double N, const double prob[5]) {
    double E = N;
    for (int j = 0; j < L; ++j) {
        int digit = idx % 5;
        E *= prob[digit];
        idx /= 5;
    }
    return E;
}

// Encode 5 base-5 digits into [0..3124]
static inline int code5(uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e) {
    return (((((int)a)*5 + (int)b)*5 + (int)c)*5 + (int)d)*5 + (int)e;
}

int main(void) {
    const char *fn = "output.dat";

    // Classic size: 256004 bytes => N = 256000 overlapping 5-letter words
    const size_t n_bytes = 256004;
    const size_t N = n_bytes - 4;

    const double prob[5] = {
        37.0/256.0, 56.0/256.0, 70.0/256.0, 56.0/256.0, 37.0/256.0
    };

    const double mean = 2500.0;
    const double stdv = sqrt(5000.0);

    FILE *fp = fopen(fn, "rb");
    if (!fp) {
        perror("fopen output.dat");
        return 1;
    }

    uint8_t *buf = (uint8_t*)malloc(n_bytes);
    if (!buf) {
        fprintf(stderr, "Out of memory.\n");
        fclose(fp);
        return 1;
    }

    size_t got = fread(buf, 1, n_bytes, fp);
    fclose(fp);

    if (got < n_bytes) {
        fprintf(stderr, "File too short: need %zu bytes, got %zu.\n", n_bytes, got);
        free(buf);
        return 1;
    }

    // Convert bytes -> letters (digits 0..4)
    uint8_t *L = (uint8_t*)malloc(n_bytes);
    if (!L) {
        fprintf(stderr, "Out of memory.\n");
        free(buf);
        return 1;
    }

    for (size_t i = 0; i < n_bytes; ++i) {
        L[i] = byte_to_letter(buf[i]);
    }

    // Count 5-letter words and 4-letter suffix words
    uint32_t f5[3125] = {0};
    uint32_t f4[625]  = {0};

    for (size_t i = 0; i < N; ++i) {
        int w5 = code5(L[i], L[i+1], L[i+2], L[i+3], L[i+4]); // 0..3124
        ++f5[w5];

        int suffix4 = w5 % 625; // last 4 letters
        ++f4[suffix4];
    }

    // Compute Q4 and Q5
    double Q4 = 0.0, Q5 = 0.0;

    for (int i = 0; i < 625; ++i) {
        double E = expected_count(i, 4, (double)N, prob);
        double diff = (double)f4[i] - E;
        Q4 += (diff * diff) / E;
    }

    for (int i = 0; i < 3125; ++i) {
        double E = expected_count(i, 5, (double)N, prob);
        double diff = (double)f5[i] - E;
        Q5 += (diff * diff) / E;
    }

    double stat = Q5 - Q4;
    double z = (stat - mean) / stdv;
    double p = 1.0 - Phi(z);

    printf("COUNT-THE-1's TEST (stream of bytes)\n");
    printf("File: %s\n", fn);
    printf("Bytes read: %zu\n", n_bytes);
    printf("Overlapping 5-letter words N = %zu\n", N);
    printf("Degrees of freedom (5^4 - 5^3) = 2500 (Diehard Q5-Q4)\n\n");
    printf("Q4      = %.6f\n", Q4);
    printf("Q5      = %.6f\n", Q5);
    printf("Q5 - Q4 = %.6f\n", stat);
    printf("z-score = % .6f\n", z);
    printf("p-value = %.10f\n", p);

    free(L);
    free(buf);
    return 0;
}
