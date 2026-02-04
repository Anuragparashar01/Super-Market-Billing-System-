32x32
    // diehard_rank32.cpp
//
// Binary Rank 32x32 test (original Diehard style).
// INPUT  : output.dat  (TEXT file containing HEX BYTES, e.g. "0A FF 1c ...")
// OUTPUT : counts + chi-square + p-value
//
// Data need (original Diehard): 40,000 matrices × 128 bytes = 5,120,000 bytes (as hex-byte tokens)
//
// Compile:
//   g++ -O3 -std=c++17 diehard_rank32.cpp -o diehard_rank32
//
// Run (reads output.dat by default):
//   ./diehard_rank32
//
// Or specify file:
//   ./diehard_rank32 output.dat
//
// If your 32-bit word byte-order is different, flip WORD_LITTLE_ENDIAN.

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

static constexpr bool WORD_LITTLE_ENDIAN = true; // set false if your word order is big-endian

// Diehard theoretical probabilities (32x32 GF(2) rank)
static constexpr double P32   = 0.288788095;
static constexpr double P31   = 0.577576190;
static constexpr double PLE30 = 0.133635715;

// Read next hex byte token from stream (robust to separators, optional 0x prefix).
static bool read_hex_byte(istream& in, uint8_t& outByte) {
    string tok;
    while (in >> tok) {
        // strip trailing separators like "," ";" ":" etc.
        while (!tok.empty() && (tok.back() == ',' || tok.back() == ';' || tok.back() == ':'))
            tok.pop_back();
        if (tok.empty()) continue;

        // optional leading 0x
        if (tok.size() >= 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X')) {
            tok = tok.substr(2);
        }
        if (tok.empty()) continue;

        char* endp = nullptr;
        long v = strtol(tok.c_str(), &endp, 16);
        if (endp == tok.c_str() || *endp != '\0' || v < 0 || v > 255) {
            // not a clean hex-byte token -> skip
            continue;
        }
        outByte = static_cast<uint8_t>(v);
        return true;
    }
    return false; // EOF
}

static uint32_t bytes_to_u32(const uint8_t b[4]) {
    if (WORD_LITTLE_ENDIAN) {
        return (uint32_t)b[0]
             | ((uint32_t)b[1] << 8)
             | ((uint32_t)b[2] << 16)
             | ((uint32_t)b[3] << 24);
    } else {
        return ((uint32_t)b[0] << 24)
             | ((uint32_t)b[1] << 16)
             | ((uint32_t)b[2] << 8)
             | (uint32_t)b[3];
    }
}

// Rank of 32x32 binary matrix over GF(2), rows as uint32.
static int gf2_rank_32(uint32_t rows[32]) {
    int rank = 0;

    // pivot columns from MSB->LSB (31..0)
    for (int col = 31; col >= 0 && rank < 32; --col) {
        uint32_t mask = (1u << col);

        // find pivot row at/below 'rank'
        int pivot = -1;
        for (int r = rank; r < 32; ++r) {
            if (rows[r] & mask) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        // swap pivot into position
        if (pivot != rank) {
            uint32_t tmp = rows[pivot];
            rows[pivot] = rows[rank];
            rows[rank] = tmp;
        }

        // eliminate from all other rows
        for (int r = 0; r < 32; ++r) {
            if (r != rank && (rows[r] & mask)) {
                rows[r] ^= rows[rank];
            }
        }

        ++rank;
    }
    return rank;
}

int main(int argc, char** argv) {
    // Default file name as requested
    string filename = "output.dat";
    if (argc >= 2) filename = argv[1];

    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: cannot open file '" << filename << "'\n";
        return 1;
    }

    // Original Diehard uses 40,000 matrices for rank32 test
    const int M_target = 40000;

    long long N32 = 0, N31 = 0, NLE30 = 0;
    int matrices_done = 0;

    for (; matrices_done < M_target; ++matrices_done) {
        uint32_t rows[32];

        // build one 32x32 matrix from 128 bytes (32 words)
        for (int r = 0; r < 32; ++r) {
            uint8_t b[4];
            for (int k = 0; k < 4; ++k) {
                if (!read_hex_byte(fin, b[k])) {
                    cerr << "EOF: not enough hex bytes to complete matrix " << (matrices_done + 1) << "\n";
                    goto done;
                }
            }
            rows[r] = bytes_to_u32(b);
        }

        int rank = gf2_rank_32(rows);
        if (rank == 32) ++N32;
        else if (rank == 31) ++N31;
        else ++NLE30;
    }

done:
    long long M = N32 + N31 + NLE30;
    if (M == 0) {
        cerr << "No matrices processed.\n";
        return 1;
    }

    // Expected counts
    double E32   = (double)M * P32;
    double E31   = (double)M * P31;
    double ELE30 = (double)M * PLE30;

    // Chi-square (df=2)
    double chi2 =
        ((N32   - E32)   * (N32   - E32))   / E32 +
        ((N31   - E31)   * (N31   - E31))   / E31 +
        ((NLE30 - ELE30) * (NLE30 - ELE30)) / ELE30;

    // For df=2, p-value = exp(-chi2/2)
    double p_value = exp(-0.5 * chi2);

    cout << "Binary Rank 32x32 (Diehard-style)\n";
    cout << "Input file: " << filename << "\n";
    cout << "Matrices processed M = " << M << "\n\n";

    cout << "Observed counts:\n";
    cout << "  N(rank=32)  = " << N32 << "\n";
    cout << "  N(rank=31)  = " << N31 << "\n";
    cout << "  N(rank<=30) = " << NLE30 << "\n\n";

    cout << "Expected counts (Diehard):\n";
    cout << "  E32   = " << E32 << "\n";
    cout << "  E31   = " << E31 << "\n";
    cout << "  E<=30 = " << ELE30 << "\n\n";

    cout << "chi^2 (df=2) = " << chi2 << "\n";
    cout << "p-value      = " << p_value << "\n";

    if (p_value < 1e-6 || p_value > 1.0 - 1e-6) {
        cout << "WARNING: Extreme p-value (very close to 0 or 1) is suspicious.\n";
    }

    // Practical hint about endianness
    cout << "\nNote: WORD_LITTLE_ENDIAN = " << (WORD_LITTLE_ENDIAN ? "true" : "false") << "\n";
    cout << "If results look unreasonable, flip WORD_LITTLE_ENDIAN and rerun.\n";

    return 0;
}




31x31
    // diehard_rank31.cpp
//
// Binary Rank 31x31 test (original Diehard-style binning: 31, 30, 29, <=28).
// INPUT  : output.dat  (TEXT file containing HEX BYTES, e.g. "0A FF 1c ...")
// OUTPUT : counts + chi-square + p-value
//
// Matrix construction (Diehard style):
// - Build 31 rows. Each row is taken from ONE 32-bit word but ONLY 31 bits are used.
// - We mask the top bit off: row = word & 0x7FFFFFFF.
//
// Recommended M in classic Diehard is 40,000 matrices.
//
// Compile:
//   g++ -O3 -std=c++17 diehard_rank31.cpp -o diehard_rank31
//
// Run (reads output.dat by default):
//   ./diehard_rank31
// Or:
//   ./diehard_rank31 output.dat
//
// NOTE: If your 32-bit word byte-order is different, flip WORD_LITTLE_ENDIAN.

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

static constexpr bool WORD_LITTLE_ENDIAN = true; // set false if your word order is big-endian

// Diehard expected probabilities for 31x31 bins: rank 31, 30, 29, <=28
// These correspond to expected counts for M=40000 shown in classic Diehard writeups.
// E31=11551.5, E30=23103.0, E29=5134.0, E<=28=211.4
static constexpr double P31   = 11551.5 / 40000.0;
static constexpr double P30   = 23103.0 / 40000.0;
static constexpr double P29   =  5134.0 / 40000.0;
static constexpr double PLE28 =   211.4 / 40000.0;

// Read next hex byte token from stream (robust to separators, optional 0x prefix).
static bool read_hex_byte(istream& in, uint8_t& outByte) {
    string tok;
    while (in >> tok) {
        while (!tok.empty() && (tok.back() == ',' || tok.back() == ';' || tok.back() == ':'))
            tok.pop_back();
        if (tok.empty()) continue;

        if (tok.size() >= 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X')) {
            tok = tok.substr(2);
        }
        if (tok.empty()) continue;

        char* endp = nullptr;
        long v = strtol(tok.c_str(), &endp, 16);
        if (endp == tok.c_str() || *endp != '\0' || v < 0 || v > 255) {
            continue;
        }
        outByte = static_cast<uint8_t>(v);
        return true;
    }
    return false;
}

static uint32_t bytes_to_u32(const uint8_t b[4]) {
    if (WORD_LITTLE_ENDIAN) {
        return (uint32_t)b[0]
             | ((uint32_t)b[1] << 8)
             | ((uint32_t)b[2] << 16)
             | ((uint32_t)b[3] << 24);
    } else {
        return ((uint32_t)b[0] << 24)
             | ((uint32_t)b[1] << 16)
             | ((uint32_t)b[2] << 8)
             | (uint32_t)b[3];
    }
}

// Rank of 31x31 binary matrix over GF(2).
// rows[0..30] each has only 31 active bits (bit 30..0). Bit 31 is 0.
// NOTE: modifies rows in-place.
static int gf2_rank_31(uint32_t rows[31]) {
    int rank = 0;

    // pivot from bit 30 down to bit 0 (31 columns)
    for (int col = 30; col >= 0 && rank < 31; --col) {
        uint32_t mask = (1u << col);

        int pivot = -1;
        for (int r = rank; r < 31; ++r) {
            if (rows[r] & mask) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        if (pivot != rank) {
            uint32_t tmp = rows[pivot];
            rows[pivot] = rows[rank];
            rows[rank] = tmp;
        }

        for (int r = 0; r < 31; ++r) {
            if (r != rank && (rows[r] & mask)) {
                rows[r] ^= rows[rank];
            }
        }

        ++rank;
    }

    return rank;
}

int main(int argc, char** argv) {
    string filename = "output.dat";
    if (argc >= 2) filename = argv[1];

    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: cannot open file '" << filename << "'\n";
        return 1;
    }

    const int M_target = 40000;

    long long N31 = 0, N30 = 0, N29 = 0, NLE28 = 0;
    int matrices_done = 0;

    for (; matrices_done < M_target; ++matrices_done) {
        uint32_t rows[31];

        // For each row: read one 32-bit word (4 bytes) then keep only 31 bits
        for (int r = 0; r < 31; ++r) {
            uint8_t b[4];
            for (int k = 0; k < 4; ++k) {
                if (!read_hex_byte(fin, b[k])) {
                    cerr << "EOF: not enough hex bytes to complete matrix " << (matrices_done + 1) << "\n";
                    goto done;
                }
            }
            uint32_t w = bytes_to_u32(b);
            rows[r] = (w & 0x7FFFFFFFu); // keep only 31 bits
        }

        int rank = gf2_rank_31(rows);
        if (rank == 31) ++N31;
        else if (rank == 30) ++N30;
        else if (rank == 29) ++N29;
        else ++NLE28;
    }

done:
    long long M = N31 + N30 + N29 + NLE28;
    if (M == 0) {
        cerr << "No matrices processed.\n";
        return 1;
    }

    // Expected counts
    double E31   = (double)M * P31;
    double E30   = (double)M * P30;
    double E29   = (double)M * P29;
    double ELE28 = (double)M * PLE28;

    // Chi-square (4 bins => df=3)
    double chi2 =
        ((N31   - E31)   * (N31   - E31))   / E31 +
        ((N30   - E30)   * (N30   - E30))   / E30 +
        ((N29   - E29)   * (N29   - E29))   / E29 +
        ((NLE28 - ELE28) * (NLE28 - ELE28)) / ELE28;

    // p-value for chi-square(df=3): use survival function via incomplete gamma.
    // We'll use a simple approximation by calling std::erfc for df=1/2 only is not enough.
    // Instead, we compute p-value using regularized gamma Q(k/2, chi2/2) with k=3.
    //
    // For df=3, Q(1.5, x) has a closed form:
    //   Q(3/2, x) = erfc(sqrt(x)) + (2/sqrt(pi)) * sqrt(x) * exp(-x)
    //
    double x = 0.5 * chi2;                // x = chi2/2
    double sx = sqrt(x);
    double p_value = erfc(sx) + (2.0 / sqrt(M_PI)) * sx * exp(-x);

    cout << "Binary Rank 31x31 (Diehard-style)\n";
    cout << "Input file: " << filename << "\n";
    cout << "Matrices processed M = " << M << "\n\n";

    cout << "Observed counts:\n";
    cout << "  N(rank=31)  = " << N31 << "\n";
    cout << "  N(rank=30)  = " << N30 << "\n";
    cout << "  N(rank=29)  = " << N29 << "\n";
    cout << "  N(rank<=28) = " << NLE28 << "\n\n";

    cout << "Expected counts (Diehard):\n";
    cout << "  E31   = " << E31 << "\n";
    cout << "  E30   = " << E30 << "\n";
    cout << "  E29   = " << E29 << "\n";
    cout << "  E<=28 = " << ELE28 << "\n\n";

    cout << "chi^2 (df=3) = " << chi2 << "\n";
    cout << "p-value      = " << p_value << "\n";

    if (p_value < 1e-6 || p_value > 1.0 - 1e-6) {
        cout << "WARNING: Extreme p-value (very close to 0 or 1) is suspicious.\n";
    }

    cout << "\nNote: WORD_LITTLE_ENDIAN = " << (WORD_LITTLE_ENDIAN ? "true" : "false") << "\n";
    cout << "If results look unreasonable, flip WORD_LITTLE_ENDIAN and rerun.\n";

    return 0;
}
32x32
    // diehard_rank32.cpp
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

static constexpr bool WORD_LITTLE_ENDIAN = true;

// Diehard probabilities for 32x32: bins 32, 31, <=30
static constexpr double P32   = 0.288788095;
static constexpr double P31   = 0.577576190;
static constexpr double PLE30 = 0.133635715;

static bool read_hex_byte(istream& in, uint8_t& outByte) {
    string tok;
    while (in >> tok) {
        while (!tok.empty() && (tok.back()==',' || tok.back()==';' || tok.back()==':'))
            tok.pop_back();
        if (tok.empty()) continue;
        if (tok.size() >= 2 && tok[0]=='0' && (tok[1]=='x' || tok[1]=='X'))
            tok = tok.substr(2);
        if (tok.empty()) continue;

        char* endp = nullptr;
        long v = strtol(tok.c_str(), &endp, 16);
        if (endp == tok.c_str() || *endp != '\0' || v < 0 || v > 255) continue;
        outByte = (uint8_t)v;
        return true;
    }
    return false;
}

static uint32_t bytes_to_u32(const uint8_t b[4]) {
    if (WORD_LITTLE_ENDIAN) {
        return (uint32_t)b[0] | ((uint32_t)b[1]<<8) | ((uint32_t)b[2]<<16) | ((uint32_t)b[3]<<24);
    } else {
        return ((uint32_t)b[0]<<24) | ((uint32_t)b[1]<<16) | ((uint32_t)b[2]<<8) | (uint32_t)b[3];
    }
}

static int rank_gf2_32(uint32_t rows[32]) {
    int rank = 0;
    for (int col = 31; col >= 0 && rank < 32; --col) {
        uint32_t mask = (1u << col);

        int pivot = -1;
        for (int r = rank; r < 32; ++r) {
            if (rows[r] & mask) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        if (pivot != rank) swap(rows[pivot], rows[rank]);

        for (int r = 0; r < 32; ++r) {
            if (r != rank && (rows[r] & mask)) rows[r] ^= rows[rank];
        }
        ++rank;
    }
    return rank;
}

int main(int argc, char** argv) {
    string filename = "output.dat";
    if (argc >= 2) filename = argv[1];

    ifstream fin(filename);
    if (!fin) { cerr << "Cannot open " << filename << "\n"; return 1; }

    const int M_target = 40000;
    long long N32=0, N31=0, NLE30=0;

    for (int m=0; m<M_target; ++m) {
        uint32_t rows[32];
        for (int r=0; r<32; ++r) {
            uint8_t b[4];
            for (int k=0; k<4; ++k) {
                if (!read_hex_byte(fin, b[k])) {
                    cerr << "EOF before completing matrix " << (m+1) << "\n";
                    goto done;
                }
            }
            rows[r] = bytes_to_u32(b);
        }

        int rk = rank_gf2_32(rows);
        if (rk==32) ++N32;
        else if (rk==31) ++N31;
        else ++NLE30;
    }

done:
    long long M = N32 + N31 + NLE30;
    if (M==0) { cerr << "No matrices processed.\n"; return 1; }

    double E32 = M*P32, E31 = M*P31, ELE30 = M*PLE30;
    double chi2 =
        ((N32-E32)*(N32-E32))/E32 +
        ((N31-E31)*(N31-E31))/E31 +
        ((NLE30-ELE30)*(NLE30-ELE30))/ELE30;

    double p = exp(-0.5*chi2); // df=2

    cout << "Binary Rank 32x32 (Diehard)\n";
    cout << "M=" << M << "\n";
    cout << "N32=" << N32 << "  N31=" << N31 << "  N<=30=" << NLE30 << "\n";
    cout << "E32=" << E32 << "  E31=" << E31 << "  E<=30=" << ELE30 << "\n";
    cout << "chi2(df=2)=" << chi2 << "  p=" << p << "\n";
    return 0;
}
31x31
    // diehard_rank31.cpp
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

static constexpr bool WORD_LITTLE_ENDIAN = true;

// Expected counts (classic Diehard) for M=40000:
static constexpr double P31   = 11551.5 / 40000.0;
static constexpr double P30   = 23103.0 / 40000.0;
static constexpr double P29   =  5134.0 / 40000.0;
static constexpr double PLE28 =   211.4 / 40000.0;

static bool read_hex_byte(istream& in, uint8_t& outByte) {
    string tok;
    while (in >> tok) {
        while (!tok.empty() && (tok.back()==',' || tok.back()==';' || tok.back()==':'))
            tok.pop_back();
        if (tok.empty()) continue;
        if (tok.size() >= 2 && tok[0]=='0' && (tok[1]=='x' || tok[1]=='X'))
            tok = tok.substr(2);
        if (tok.empty()) continue;

        char* endp = nullptr;
        long v = strtol(tok.c_str(), &endp, 16);
        if (endp == tok.c_str() || *endp != '\0' || v < 0 || v > 255) continue;
        outByte = (uint8_t)v;
        return true;
    }
    return false;
}

static uint32_t bytes_to_u32(const uint8_t b[4]) {
    if (WORD_LITTLE_ENDIAN) {
        return (uint32_t)b[0] | ((uint32_t)b[1]<<8) | ((uint32_t)b[2]<<16) | ((uint32_t)b[3]<<24);
    } else {
        return ((uint32_t)b[0]<<24) | ((uint32_t)b[1]<<16) | ((uint32_t)b[2]<<8) | (uint32_t)b[3];
    }
}

static int rank_gf2_31(uint32_t rows[31]) {
    int rank = 0;
    for (int col = 30; col >= 0 && rank < 31; --col) {
        uint32_t mask = (1u << col);

        int pivot = -1;
        for (int r = rank; r < 31; ++r) {
            if (rows[r] & mask) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        if (pivot != rank) swap(rows[pivot], rows[rank]);

        for (int r = 0; r < 31; ++r) {
            if (r != rank && (rows[r] & mask)) rows[r] ^= rows[rank];
        }
        ++rank;
    }
    return rank;
}

int main(int argc, char** argv) {
    string filename = "output.dat";
    if (argc >= 2) filename = argv[1];

    ifstream fin(filename);
    if (!fin) { cerr << "Cannot open " << filename << "\n"; return 1; }

    const int M_target = 40000;
    long long N31=0, N30=0, N29=0, NLE28=0;

    for (int m=0; m<M_target; ++m) {
        uint32_t rows[31];

        // Each row from one 32-bit word, keep 31 LSBs
        for (int r=0; r<31; ++r) {
            uint8_t b[4];
            for (int k=0; k<4; ++k) {
                if (!read_hex_byte(fin, b[k])) {
                    cerr << "EOF before completing matrix " << (m+1) << "\n";
                    goto done;
                }
            }
            uint32_t w = bytes_to_u32(b);
            rows[r] = (w & 0x7FFFFFFFu); // 31 LSBs
        }

        int rk = rank_gf2_31(rows);
        if (rk==31) ++N31;
        else if (rk==30) ++N30;
        else if (rk==29) ++N29;
        else ++NLE28;
    }

done:
    long long M = N31 + N30 + N29 + NLE28;
    if (M==0) { cerr << "No matrices processed.\n"; return 1; }

    double E31 = M*P31, E30 = M*P30, E29 = M*P29, ELE28 = M*PLE28;

    double chi2 =
        ((N31-E31)*(N31-E31))/E31 +
        ((N30-E30)*(N30-E30))/E30 +
        ((N29-E29)*(N29-E29))/E29 +
        ((NLE28-ELE28)*(NLE28-ELE28))/ELE28;

    // df=3 p-value (closed form): p = erfc(sqrt(x)) + (2/sqrt(pi))*sqrt(x)*exp(-x), x=chi2/2
    double x = 0.5*chi2;
    double sx = sqrt(x);
    double p = erfc(sx) + (2.0/sqrt(M_PI))*sx*exp(-x);

    cout << "Binary Rank 31x31 (Diehard)\n";
    cout << "M=" << M << "\n";
    cout << "N31=" << N31 << "  N30=" << N30 << "  N29=" << N29 << "  N<=28=" << NLE28 << "\n";
    cout << "E31=" << E31 << "  E30=" << E30 << "  E29=" << E29 << "  E<=28=" << ELE28 << "\n";
    cout << "chi2(df=3)=" << chi2 << "  p=" << p << "\n";
    return 0;
}

snippet code
// GF(2) rank of a 32x32 binary matrix.
// Represent the matrix as 32 rows, each row packed into a uint32_t.
// Bit operations are over GF(2): elimination uses XOR.
//
// rank = number of pivots found (0..32).
//
// Usage:
//   uint32_t rows[32] = {...};   // fill with your 32 words
//   int r = rank_gf2_32(rows);   // NOTE: modifies rows in-place
//
#include <cstdint>
using namespace std;

int rank_gf2_32(uint32_t rows[32]) {
    int rank = 0;

    // Pivot from MSB (bit 31) down to LSB (bit 0)
    for (int col = 31; col >= 0 && rank < 32; --col) {
        uint32_t mask = (1u << col);

        // 1) Find pivot row at/below current rank with a 1 in this column
        int pivot = -1;
        for (int r = rank; r < 32; ++r) {
            if (rows[r] & mask) { pivot = r; break; }
        }
        if (pivot == -1) continue; // no pivot in this column

        // 2) Swap pivot row into position "rank"
        if (pivot != rank) {
            uint32_t tmp = rows[pivot];
            rows[pivot] = rows[rank];
            rows[rank]  = tmp;
        }

        // 3) Eliminate this column from all other rows
        for (int r = 0; r < 32; ++r) {
            if (r != rank && (rows[r] & mask)) {
                rows[r] ^= rows[rank];
            }
        }

        // 4) One pivot found
        ++rank;
    }

    return rank;
}




6x8
    // diehard_rank6x8.cpp
//
// Diehard Binary Rank 6x8 test (original-style bucketing):
//   Bucket A: rank = 6
//   Bucket B: rank = 5
//   Bucket C: rank <= 4
//
// INPUT  : output.dat  (TEXT file containing HEX BYTES, e.g. "0A FF 1c ...")
// DEFAULT: M = 100000 matrices  (classic Diehard choice)
// DATA   : 6 bytes per matrix  => total bytes needed = 6*M
//
// Expected probabilities (random bits):
//   P(rank=6)  ≈ 0.773118
//   P(rank=5)  ≈ 0.217439
//   P(rank<=4) ≈ 0.009443
//
// Chi-square with 3 buckets => df = 2
// For df=2, p-value = exp(-chi2/2)
//
// Compile:
//   g++ -O3 -std=c++17 diehard_rank6x8.cpp -o diehard_rank6x8
//
// Run (reads output.dat, M=100000):
//   ./diehard_rank6x8
//
// Run with custom file and/or M:
//   ./diehard_rank6x8 output.dat 200000

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// Expected probabilities for rank buckets (6x8 over GF(2))
static constexpr double P6   = 0.773118;
static constexpr double P5   = 0.217439;
static constexpr double PLE4 = 0.009443;

// Read next hex byte token from stream (robust to separators, optional 0x prefix).
static bool read_hex_byte(istream& in, uint8_t& outByte) {
    string tok;
    while (in >> tok) {
        while (!tok.empty() && (tok.back() == ',' || tok.back() == ';' || tok.back() == ':'))
            tok.pop_back();
        if (tok.empty()) continue;

        if (tok.size() >= 2 && tok[0] == '0' && (tok[1] == 'x' || tok[1] == 'X'))
            tok = tok.substr(2);
        if (tok.empty()) continue;

        char* endp = nullptr;
        long v = strtol(tok.c_str(), &endp, 16);
        if (endp == tok.c_str() || *endp != '\0' || v < 0 || v > 255) continue;

        outByte = static_cast<uint8_t>(v);
        return true;
    }
    return false; // EOF
}

// Compute rank of a 6x8 binary matrix over GF(2).
// Rows are 8-bit values, each bit is a column.
// NOTE: modifies rows[] in-place.
static int rank_gf2_6x8(uint8_t rows[6]) {
    int rank = 0;

    // Pivot columns from MSB->LSB (bit 7..0). LSB->MSB also works; rank is same.
    for (int col = 7; col >= 0 && rank < 6; --col) {
        uint8_t mask = static_cast<uint8_t>(1u << col);

        // Find pivot row with a 1 in this column, among rows[rank..5]
        int pivot = -1;
        for (int r = rank; r < 6; ++r) {
            if (rows[r] & mask) { pivot = r; break; }
        }
        if (pivot < 0) continue;

        // Swap pivot into position
        if (pivot != rank) {
            uint8_t tmp = rows[pivot];
            rows[pivot] = rows[rank];
            rows[rank]  = tmp;
        }

        // Eliminate this column from all other rows
        for (int r = 0; r < 6; ++r) {
            if (r != rank && (rows[r] & mask)) {
                rows[r] ^= rows[rank];
            }
        }

        ++rank;
    }

    return rank; // 0..6
}

int main(int argc, char** argv) {
    string filename = "output.dat";
    long long M_target = 100000; // classic Diehard choice for 6x8 rank

    if (argc >= 2) filename = argv[1];
    if (argc >= 3) M_target = atoll(argv[2]);

    if (M_target <= 0) {
        cerr << "Error: M must be positive.\n";
        return 1;
    }

    ifstream fin(filename);
    if (!fin) {
        cerr << "Error: cannot open file '" << filename << "'\n";
        return 1;
    }

    long long N6 = 0, N5 = 0, NLE4 = 0;
    long long done = 0;

    for (; done < M_target; ++done) {
        uint8_t rows[6];

        // One 6x8 matrix = 6 bytes = 6 rows of 8 bits
        for (int r = 0; r < 6; ++r) {
            if (!read_hex_byte(fin, rows[r])) {
                cerr << "EOF: not enough hex bytes to complete matrix " << (done + 1) << "\n";
                goto finished;
            }
        }

        int rk = rank_gf2_6x8(rows);
        if (rk == 6) ++N6;
        else if (rk == 5) ++N5;
        else ++NLE4;
    }

finished:
    long long M = N6 + N5 + NLE4;
    if (M == 0) {
        cerr << "No matrices processed.\n";
        return 1;
    }

    // Expected counts
    double E6   = (double)M * P6;
    double E5   = (double)M * P5;
    double ELE4 = (double)M * PLE4;

    // Chi-square (df=2)
    double chi2 =
        ((N6   - E6)   * (N6   - E6))   / E6 +
        ((N5   - E5)   * (N5   - E5))   / E5 +
        ((NLE4 - ELE4) * (NLE4 - ELE4)) / ELE4;

    // df=2 => p = exp(-chi2/2)
    double p = exp(-0.5 * chi2);

    cout << "Binary Rank 6x8 (Diehard-style)\n";
    cout << "Input file: " << filename << "\n";
    cout << "Matrices processed M = " << M << "\n";
    cout << "Data consumed (bytes) ~= " << (6LL * M) << "\n\n";

    cout << "Observed counts:\n";
    cout << "  N(rank=6)  = " << N6 << "\n";
    cout << "  N(rank=5)  = " << N5 << "\n";
    cout << "  N(rank<=4) = " << NLE4 << "\n\n";

    cout << "Expected counts:\n";
    cout << "  E6   = " << E6 << "\n";
    cout << "  E5   = " << E5 << "\n";
    cout << "  E<=4 = " << ELE4 << "\n\n";

    cout << "chi^2 (df=2) = " << chi2 << "\n";
    cout << "p-value      = " << p << "\n";

    if (p < 1e-6 || p > 1.0 - 1e-6) {
        cout << "WARNING: Extreme p-value (very close to 0 or 1) is suspicious.\n";
    }

    return 0;
}

    
