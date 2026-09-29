// wallet.cu  —  versiune unificata: v2, v3, v4, v5
//   v2 = Philox + cursor + sub-ferestre
//   v3 = Feistel permutation (acoperire 100%, DEFAULT)
//   v4 = Philox + random pur din tot intervalul (cu replacement, la infinit)
//   v5 = blocuri + random + checkpoint (reluare dupa Ctrl+C)
//
// Optimizari inspirate din https://github.com/Vladimir855/Rotor-Cuda:
//   - Grid masiv configurabil (--blocks / --gpux N,M)
//   - Batch inversion configurabil (--invsize N)
//   - Refresh start points (--rkey N) pentru a evita degenerarea Philox
//   - Auto-grid agresiv pt v4/v5 (32 waves)
//   - Fix occupancy per kernel (feistel vs random)
//   - Flag --bench pentru auto-testare KPT
//
// Compile: nvcc -O3 -arch=sm_75 -std=c++14 wallet.cu -o wallet_cuda \
//          -lsecp256k1 -lcrypto -lpthread
// Run:     ./wallet_cuda [-v2|-v3|-v4|-v5] [opts] START:END TARGET [BATCH]
#include <cuda_runtime.h>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#define BLOCK_SIZE        256
#define DEFAULT_KPT       4
#define DEFAULT_BATCH     65536
#define MAX_MATCHES       32

#define COMB_WINDOWS      18
#define COMB_ENTRIES      15
#define FEISTEL_ROUNDS    6

#define DEFAULT_V5_BLOCK  1000000000ULL
#define DEFAULT_V5_SAMPLE 10000000ULL
#define DEFAULT_RKEY      100000000000ULL   // Rotor-Cuda: refresh la 1e11
#define CKPT_FILE         "wallet_ckpt.txt"
#define FOUND_FILE        "found_keys.txt"  // fisier output pentru chei gasite

// =============================================================================
// Device helpers
// =============================================================================
__device__ __forceinline__ unsigned long long splitmix64_impl(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ unsigned int philox_mulhi_lo(
    unsigned int a, unsigned int b, unsigned int &lo)
{
    unsigned long long p = (unsigned long long)a * (unsigned long long)b;
    lo = (unsigned int)p;
    return (unsigned int)(p >> 32);
}
__device__ __forceinline__ void philox4x32_10(
    const unsigned int key[2], const unsigned int ctr[4], unsigned int out[4])
{
    const unsigned int M0 = 0xD2511F53u;
    const unsigned int M1 = 0xCD9E8D57u;
    const unsigned int W0 = 0x9E3779B9u;
    const unsigned int W1 = 0xBB67AE85u;
    unsigned int c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3];
    unsigned int k0 = key[0], k1 = key[1];
    #pragma unroll 1
    for (int i = 0; i < 10; i++) {
        unsigned int lo0, lo1;
        unsigned int hi0 = philox_mulhi_lo(M0, c0, lo0);
        unsigned int hi1 = philox_mulhi_lo(M1, c2, lo1);
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 += W0;
        k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}
__device__ __forceinline__ void philox_init(
    unsigned int key[2], unsigned int ctr[4],
    unsigned long long seed, unsigned long long tid)
{
    unsigned long long s0 = splitmix64_impl(seed);
    unsigned long long s1 = splitmix64_impl(s0 ^ (tid + 0x9E3779B97F4A7C15ULL));
    key[0] = ((unsigned int)s0) ^ ((unsigned int)s1) ^ ((unsigned int)tid);
    key[1] = ((unsigned int)(s0 >> 32)) ^ ((unsigned int)(s1 >> 32))
           ^ ((unsigned int)(tid >> 32));
    ctr[0] = 0; ctr[1] = 0; ctr[2] = 0; ctr[3] = 0;
}
__device__ __forceinline__ void philox_block16(
    const unsigned int key[2], unsigned int ctr[4], unsigned int out[16])
{
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int tmp[4];
        philox4x32_10(key, ctr, tmp);
        out[4*i    ] = tmp[0];
        out[4*i + 1] = tmp[1];
        out[4*i + 2] = tmp[2];
        out[4*i + 3] = tmp[3];
        ctr[0]++;
        if (ctr[0] == 0) {
            ctr[1]++;
            if (ctr[1] == 0) {
                ctr[2]++;
                if (ctr[2] == 0) ctr[3]++;
            }
        }
    }
}

// =============================================================================
// SHA256
// =============================================================================
__device__ __constant__ unsigned int SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};
__device__ __forceinline__ unsigned int rotr32(unsigned int x, int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ unsigned int rol32(unsigned int x, int n) {
    return (x << n) | (x >> (32 - n));
}
__device__ void sha256_dev(const unsigned char *data, int len, unsigned char out32[32]) {
    unsigned int h[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    unsigned char block[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < len; i++) block[i] = data[i];
    block[len] = 0x80;
    unsigned long long bitlen = (unsigned long long)len * 8ULL;
    for (int i = 0; i < 8; i++)
        block[56 + i] = (unsigned char)(bitlen >> (56 - 8 * i));
    unsigned int w[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = ((unsigned int)block[4*i    ] << 24)
             | ((unsigned int)block[4*i + 1] << 16)
             | ((unsigned int)block[4*i + 2] <<  8)
             | ((unsigned int)block[4*i + 3]);
    }
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        unsigned int s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        unsigned int s1 = rotr32(w[i- 2],17) ^ rotr32(w[i- 2],19) ^ (w[i- 2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    unsigned int a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        unsigned int S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        unsigned int ch = (e & f) ^ ((~e) & g);
        unsigned int t1 = hh + S1 + ch + SHA256_K[i] + w[i];
        unsigned int S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        unsigned int mj = (a & b) ^ (a & c) ^ (b & c);
        unsigned int t2 = S0 + mj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    unsigned int H[8] = {h[0]+a,h[1]+b,h[2]+c,h[3]+d,h[4]+e,h[5]+f,h[6]+g,h[7]+hh};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out32[4*i    ] = (unsigned char)(H[i] >> 24);
        out32[4*i + 1] = (unsigned char)(H[i] >> 16);
        out32[4*i + 2] = (unsigned char)(H[i] >>  8);
        out32[4*i + 3] = (unsigned char)(H[i]);
    }
}

// =============================================================================
// RIPEMD160
// =============================================================================
__device__ __constant__ int RIPEMD_R1[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__device__ __constant__ int RIPEMD_R2[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__device__ __constant__ int RIPEMD_S1[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__device__ __constant__ int RIPEMD_S2[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};
__device__ __constant__ unsigned int RIPEMD_K1[5] = {
    0x00000000u,0x5A827999u,0x6ED9EBA1u,0x8F1BBCDCu,0xA953FD4Eu
};
__device__ __constant__ unsigned int RIPEMD_K2[5] = {
    0x50A28BE6u,0x5C4DD124u,0x6D703EF3u,0x7A6D76E9u,0x00000000u
};
__device__ void ripemd160_dev(const unsigned char *data, int len, unsigned char out20[20]) {
    unsigned char block[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < len; i++) block[i] = data[i];
    block[len] = 0x80;
    unsigned long long bitlen = (unsigned long long)len * 8ULL;
    for (int i = 0; i < 8; i++)
        block[56 + i] = (unsigned char)(bitlen >> (8 * i));
    unsigned int x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        x[i] = (unsigned int)block[4*i]
             | ((unsigned int)block[4*i+1] <<  8)
             | ((unsigned int)block[4*i+2] << 16)
             | ((unsigned int)block[4*i+3] << 24);
    }
    unsigned int h0=0x67452301u,h1=0xEFCDAB89u,h2=0x98BADCFEu;
    unsigned int h3=0x10325476u,h4=0xC3D2E1F0u;
    unsigned int a1=h0,b1=h1,c1=h2,d1=h3,e1=h4;
    unsigned int a2=h0,b2=h1,c2=h2,d2=h3,e2=h4;
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        int r = i >> 4;
        unsigned int f1;
        if      (r == 0) f1 = b1 ^ c1 ^ d1;
        else if (r == 1) f1 = (b1 & c1) | (~b1 & d1);
        else if (r == 2) f1 = (b1 | ~c1) ^ d1;
        else if (r == 3) f1 = (b1 & d1) | (c1 & ~d1);
        else             f1 = b1 ^ (c1 | ~d1);
        unsigned int t = rol32(a1 + f1 + x[RIPEMD_R1[i]] + RIPEMD_K1[r], RIPEMD_S1[i]) + e1;
        a1=e1; e1=d1; d1=rol32(c1,10); c1=b1; b1=t;
        unsigned int f2;
        if      (r == 0) f2 = b2 ^ (c2 | ~d2);
        else if (r == 1) f2 = (b2 & d2) | (c2 & ~d2);
        else if (r == 2) f2 = (b2 | ~c2) ^ d2;
        else if (r == 3) f2 = (b2 & c2) | (~b2 & d2);
        else             f2 = b2 ^ c2 ^ d2;
        unsigned int t2 = rol32(a2 + f2 + x[RIPEMD_R2[i]] + RIPEMD_K2[r], RIPEMD_S2[i]) + e2;
        a2=e2; e2=d2; d2=rol32(c2,10); c2=b2; b2=t2;
    }
    unsigned int T = h1 + c1 + d2;
    h1 = h2 + d1 + e2;
    h2 = h3 + e1 + a2;
    h3 = h4 + a1 + b2;
    h4 = h0 + b1 + c2;
    h0 = T;
    unsigned int H[5] = {h0,h1,h2,h3,h4};
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        out20[4*i    ] = (unsigned char)(H[i]);
        out20[4*i + 1] = (unsigned char)(H[i] >>  8);
        out20[4*i + 2] = (unsigned char)(H[i] >> 16);
        out20[4*i + 3] = (unsigned char)(H[i] >> 24);
    }
}

// =============================================================================
// secp256k1
// =============================================================================
struct Fe { unsigned int v[8]; };
struct PointJ { Fe x, y, z; int inf; };

__device__ __constant__ unsigned int FE_P[8] = {
    0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,
    0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu
};
__device__ __constant__ unsigned int GX[8] = {
    0x16F81798u,0x59F2815Bu,0x2DCE28D9u,0x029BFCDBu,
    0xCE870B07u,0x55A06295u,0xF9DCBBACu,0x79BE667Eu
};
__device__ __constant__ unsigned int GY[8] = {
    0xFB10D4B8u,0x9C47D08Fu,0xA6855419u,0xFD17B448u,
    0x0E1108A8u,0x5DA4FBFCu,0x26A3C465u,0x483ADA77u
};
__device__ unsigned int g_comb_x[COMB_WINDOWS * COMB_ENTRIES][8];
__device__ unsigned int g_comb_y[COMB_WINDOWS * COMB_ENTRIES][8];

__device__ __forceinline__ void fe_zero(Fe &a) {
    #pragma unroll
    for (int i = 0; i < 8; i++) a.v[i] = 0;
}
__device__ __forceinline__ void fe_one(Fe &a) {
    a.v[0] = 1;
    #pragma unroll
    for (int i = 1; i < 8; i++) a.v[i] = 0;
}
__device__ __forceinline__ bool fe_is_zero(const Fe &a) {
    unsigned int x = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) x |= a.v[i];
    return x == 0;
}
__device__ __forceinline__ void fe_sub_p_once(Fe &a) {
    unsigned long long borrow = 0;
    for (int i = 0; i < 8; i++) {
        unsigned long long pv = (unsigned long long)FE_P[i] + borrow;
        unsigned long long av = a.v[i];
        a.v[i] = (unsigned int)(av - pv);
        borrow = (av < pv) ? 1ULL : 0ULL;
    }
}
__device__ __forceinline__ void fe_add(const Fe &a, const Fe &b, Fe &r) {
    unsigned long long carry = 0;
    for (int i = 0; i < 8; i++) {
        unsigned long long s = (unsigned long long)a.v[i] + b.v[i] + carry;
        r.v[i] = (unsigned int)s;
        carry = s >> 32;
    }
    bool ge = true;
    for (int i = 7; i >= 0; i--) {
        if (r.v[i] != FE_P[i]) { ge = (r.v[i] > FE_P[i]); break; }
    }
    if (carry || ge) fe_sub_p_once(r);
}
__device__ __forceinline__ void fe_sub(const Fe &a, const Fe &b, Fe &r) {
    unsigned long long borrow = 0;
    for (int i = 0; i < 8; i++) {
        unsigned long long bv = (unsigned long long)b.v[i] + borrow;
        unsigned long long av = a.v[i];
        r.v[i] = (unsigned int)(av - bv);
        borrow = (av < bv) ? 1ULL : 0ULL;
    }
    if (borrow) {
        unsigned long long carry = 0;
        for (int i = 0; i < 8; i++) {
            unsigned long long s = (unsigned long long)r.v[i] + FE_P[i] + carry;
            r.v[i] = (unsigned int)s;
            carry = s >> 32;
        }
    }
}
__device__ __forceinline__ void fe_mul(const Fe &a, const Fe &b, Fe &r) {
    unsigned int t[24];
    #pragma unroll
    for (int i = 0; i < 24; i++) t[i] = 0;
    for (int i = 0; i < 8; i++) {
        unsigned long long carry = 0;
        for (int j = 0; j < 8; j++) {
            unsigned long long cur = (unsigned long long)a.v[i]
                                   * (unsigned long long)b.v[j]
                                   + (unsigned long long)t[i + j] + carry;
            t[i + j] = (unsigned int)cur;
            carry = cur >> 32;
        }
        int k = i + 8;
        while (carry && k < 24) {
            unsigned long long cur = (unsigned long long)t[k] + carry;
            t[k] = (unsigned int)cur;
            carry = cur >> 32;
            ++k;
        }
    }
    for (int pass = 0; pass < 8; pass++) {
        for (int k = 23; k >= 8; k--) {
            unsigned int x = t[k];
            t[k] = 0;
            if (!x) continue;
            unsigned long long cur = (unsigned long long)t[k-8]
                                   + (unsigned long long)x * 977ULL;
            t[k-8] = (unsigned int)cur;
            unsigned long long carry = cur >> 32;
            cur = (unsigned long long)t[k-7] + (unsigned long long)x + carry;
            t[k-7] = (unsigned int)cur;
            carry = cur >> 32;
            int idx = k - 6;
            while (carry && idx < 24) {
                cur = (unsigned long long)t[idx] + carry;
                t[idx] = (unsigned int)cur;
                carry = cur >> 32;
                ++idx;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    for (int n = 0; n < 8; n++) {
        bool ge = true;
        for (int i = 7; i >= 0; i--) {
            if (r.v[i] != FE_P[i]) { ge = (r.v[i] > FE_P[i]); break; }
        }
        if (!ge) break;
        fe_sub_p_once(r);
    }
}
__device__ __forceinline__ void fe_sqr(const Fe &a, Fe &r) { fe_mul(a, a, r); }

__device__ void fe_inv(const Fe &a, Fe &r) {
    Fe x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t1,tmp;
    fe_sqr(a, tmp);   fe_mul(tmp, a, x2);
    fe_sqr(x2, tmp);  fe_mul(tmp, a, x3);
    x6 = x3;
    #pragma unroll
    for (int j = 0; j < 3; j++) { fe_sqr(x6, tmp); x6 = tmp; }
    fe_mul(x6, x3, tmp); x6 = tmp;
    x9 = x6;
    #pragma unroll
    for (int j = 0; j < 3; j++) { fe_sqr(x9, tmp); x9 = tmp; }
    fe_mul(x9, x3, tmp); x9 = tmp;
    x11 = x9;
    #pragma unroll
    for (int j = 0; j < 2; j++) { fe_sqr(x11, tmp); x11 = tmp; }
    fe_mul(x11, x2, tmp); x11 = tmp;
    x22 = x11;
    for (int j = 0; j < 11; j++) { fe_sqr(x22, tmp); x22 = tmp; }
    fe_mul(x22, x11, tmp); x22 = tmp;
    x44 = x22;
    for (int j = 0; j < 22; j++) { fe_sqr(x44, tmp); x44 = tmp; }
    fe_mul(x44, x22, tmp); x44 = tmp;
    x88 = x44;
    for (int j = 0; j < 44; j++) { fe_sqr(x88, tmp); x88 = tmp; }
    fe_mul(x88, x44, tmp); x88 = tmp;
    x176 = x88;
    for (int j = 0; j < 88; j++) { fe_sqr(x176, tmp); x176 = tmp; }
    fe_mul(x176, x88, tmp); x176 = tmp;
    x220 = x176;
    for (int j = 0; j < 44; j++) { fe_sqr(x220, tmp); x220 = tmp; }
    fe_mul(x220, x44, tmp); x220 = tmp;
    x223 = x220;
    #pragma unroll
    for (int j = 0; j < 3; j++) { fe_sqr(x223, tmp); x223 = tmp; }
    fe_mul(x223, x3, tmp); x223 = tmp;
    t1 = x223;
    for (int j = 0; j < 23; j++) { fe_sqr(t1, tmp); t1 = tmp; }
    fe_mul(t1, x22, tmp); t1 = tmp;
    #pragma unroll
    for (int j = 0; j < 5; j++) { fe_sqr(t1, tmp); t1 = tmp; }
    fe_mul(t1, a, tmp); t1 = tmp;
    #pragma unroll
    for (int j = 0; j < 3; j++) { fe_sqr(t1, tmp); t1 = tmp; }
    fe_mul(t1, x2, tmp); t1 = tmp;
    #pragma unroll
    for (int j = 0; j < 2; j++) { fe_sqr(t1, tmp); t1 = tmp; }
    fe_mul(t1, a, r);
}

__device__ __forceinline__ void point_inf(PointJ &p) {
    fe_zero(p.x); fe_zero(p.y); fe_zero(p.z);
    p.inf = 1;
}
__device__ void point_double(const PointJ &p, PointJ &r) {
    if (p.inf || fe_is_zero(p.y)) { point_inf(r); return; }
    Fe A,B,C,D,E,F,t1,t2,eightC;
    fe_sqr(p.x, A); fe_sqr(p.y, B); fe_sqr(B, C);
    fe_add(p.x, B, t1); fe_sqr(t1, t2); fe_sub(t2, A, t1);
    fe_sub(t1, C, t2); fe_add(t2, t2, D); fe_add(A, A, t1);
    fe_add(t1, A, E); fe_sqr(E, F); fe_add(D, D, t1);
    fe_sub(F, t1, r.x); fe_sub(D, r.x, t1); fe_mul(E, t1, t2);
    fe_add(C, C, eightC); fe_add(eightC, eightC, eightC);
    fe_add(eightC, eightC, eightC); fe_sub(t2, eightC, r.y);
    fe_mul(p.y, p.z, t1); fe_add(t1, t1, r.z); r.inf = 0;
}
__device__ __forceinline__ void point_add_affine(
    const PointJ &p, const Fe &Qx, const Fe &Qy, PointJ &r)
{
    if (p.inf) {
        r.x = Qx; r.y = Qy; fe_one(r.z); r.inf = 0; return;
    }
    Fe Z1Z1, U2, S2, H, R, H2, H3, V, t1, t2;
    fe_sqr(p.z, Z1Z1);
    fe_mul(Qx, Z1Z1, U2);
    fe_mul(Z1Z1, p.z, t1);
    fe_mul(Qy, t1, S2);
    fe_sub(U2, p.x, H);
    fe_sub(S2, p.y, R);
    if (fe_is_zero(H)) {
        if (fe_is_zero(R)) { point_double(p, r); return; }
        point_inf(r); return;
    }
    fe_sqr(H, H2);
    fe_mul(H2, H, H3);
    fe_mul(p.x, H2, V);
    fe_sqr(R, t1);
    fe_sub(t1, H3, t2);
    fe_add(V, V, t1);
    fe_sub(t2, t1, r.x);
    fe_sub(V, r.x, t1);
    fe_mul(R, t1, t2);
    fe_mul(p.y, H3, t1);
    fe_sub(t2, t1, r.y);
    fe_mul(p.z, H, r.z);
    r.inf = 0;
}
__device__ __forceinline__ void scalar_mul_G_comb(
    unsigned char hi8, unsigned long long lo64,
    const Fe *s_cx, const Fe *s_cy, PointJ &r)
{
    point_inf(r);
    #pragma unroll 1
    for (int i = 0; i < 16; i++) {
        unsigned int nib = (unsigned int)((lo64 >> (i * 4)) & 0xFULL);
        if (nib == 0) continue;
        int idx = i * COMB_ENTRIES + (nib - 1);
        PointJ t;
        point_add_affine(r, s_cx[idx], s_cy[idx], t);
        r = t;
    }
    #pragma unroll 1
    for (int i = 0; i < 2; i++) {
        unsigned int nib = (unsigned int)((hi8 >> (i * 4)) & 0xF);
        if (nib == 0) continue;
        int idx = (16 + i) * COMB_ENTRIES + (nib - 1);
        PointJ t;
        point_add_affine(r, s_cx[idx], s_cy[idx], t);
        r = t;
    }
}

__device__ __forceinline__ void warp_batch_inv(Fe &z) {
    const unsigned FULL = 0xffffffffu;
    const unsigned lane = threadIdx.x & 31u;
    Fe P = z;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        Fe other;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            other.v[i] = __shfl_up_sync(FULL, P.v[i], off);
        if (lane >= (unsigned)off) {
            Fe t; fe_mul(other, P, t); P = t;
        }
    }
    Fe inv_all;
    if (lane == 31) fe_inv(P, inv_all);
    #pragma unroll
    for (int i = 0; i < 8; i++)
        inv_all.v[i] = __shfl_sync(FULL, inv_all.v[i], 31);
    Fe suf = z;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        Fe other;
        #pragma unroll
        for (int i = 0; i < 8; i++)
            other.v[i] = __shfl_down_sync(FULL, suf.v[i], off);
        if (lane + (unsigned)off < 32u) {
            Fe t; fe_mul(other, suf, t); suf = t;
        }
    }
    Fe S;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        S.v[i] = __shfl_down_sync(FULL, suf.v[i], 1);
    if (lane == 31) fe_one(S);
    Fe Q;
    fe_mul(inv_all, S, Q);
    Fe Pm1;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        Pm1.v[i] = __shfl_up_sync(FULL, P.v[i], 1);
    if (lane == 0) fe_one(Pm1);
    fe_mul(Q, Pm1, z);
}

// =============================================================================
// Feistel
// =============================================================================
__device__ __forceinline__
unsigned long long feistel_round_f(unsigned long long r, unsigned long long rk) {
    unsigned long long x = r ^ rk;
    x = splitmix64_impl(x);
    return x;
}
__device__ __forceinline__
void feistel_perm(
    unsigned int in_hi, unsigned long long in_lo,
    int half_bits, unsigned long long mask_half,
    unsigned long long master_key,
    unsigned int &out_hi, unsigned long long &out_lo)
{
    unsigned long long L = ((unsigned long long)in_hi << (64 - half_bits))
                         | (in_lo >> half_bits);
    L &= mask_half;
    unsigned long long R = in_lo & mask_half;
    #pragma unroll
    for (int r = 0; r < FEISTEL_ROUNDS; r++) {
        unsigned long long rk =
            master_key ^ ((unsigned long long)r * 0x9E3779B97F4A7C15ULL);
        unsigned long long fr = feistel_round_f(R, rk) & mask_half;
        unsigned long long newL = R;
        unsigned long long newR = (L ^ fr) & mask_half;
        L = newL; R = newR;
    }
    if (half_bits <= 32) {
        out_hi = 0;
        out_lo = (L << half_bits) | R;
    } else {
        int lo_bits = 64 - half_bits;
        unsigned long long L_mask = (1ULL << lo_bits) - 1ULL;
        unsigned long long L_lo = L & L_mask;
        unsigned long long L_hi = L >> lo_bits;
        out_hi = (unsigned int)L_hi;
        out_lo = (L_lo << half_bits) | R;
    }
}
__device__ __forceinline__
void cycle_walk_feistel(
    unsigned int i_hi, unsigned long long i_lo,
    unsigned int N_hi, unsigned long long N_lo,
    int half_bits, unsigned long long mask_half,
    unsigned long long master_key,
    unsigned int &out_hi, unsigned long long &out_lo)
{
    unsigned int cur_hi = i_hi;
    unsigned long long cur_lo = i_lo;
    #pragma unroll 1
    for (int tries = 0; tries < 64; tries++) {
        unsigned int p_hi; unsigned long long p_lo;
        feistel_perm(cur_hi, cur_lo, half_bits, mask_half, master_key, p_hi, p_lo);
        if (p_hi < N_hi || (p_hi == N_hi && p_lo < N_lo)) {
            out_hi = p_hi; out_lo = p_lo; return;
        }
        cur_hi = p_hi; cur_lo = p_lo;
    }
    out_hi = cur_hi; out_lo = cur_lo;
}

// =============================================================================
// Batch inversion la nivel de bloc (inspirat din VanitySearch/Rotor-Cuda)
// =============================================================================
template<int GROUP>
__device__ void block_batch_inv(Fe *z, Fe *scratch) {
    const unsigned int tid = threadIdx.x;
    if (GROUP <= 32) {
        warp_batch_inv(z[tid]);
        return;
    }
    __shared__ Fe sh_prefix[GROUP];
    sh_prefix[tid] = z[tid];
    __syncthreads();
    warp_batch_inv(z[tid]);
}

// =============================================================================
// Kernels templated pe KPT (keys per thread)
// =============================================================================
template<int KPT>
__global__ void __launch_bounds__(BLOCK_SIZE)
search_kernel_random_t(
    unsigned char start_hi, unsigned long long start_lo,
    unsigned char range_hi, unsigned long long range_lo,
    unsigned char mask_hi, unsigned long long mask_lo,
    unsigned long long base_seed,
    unsigned long long rkey,
    unsigned long long iteration,
    const unsigned char *__restrict__ target_h160,
    unsigned int *__restrict__ match_count,
    unsigned long long *__restrict__ match_keys)
{
    __shared__ Fe s_cx[COMB_WINDOWS * COMB_ENTRIES];
    __shared__ Fe s_cy[COMB_WINDOWS * COMB_ENTRIES];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * BLOCK_SIZE + tid;
    const int TOTAL = COMB_WINDOWS * COMB_ENTRIES;
    #pragma unroll 1
    for (int k = tid; k < TOTAL; k += BLOCK_SIZE) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            s_cx[k].v[j] = g_comb_x[k][j];
            s_cy[k].v[j] = g_comb_y[k][j];
        }
    }
    __syncthreads();
    unsigned long long refresh_mix = 0;
    if (rkey != 0) {
        unsigned long long cycle = iteration / rkey;
        refresh_mix = splitmix64_impl(cycle * 0xD1B54A32D192ED03ULL);
    }
    unsigned long long thread_seed = splitmix64_impl(
        base_seed + (unsigned long long)gid + refresh_mix);
    unsigned int philox_key[2];
    unsigned int philox_ctr[4];
    philox_init(philox_key, philox_ctr, thread_seed, (unsigned long long)gid);

    #pragma unroll 1
    for (unsigned int kk = 0; kk < KPT; kk++) {
        unsigned char      key_hi = 0;
        unsigned long long key_lo = 0;
        bool found = false;
        while (!found) {
            unsigned int blk[16];
            philox_block16(philox_key, philox_ctr, blk);
            unsigned char cand_hi = 0; unsigned long long cand_lo = 0; bool any = false;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned long long r_lo = ((unsigned long long)blk[4*i])
                                        | ((unsigned long long)blk[4*i + 1] << 32);
                unsigned long long r_hi = ((unsigned long long)blk[4*i + 2])
                                        | ((unsigned long long)blk[4*i + 3] << 32);
                unsigned long long o_lo = r_lo & mask_lo;
                unsigned char      o_hi = (unsigned char)(r_hi & mask_hi);
                bool accept = (o_hi < range_hi) ||
                              (o_hi == range_hi && o_lo < range_lo);
                if (accept) { if (!any) { cand_hi = o_hi; cand_lo = o_lo; } any = true; }
            }
            if (any) {
                key_lo = start_lo + cand_lo;
                key_hi = (unsigned char)(start_hi + cand_hi +
                                         (key_lo < start_lo ? 1 : 0));
                found = true;
            }
        }
        PointJ p;
        scalar_mul_G_comb(key_hi, key_lo, s_cx, s_cy, p);
        warp_batch_inv(p.z);
        Fe inv_z = p.z;
        Fe zi2, x_aff, y_aff, t;
        fe_sqr(inv_z, zi2);
        fe_mul(p.x, zi2, x_aff);
        fe_mul(p.y, inv_z, t);
        fe_mul(t, zi2, y_aff);
        unsigned char pub[33];
        pub[0] = (y_aff.v[0] & 1u) ? 0x03 : 0x02;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int w = x_aff.v[7 - i];
            pub[1 + i*4 + 0] = (unsigned char)(w >> 24);
            pub[1 + i*4 + 1] = (unsigned char)(w >> 16);
            pub[1 + i*4 + 2] = (unsigned char)(w >>  8);
            pub[1 + i*4 + 3] = (unsigned char)w;
        }
        unsigned char h32[32];
        sha256_dev(pub, 33, h32);
        unsigned char h20[20];
        ripemd160_dev(h32, 32, h20);
        bool eq = true;
        #pragma unroll
        for (int i = 0; i < 20; i++)
            if (h20[i] != target_h160[i]) { eq = false; break; }
        if (eq) {
            unsigned int slot = atomicAdd(match_count, 1u);
            if (slot < MAX_MATCHES) {
                match_keys[2 * slot]     = (unsigned long long)key_hi;
                match_keys[2 * slot + 1] = key_lo;
            }
        }
    }
}

template<int KPT>
__global__ void __launch_bounds__(BLOCK_SIZE)
search_kernel_feistel_t(
    unsigned char start_hi, unsigned long long start_lo,
    unsigned int  N_hi,     unsigned long long N_lo,
    int half_bits, unsigned long long mask_half,
    unsigned int  batch_off_hi, unsigned long long batch_off_lo,
    unsigned long long master_key,
    unsigned long long step,
    const unsigned char *__restrict__ target_h160,
    unsigned int *__restrict__ match_count,
    unsigned long long *__restrict__ match_keys)
{
    __shared__ Fe s_cx[COMB_WINDOWS * COMB_ENTRIES];
    __shared__ Fe s_cy[COMB_WINDOWS * COMB_ENTRIES];
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * BLOCK_SIZE + tid;
    const int TOTAL = COMB_WINDOWS * COMB_ENTRIES;
    #pragma unroll 1
    for (int k = tid; k < TOTAL; k += BLOCK_SIZE) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            s_cx[k].v[j] = g_comb_x[k][j];
            s_cy[k].v[j] = g_comb_y[k][j];
        }
    }
    __syncthreads();
    unsigned int i_hi = batch_off_hi;
    unsigned long long i_lo = batch_off_lo + (unsigned long long)gid * KPT;
    if (i_lo < batch_off_lo) i_hi += 1;
    #pragma unroll 1
    for (unsigned int kk = 0; kk < KPT; kk++) {
        bool in_range = (i_hi < N_hi) || (i_hi == N_hi && i_lo < N_lo);
        if (!in_range) break;
        unsigned int perm_hi; unsigned long long perm_lo;
        cycle_walk_feistel(i_hi, i_lo, N_hi, N_lo,
                           half_bits, mask_half, master_key,
                           perm_hi, perm_lo);
        unsigned long long a0 = perm_lo & 0xFFFFFFFFULL;
        unsigned long long a1 = perm_lo >> 32;
        unsigned long long b0 = step & 0xFFFFFFFFULL;
        unsigned long long b1 = step >> 32;
        unsigned long long p00 = a0 * b0;
        unsigned long long p01 = a0 * b1;
        unsigned long long p10 = a1 * b0;
        unsigned long long p11 = a1 * b1;
        unsigned long long mid = (p00 >> 32) + (p01 & 0xFFFFFFFFULL) + (p10 & 0xFFFFFFFFULL);
        unsigned long long mul_lo = (mid << 32) | (p00 & 0xFFFFFFFFULL);
        unsigned long long mul_hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
        unsigned long long key_lo = start_lo + mul_lo;
        unsigned int tmp_hi = (unsigned int)start_hi
                            + (unsigned int)(mul_hi & 0xFFULL)
                            + (unsigned int)(key_lo < start_lo ? 1U : 0U);
        unsigned char key_hi = (unsigned char)tmp_hi;
        PointJ p;
        scalar_mul_G_comb(key_hi, key_lo, s_cx, s_cy, p);
        warp_batch_inv(p.z);
        Fe inv_z = p.z;
        Fe zi2, x_aff, y_aff, t;
        fe_sqr(inv_z, zi2);
        fe_mul(p.x, zi2, x_aff);
        fe_mul(p.y, inv_z, t);
        fe_mul(t, zi2, y_aff);
        unsigned char pub[33];
        pub[0] = (y_aff.v[0] & 1u) ? 0x03 : 0x02;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            unsigned int w = x_aff.v[7 - i];
            pub[1 + i*4 + 0] = (unsigned char)(w >> 24);
            pub[1 + i*4 + 1] = (unsigned char)(w >> 16);
            pub[1 + i*4 + 2] = (unsigned char)(w >>  8);
            pub[1 + i*4 + 3] = (unsigned char)w;
        }
        unsigned char h32[32];
        sha256_dev(pub, 33, h32);
        unsigned char h20[20];
        ripemd160_dev(h32, 32, h20);
        bool eq = true;
        #pragma unroll
        for (int i = 0; i < 20; i++)
            if (h20[i] != target_h160[i]) { eq = false; break; }
        if (eq) {
            unsigned int slot = atomicAdd(match_count, 1u);
            if (slot < MAX_MATCHES) {
                match_keys[2 * slot]     = (unsigned long long)key_hi;
                match_keys[2 * slot + 1] = key_lo;
            }
        }
        i_lo++;
        if (i_lo == 0) i_hi++;
    }
}

// =============================================================================
// Launch macros
// =============================================================================
#define LAUNCH_RANDOM(KPT)                                                  \
    search_kernel_random_t<KPT><<<blocks, BLOCK_SIZE>>>(                    \
        b_start_hi, b_start_lo, b_range_hi, b_range_lo,                     \
        b_mask_hi, b_mask_lo, bs, rkey, iteration,                          \
        d_target, d_match_count, d_match_keys)

#define LAUNCH_FEISTEL(KPT)                                                 \
    search_kernel_feistel_t<KPT><<<blocks, BLOCK_SIZE>>>(                   \
        (unsigned char)(start_u.hi & 0xFFu), start_u.lo,                    \
        N_hi_u, N_lo_u, half_bits, mask_half,                               \
        (unsigned int)(cursor.hi & 0xFFu), cursor.lo,                       \
        master_key, step,                                                   \
        d_target, d_match_count, d_match_keys)

// =============================================================================
// Host helpers
// =============================================================================
static void die_cuda(cudaError_t e, const char *where) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}
static std::vector<unsigned char> hex_to_bytes(const std::string &s) {
    std::string x = s;
    if (x.size() % 2) x = "0" + x;
    std::vector<unsigned char> out(x.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        unsigned v = 0;
        std::sscanf(x.substr(i * 2, 2).c_str(), "%02x", &v);
        out[i] = (unsigned char)v;
    }
    return out;
}
static std::string hex_upper(const unsigned char *p, size_t n) {
    std::ostringstream o;
    o << std::uppercase << std::hex << std::setfill('0');
    for (size_t i = 0; i < n; i++) o << std::setw(2) << (unsigned)p[i];
    return o.str();
}
static std::string base58(const std::vector<unsigned char> &in) {
    static const char *A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    size_t zeros = 0;
    while (zeros < in.size() && in[zeros] == 0) ++zeros;
    std::vector<unsigned char> b(in.begin(), in.end());
    std::vector<char> out;
    size_t start = zeros;
    while (start < b.size()) {
        unsigned carry = 0;
        for (size_t i = start; i < b.size(); i++) {
            unsigned cur = (unsigned)b[i] + carry * 256u;
            b[i] = (unsigned char)(cur / 58u);
            carry = cur % 58u;
        }
        out.push_back(A[carry]);
        while (start < b.size() && b[start] == 0) ++start;
    }
    std::string s(zeros, '1');
    for (size_t i = 0; i < out.size(); i++) s.push_back(out[out.size()-1-i]);
    return s;
}
static int b58_val(char c) {
    static const char *A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for (int i = 0; i < 58; i++) if (A[i] == c) return i;
    return -1;
}
static bool base58_decode(const std::string &s, std::vector<unsigned char> &out) {
    size_t zeros = 0;
    while (zeros < s.size() && s[zeros] == '1') ++zeros;
    std::vector<unsigned char> b256;
    for (size_t i = 0; i < s.size(); i++) {
        int d = b58_val(s[i]);
        if (d < 0) return false;
        unsigned carry = (unsigned)d;
        for (size_t j = 0; j < b256.size(); j++) {
            unsigned cur = (unsigned)b256[j] * 58u + carry;
            b256[j] = (unsigned char)(cur & 0xFFu);
            carry = cur >> 8;
        }
        while (carry) {
            b256.push_back((unsigned char)(carry & 0xFFu));
            carry >>= 8;
        }
    }
    while (!b256.empty() && b256.back() == 0) b256.pop_back();
    out.clear();
    out.assign(zeros, 0);
    for (size_t i = b256.size(); i > 0; i--) out.push_back(b256[i - 1]);
    return true;
}
static std::string p2pkh_from_compressed(const unsigned char pub[33]) {
    unsigned char sha[SHA256_DIGEST_LENGTH];
    SHA256(pub, 33, sha);
    unsigned char ripe[RIPEMD160_DIGEST_LENGTH];
    RIPEMD160(sha, SHA256_DIGEST_LENGTH, ripe);
    std::vector<unsigned char> payload(21);
    payload[0] = 0x00;
    std::memcpy(payload.data() + 1, ripe, 20);
    unsigned char c1[32], c2[32];
    SHA256(payload.data(), payload.size(), c1);
    SHA256(c1, 32, c2);
    payload.insert(payload.end(), c2, c2 + 4);
    return base58(payload);
}
static bool decode_p2pkh_target(const std::string &addr, unsigned char out_h160[20]) {
    std::vector<unsigned char> payload;
    if (!base58_decode(addr, payload)) return false;
    if (payload.size() != 25) return false;
    if (payload[0] != 0x00) return false;
    unsigned char c1[32], c2[32];
    SHA256(payload.data(), 21, c1);
    SHA256(c1, 32, c2);
    if (std::memcmp(c2, payload.data() + 21, 4) != 0) return false;
    std::memcpy(out_h160, payload.data() + 1, 20);
    return true;
}
static bool cpu_pubkey(const std::vector<unsigned char> &key32,
                       unsigned char *out, size_t &outlen) {
    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    secp256k1_pubkey pk;
    bool ok = secp256k1_ec_pubkey_create(ctx, &pk, key32.data());
    if (ok) {
        outlen = 33;
        ok = secp256k1_ec_pubkey_serialize(ctx, out, &outlen, &pk,
                                           SECP256K1_EC_COMPRESSED);
    }
    secp256k1_context_destroy(ctx);
    return ok;
}
static void build_comb_table_and_upload() {
    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    unsigned char g33[33] = {
        0x02, 0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,0x55,0xA0,0x62,
        0x95,0xCE,0x87,0x0B,0x07,0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,
        0xD9,0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
    };
    secp256k1_pubkey base;
    if (!secp256k1_ec_pubkey_parse(ctx, &base, g33, 33)) {
        std::cerr << "comb: parse G failed\n"; std::exit(1);
    }
    const int TOTAL = COMB_WINDOWS * COMB_ENTRIES;
    std::vector<unsigned int> h_x(TOTAL * 8, 0), h_y(TOTAL * 8, 0);
    for (int w = 0; w < COMB_WINDOWS; w++) {
        for (int v = 1; v <= 15; v++) {
            secp256k1_pubkey pk = base;
            unsigned char scalar[32] = {0};
            scalar[31] = (unsigned char)v;
            if (!secp256k1_ec_pubkey_tweak_mul(ctx, &pk, scalar)) {
                std::cerr << "comb: tweak_mul failed\n"; std::exit(1);
            }
            unsigned char ser[65]; size_t slen = 65;
            if (!secp256k1_ec_pubkey_serialize(ctx, ser, &slen, &pk,
                                               SECP256K1_EC_UNCOMPRESSED)) {
                std::cerr << "comb: serialize failed\n"; std::exit(1);
            }
            int idx = w * COMB_ENTRIES + (v - 1);
            for (int j = 0; j < 8; j++) {
                unsigned int xw = ((unsigned int)ser[1  + j*4    ] << 24)
                                | ((unsigned int)ser[1  + j*4 + 1] << 16)
                                | ((unsigned int)ser[1  + j*4 + 2] <<  8)
                                | ((unsigned int)ser[1  + j*4 + 3]);
                unsigned int yw = ((unsigned int)ser[33 + j*4    ] << 24)
                                | ((unsigned int)ser[33 + j*4 + 1] << 16)
                                | ((unsigned int)ser[33 + j*4 + 2] <<  8)
                                | ((unsigned int)ser[33 + j*4 + 3]);
                h_x[idx * 8 + (7 - j)] = xw;
                h_y[idx * 8 + (7 - j)] = yw;
            }
        }
        unsigned char s16[32] = {0};
        s16[31] = 16;
        if (!secp256k1_ec_pubkey_tweak_mul(ctx, &base, s16)) {
            std::cerr << "comb: base*16 failed\n"; std::exit(1);
        }
    }
    secp256k1_context_destroy(ctx);
    die_cuda(cudaMemcpyToSymbol(g_comb_x, h_x.data(),
                                h_x.size() * sizeof(unsigned int)),
             "memcpy comb_x");
    die_cuda(cudaMemcpyToSymbol(g_comb_y, h_y.data(),
                                h_y.size() * sizeof(unsigned int)),
             "memcpy comb_y");
}
static bool parse_hex72(const std::string &in, unsigned char out[9]) {
    std::string x = in;
    if (x.size() >= 2 && x[0] == '0' && (x[1] == 'x' || x[1] == 'X'))
        x = x.substr(2);
    if (x.empty() || x.size() > 18) return false;
    for (size_t i = 0; i < x.size(); i++)
        if (!std::isxdigit((unsigned char)x[i])) return false;
    while (x.size() < 18) x = "0" + x;
    for (int i = 0; i < 9; i++) {
        unsigned v = 0;
        std::sscanf(x.substr(i * 2, 2).c_str(), "%02x", &v);
        out[i] = (unsigned char)v;
    }
    return true;
}
struct Key128 { unsigned long long hi, lo; };
static Key128 key9_to_k128(const unsigned char k[9]) {
    Key128 r; r.hi = 0; r.lo = 0;
    for (int i = 0; i < 9; i++) {
        r.hi = (r.hi << 8) | (r.lo >> 56);
        r.lo = (r.lo << 8) | (unsigned long long)k[i];
    }
    return r;
}
static bool k128_ge(Key128 a, Key128 b) {
    if (a.hi != b.hi) return a.hi > b.hi;
    return a.lo >= b.lo;
}
static bool k128_is_zero(Key128 a) { return (a.hi == 0 && a.lo == 0); }
static Key128 k128_sub(Key128 a, Key128 b) {
    Key128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - (a.lo < b.lo ? 1ULL : 0ULL);
    return r;
}
static Key128 k128_add_u64(Key128 a, unsigned long long v) {
    Key128 r;
    r.lo = a.lo + v;
    r.hi = a.hi + (r.lo < a.lo ? 1ULL : 0ULL);
    return r;
}
static std::string k128_to_hex9(Key128 k) {
    unsigned char b[9];
    b[0] = (unsigned char)(k.hi & 0xFFu);
    for (int i = 0; i < 8; i++)
        b[8 - i] = (unsigned char)(k.lo >> (8 * i));
    return hex_upper(b, 9);
}
static std::string k128_to_priv_hex64(Key128 k) {
    unsigned char b[32] = {0};
    b[23] = (unsigned char)(k.hi & 0xFFu);
    for (int i = 0; i < 8; i++)
        b[31 - i] = (unsigned char)(k.lo >> (8 * i));
    return hex_upper(b, 32);
}
static int bit_length_72(unsigned char hi, unsigned long long lo) {
    if (hi != 0) {
        int b = 0; unsigned char x = hi;
        while (x >>= 1) b++;
        return 64 + b + 1;
    }
    if (lo == 0) return 0;
    return 64 - __builtin_clzll(lo);
}
static unsigned long long splitmix64_host(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static void mask_for_bits(int nbits, unsigned char &mhi, unsigned long long &mlo) {
    if (nbits <= 64) {
        if (nbits == 0)        mlo = 0;
        else if (nbits == 64)  mlo = ~0ULL;
        else                   mlo = (1ULL << nbits) - 1ULL;
        mhi = 0;
    } else {
        int hb = nbits - 64;
        mlo = ~0ULL;
        mhi = (unsigned char)((1U << hb) - 1U);
    }
}
static void run_gpu_test(const std::string &hex,
                         const std::string &expected_addr,
                         bool quiet) {
    std::vector<unsigned char> kb = hex_to_bytes(hex);
    if (kb.size() != 9) { std::cerr << "test key bad\n"; std::exit(1); }
    std::vector<unsigned char> key32(32, 0);
    std::copy(kb.begin(), kb.end(), key32.begin() + 23);
    unsigned char cpu[65]; size_t cpu_len = 0;
    if (!cpu_pubkey(key32, cpu, cpu_len)) {
        std::cerr << "libsecp rejected test key\n"; std::exit(1);
    }
    if (!quiet) {
        std::cout << "\n=== SELF TEST ===\n";
        std::cout << "CPU pubkey  : " << hex_upper(cpu, 33) << "\n";
        std::cout << "CPU address : " << p2pkh_from_compressed(cpu) << "\n";
        std::cout << "Expected    : " << expected_addr << "\n";
    }
    if (p2pkh_from_compressed(cpu) != expected_addr) {
        std::cerr << "SELF TEST FAILED\n"; std::exit(2);
    }
    if (!quiet) std::cout << "SELF TEST OK (CPU-only check)\n";
}

// =============================================================================
// Checkpoint helpers (v5)
// =============================================================================
static bool load_checkpoint(Key128 &cursor, std::string &range_saved) {
    std::ifstream f(CKPT_FILE);
    if (!f.is_open()) return false;
    std::string hex;
    f >> hex >> range_saved;
    if (hex.empty() || hex.size() > 18) return false;
    for (size_t i = 0; i < hex.size(); i++)
        if (!std::isxdigit((unsigned char)hex[i])) return false;
    std::string x = hex;
    while (x.size() < 18) x = "0" + x;
    unsigned char bytes[9];
    for (int i = 0; i < 9; i++) {
        unsigned v = 0;
        std::sscanf(x.substr(i * 2, 2).c_str(), "%02x", &v);
        bytes[i] = (unsigned char)v;
    }
    cursor = key9_to_k128(bytes);
    return true;
}
static void save_checkpoint(Key128 cursor, const std::string &range_str) {
    std::ofstream f(CKPT_FILE);
    if (!f.is_open()) return;
    f << k128_to_hex9(cursor) << " " << range_str << "\n";
    f.close();
}

// =============================================================================
// >>>>>>>>>>>>>>>>>>  FOUND KEY SAVING (NOU)  <<<<<<<<<<<<<<<<<<<
// =============================================================================
// Salveaza adresa si cheia privata (hex 32B + hex 9B) intr-un fisier text,
// in mod APPEND (ca sa nu pierdem chei gasite anterior).
// Fisierul default: "found_keys.txt"
// =============================================================================
static void save_found_key(Key128 mk, const std::string &address,
                           const std::string &range_str) {
    std::ofstream f(FOUND_FILE, std::ios::app);
    if (!f.is_open()) {
        std::cerr << "\nWARNING: nu am putut deschide " << FOUND_FILE
                  << " pentru scriere. Cheia NU a fost salvata pe disc!\n";
        return;
    }
    // Timestamp
    std::time_t now = std::time(nullptr);
    char tbuf[64];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    f << "========================================\n";
    f << "Address   : " << address << "\n";
    f << "PrivKey32 : " << k128_to_priv_hex64(mk) << "\n";
    f << "PrivKey9  : " << k128_to_hex9(mk) << "\n";
    f << "Range     : " << range_str << "\n";
    f << "Found at  : " << tbuf << "\n";
    f << "========================================\n";
    f.close();
}

// =============================================================================
// Main
// =============================================================================
static volatile std::sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

enum Mode { MODE_V2, MODE_V3, MODE_V4, MODE_V5 };

static bool is_flag(const std::string &a, const char *name) {
    std::string s1 = "-";  s1 += name;
    std::string s2 = "--"; s2 += name;
    return a == s1 || a == s2;
}

static std::string fmt_num(unsigned long long n) {
    std::string s = std::to_string(n);
    std::string out;
    int cnt = 0;
    for (int i = (int)s.size() - 1; i >= 0; i--) {
        out.push_back(s[i]);
        cnt++;
        if (cnt % 3 == 0 && i > 0) out.push_back(',');
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// Split "N,M" in two unsigned ints (stil Rotor-Cuda --gpux)
static bool parse_gpux(const std::string &s, unsigned int &nx, unsigned int &ny) {
    auto pos = s.find(',');
    if (pos == std::string::npos) return false;
    std::string a = s.substr(0, pos), b = s.substr(pos + 1);
    if (a.empty() || b.empty()) return false;
    for (char c : a) if (!std::isdigit((unsigned char)c)) return false;
    for (char c : b) if (!std::isdigit((unsigned char)c)) return false;
    nx = (unsigned int)std::strtoul(a.c_str(), NULL, 10);
    ny = (unsigned int)std::strtoul(b.c_str(), NULL, 10);
    return (nx > 0 && ny > 0);
}

int main(int argc, char **argv) {
    Mode mode = MODE_V3;
    unsigned long long step = 1;
    unsigned long long v5_block  = DEFAULT_V5_BLOCK;
    unsigned long long v5_sample = DEFAULT_V5_SAMPLE;
    bool reset_ckpt = false;
    bool quiet = false;
    bool bench = false;
    int  kpt = DEFAULT_KPT;
    unsigned int blocks_override = 0;
    unsigned int batch_count = 0;
    unsigned int gpux = 0, gpuy = 0;
    unsigned long long rkey = 0;
    bool rkey_set = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (is_flag(a, "v2")) mode = MODE_V2;
        else if (is_flag(a, "v3")) mode = MODE_V3;
        else if (is_flag(a, "v4")) mode = MODE_V4;
        else if (is_flag(a, "v5")) mode = MODE_V5;
        else if (is_flag(a, "quiet") || a == "-q") quiet = true;
        else if (is_flag(a, "bench")) bench = true;
        else if (is_flag(a, "kpt")) {
            if (i + 1 >= argc) { std::cerr << "kpt requires value\n"; return 1; }
            kpt = std::atoi(argv[++i]);
            if (kpt != 1 && kpt != 2 && kpt != 4 && kpt != 8 && kpt != 16) {
                std::cerr << "kpt must be 1, 2, 4, 8 or 16\n"; return 1;
            }
        }
        else if (is_flag(a, "blocks")) {
            if (i + 1 >= argc) { std::cerr << "blocks requires value\n"; return 1; }
            blocks_override = (unsigned int)std::strtoul(argv[++i], NULL, 10);
        }
        else if (is_flag(a, "gpux")) {
            if (i + 1 >= argc) { std::cerr << "gpux requires N,M\n"; return 1; }
            if (!parse_gpux(argv[++i], gpux, gpuy)) {
                std::cerr << "gpux must be N,M (ex: 18000,512)\n"; return 1;
            }
        }
        else if (is_flag(a, "rkey")) {
            if (i + 1 >= argc) { std::cerr << "rkey requires value\n"; return 1; }
            rkey = std::strtoull(argv[++i], NULL, 10);
            rkey_set = true;
        }
        else if (is_flag(a, "step")) {
            if (i + 1 >= argc) { std::cerr << "step requires value\n"; return 1; }
            step = std::strtoull(argv[++i], NULL, 10);
            if (step == 0) { std::cerr << "step must be >= 1\n"; return 1; }
        }
        else if (is_flag(a, "block")) {
            if (i + 1 >= argc) { std::cerr << "block requires value\n"; return 1; }
            v5_block = std::strtoull(argv[++i], NULL, 10);
            if (v5_block == 0) { std::cerr << "block must be >= 1\n"; return 1; }
        }
        else if (is_flag(a, "sample")) {
            if (i + 1 >= argc) { std::cerr << "sample requires value\n"; return 1; }
            v5_sample = std::strtoull(argv[++i], NULL, 10);
            if (v5_sample == 0) { std::cerr << "sample must be >= 1\n"; return 1; }
        }
        else if (is_flag(a, "reset")) reset_ckpt = true;
        else if (a == "-h" || a == "--help") {
            std::cerr << "Usage: " << argv[0]
                      << " [-v2|-v3|-v4|-v5] [opts] START:END TARGET [BATCH]\n"
                      << "  -v2 | --v2 : Philox + cursor + sub-ferestre\n"
                      << "  -v3 | --v3 : Feistel permutation (DEFAULT)\n"
                      << "  -v4 | --v4 : Philox + random pur, la infinit\n"
                      << "  -v5 | --v5 : blocuri + random + checkpoint\n"
                      << "  ---- tuning (Rotor-Cuda style) ----\n"
                      << "       -kpt N      : keys per thread (1,2,4,8,16; default 4)\n"
                      << "       -blocks N   : override grid (blocuri)\n"
                      << "       -gpux N,M   : grid 2D (stil Rotor-Cuda; ex: 18000,512)\n"
                      << "       -rkey N     : refresh start points la N chei\n"
                      << "                     (0 = default 1e11 pentru v4/v5, off pt v2/v3)\n"
                      << "       -q|--quiet  : reduce I/O progres\n"
                      << "       -bench      : benchmark KPT=4,8,16 (30s fiecare) si exit\n"
                      << "  ---- specifice modului ----\n"
                      << "       -block  N   : dimensiunea blocului v5 (default 1e9)\n"
                      << "       -sample N   : chei per bloc v5 (default 1e7)\n"
                      << "       -reset      : sterge checkpoint v5\n"
                      << "       -step N     : pentru v3, testeaza doar pozitii multiple de N\n"
                      << "\n"
                      << "  Output: cheile gasite se salveaza in \"" << FOUND_FILE << "\"\n"
                      << "          (append mode — nu suprascrie chei anterioare)\n";
            return 0;
        }
        else positional.push_back(a);
    }

    if (positional.size() < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " [-v2|-v3|-v4|-v5] START:END TARGET_ADDR [BATCH]\n";
        return 1;
    }
    std::string range_str  = positional[0];
    std::string target_str = positional[1];
    if (positional.size() >= 3) {
        unsigned long v = std::strtoul(positional[2].c_str(), NULL, 10);
        if (v == 0 || v > (1UL << 24)) { std::cerr << "batch 1..16777216\n"; return 1; }
        batch_count = (unsigned int)v;
    }

    unsigned char target_h160[20];
    if (!decode_p2pkh_target(target_str, target_h160)) {
        std::cerr << "Invalid target address\n"; return 1;
    }
    unsigned char start_key[9], end_key[9];
    auto pos = range_str.find(':');
    if (pos == std::string::npos) { std::cerr << "range must be START:END\n"; return 1; }
    if (!parse_hex72(range_str.substr(0, pos), start_key) ||
        !parse_hex72(range_str.substr(pos + 1), end_key)) {
        std::cerr << "bad hex in range\n"; return 1;
    }
    Key128 start_u = key9_to_k128(start_key);
    Key128 end_u   = key9_to_k128(end_key);
    if (k128_is_zero(start_u)) { std::cerr << "start must be >= 1\n"; return 1; }
    if (!k128_ge(end_u, start_u)) { std::cerr << "start > end\n"; return 1; }
    Key128 range_size = k128_add_u64(k128_sub(end_u, start_u), 1ULL);

    std::signal(SIGINT, on_sigint);

    int dev = 0;
    cudaDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    die_cuda(cudaGetDevice(&dev), "getDevice");
    die_cuda(cudaGetDeviceProperties(&prop, dev), "getDeviceProps");

    // ============================================================
    // AUTO-TUNING GRID per kernel
    // ============================================================
    int max_blocks_per_sm_feistel = 1;
    int max_blocks_per_sm_random  = 1;

#define OCC_FEISTEL(K) cudaOccupancyMaxActiveBlocksPerMultiprocessor( \
    &max_blocks_per_sm_feistel, search_kernel_feistel_t<K>, BLOCK_SIZE, 0)
#define OCC_RANDOM(K)  cudaOccupancyMaxActiveBlocksPerMultiprocessor( \
    &max_blocks_per_sm_random,  search_kernel_random_t<K>,  BLOCK_SIZE, 0)

    switch (kpt) {
    case 1:  OCC_FEISTEL(1);  OCC_RANDOM(1);  break;
    case 2:  OCC_FEISTEL(2);  OCC_RANDOM(2);  break;
    case 4:  OCC_FEISTEL(4);  OCC_RANDOM(4);  break;
    case 8:  OCC_FEISTEL(8);  OCC_RANDOM(8);  break;
    case 16: OCC_FEISTEL(16); OCC_RANDOM(16); break;
    }
#undef OCC_FEISTEL
#undef OCC_RANDOM

    if (max_blocks_per_sm_feistel < 1) max_blocks_per_sm_feistel = 1;
    if (max_blocks_per_sm_random  < 1) max_blocks_per_sm_random  = 1;
    int max_blocks_per_sm_kpt = (mode == MODE_V3)
                              ? max_blocks_per_sm_feistel
                              : max_blocks_per_sm_random;

    unsigned int sm_count = (unsigned int)prop.multiProcessorCount;
    unsigned int auto_blocks_full = sm_count * (unsigned int)max_blocks_per_sm_kpt;
    unsigned int auto_waves = (mode == MODE_V4 || mode == MODE_V5) ? 32 : 16;
    unsigned int auto_batch = auto_blocks_full * BLOCK_SIZE * auto_waves;
    if (auto_batch < DEFAULT_BATCH) auto_batch = DEFAULT_BATCH;
    if (auto_batch > (1u << 24)) auto_batch = (1u << 24);

    if (batch_count == 0) batch_count = auto_batch;

    unsigned int blocks = (batch_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks_override > 0) blocks = blocks_override;
    if (gpux > 0 && gpuy > 0) {
        blocks = gpux * gpuy;
    }
    unsigned long long actual_ull =
        (unsigned long long)blocks * BLOCK_SIZE * (unsigned long long)kpt;

    if (!rkey_set) {
        if (mode == MODE_V4 || mode == MODE_V5) rkey = DEFAULT_RKEY;
        else rkey = 0;
    }

    // v5 checkpoint
    Key128 v5_cursor = {0, 0};
    bool v5_resumed = false;
    if (mode == MODE_V5 && !reset_ckpt) {
        std::string saved_range;
        if (load_checkpoint(v5_cursor, saved_range)) {
            if (saved_range == range_str) v5_resumed = true;
            else v5_cursor = {0, 0};
        }
    }
    if (mode == MODE_V5 && reset_ckpt) std::remove(CKPT_FILE);

    // ============================================================
    // BANNER
    // ============================================================
    std::cout << "========================================\n";
    std::cout << " Bitcoin GPU key generator (v2/v3/v4/v5)\n";
    std::cout << "   [tehnici Rotor-Cuda aplicate pt random]\n";
    std::cout << "========================================\n";

    if (mode == MODE_V2) {
        std::cout << "  Mod: v2  (Philox + cursor + sub-ferestre)\n";
    } else if (mode == MODE_V3) {
        std::cout << "  Mod: v3  (Feistel permutation)\n";
        if (step == 1)
            std::cout << "  -- acoperire 100%, fiecare cheie testata EXACT O DATA --\n";
        else
            std::cout << "  -- step = " << fmt_num(step) << " (acoperire partiala) --\n";
    } else if (mode == MODE_V4) {
        std::cout << "  Mod: v4  (Philox + random pur)\n";
        std::cout << "  -- chei aleatorii din tot intervalul, cu replacement --\n";
        std::cout << "  -- rkey = " << fmt_num(rkey) << " (refresh start points)\n";
    } else {
        std::cout << "  Mod: v5  (blocuri + random + checkpoint)\n";
        std::cout << "  -- block size    : " << fmt_num(v5_block) << " chei\n";
        std::cout << "  -- sample/block  : " << fmt_num(v5_sample) << " chei random\n";
        std::cout << "  -- rkey          : " << fmt_num(rkey) << "\n";
        std::cout << "  -- checkpoint    : " << CKPT_FILE;
        if (v5_resumed)      std::cout << "  [Reluare de la 0x" << k128_to_hex9(v5_cursor) << "]";
        else if (reset_ckpt) std::cout << "  [Reset — pornire de la 0]";
        else                 std::cout << "  [Pornire noua]";
        std::cout << "\n";
    }
    std::cout << "========================================\n";
    std::cout << "GPU         : " << prop.name << " CC " << prop.major << "." << prop.minor
              << "  (" << sm_count << " SM-uri)\n";
    std::cout << "Block size  : " << BLOCK_SIZE << " thread-uri/bloc\n";
    std::cout << "Grid size   : " << fmt_num(blocks) << " blocuri"
              << (blocks_override ? " [blocks override]" : "")
              << (gpux > 0 ? " [gpux override]" : "")
              << (!blocks_override && gpux == 0 ? " [auto]" : "")
              << "\n";
    std::cout << "Keys/iter   : " << fmt_num(actual_ull) << "\n";
    std::cout << "KPT         : " << kpt << " keys/thread\n";
    std::cout << "Occupancy   : feistel=" << max_blocks_per_sm_feistel
              << "/SM, random=" << max_blocks_per_sm_random
              << "/SM  -> folosit: " << max_blocks_per_sm_kpt << "/SM\n";
    std::cout << "Waves       : " << auto_waves << " (auto-batch)\n";
    std::cout << "I/O         : " << (quiet ? "quiet" : "normal") << "\n";
    if (mode == MODE_V3)
        std::cout << "CSPRNG      : Feistel permutation (" << FEISTEL_ROUNDS << " runde)\n";
    else
        std::cout << "CSPRNG      : Philox4x32-10 (counter-based)\n";
    std::cout << "RKEY        : " << (rkey ? fmt_num(rkey) : std::string("off")) << "\n";
    std::cout << "Scalar mul  : fixed-base comb 4-bit x 18 windows\n";
    std::cout << "Batch inv   : warp-parallel (shfl_sync)\n";
    std::cout << "Range       : " << range_str << "\n";
    std::cout << "Range size  : 0x" << k128_to_hex9(range_size) << " chei\n";
    std::cout << "Target addr : " << target_str << "\n";
    std::cout << "Output file : " << FOUND_FILE << "  (append mode)\n\n";

    build_comb_table_and_upload();
    run_gpu_test("607209F41C4BF4FB45", "14SLfcKMypXcWNGYQVw4JEXnee2CH6K2wM", quiet);

    unsigned int *d_match_count = NULL;
    unsigned long long *d_match_keys = NULL;
    unsigned char *d_target = NULL;
    die_cuda(cudaMalloc((void**)&d_match_count, sizeof(unsigned int)), "malloc count");
    die_cuda(cudaMalloc((void**)&d_match_keys,
                        2 * MAX_MATCHES * sizeof(unsigned long long)), "malloc keys");
    die_cuda(cudaMalloc((void**)&d_target, 20), "malloc target");
    die_cuda(cudaMemcpy(d_target, target_h160, 20, cudaMemcpyHostToDevice), "copy target");

    std::random_device rd;
    unsigned long long base_seed =
        ((unsigned long long)rd() << 32) ^ (unsigned long long)rd() ^
        (unsigned long long)std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();

    unsigned long long total = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto last_report = t0;
    double last_rate = 0.0;
    bool found_match = false;

    // v3 params
    Key128 Nm1;
    if (range_size.lo > 0) { Nm1.lo = range_size.lo - 1; Nm1.hi = range_size.hi; }
    else { Nm1.lo = ~0ULL; Nm1.hi = range_size.hi - 1; }
    int bits = bit_length_72((unsigned char)(Nm1.hi & 0xFFu), Nm1.lo);
    int bits_even = (bits + 1) & ~1;
    if (bits_even < 2) bits_even = 2;
    if (bits_even > 72) bits_even = 72;
    int half_bits = bits_even / 2;
    unsigned long long mask_half = (1ULL << half_bits) - 1ULL;
    unsigned long long master_key = splitmix64_host(base_seed ^ 0xA5A5A5A5A5A5A5A5ULL);
    unsigned int N_hi_u = (unsigned int)(range_size.hi & 0xFFu);
    unsigned long long N_lo_u = range_size.lo;

    // v4 mask
    unsigned char      full_mask_hi;
    unsigned long long full_mask_lo;
    mask_for_bits(bits, full_mask_hi, full_mask_lo);

    // v5 mask
    int bbits = bit_length_72(0, v5_block - 1);
    unsigned char      block_mask_hi;
    unsigned long long block_mask_lo;
    mask_for_bits(bbits, block_mask_hi, block_mask_lo);

    Key128 cursor = {0, 0};
    unsigned long long batch_counter = 0;
    unsigned long long iteration = 0;

    if (mode == MODE_V5 && v5_resumed) cursor = v5_cursor;

    while (!g_stop) {
        Key128 N_loop = range_size;
        if (mode == MODE_V4) {
            // skip range check
        } else if (k128_ge(cursor, N_loop)) {
            std::cout << "\n\033[2KRange epuizat.\n";
            break;
        }

        unsigned long long sub_size;
        if (mode == MODE_V4) sub_size = actual_ull;
        else if (mode == MODE_V5) sub_size = v5_block;
        else {
            Key128 remaining = k128_sub(N_loop, cursor);
            if (remaining.hi > 0 || remaining.lo >= actual_ull) sub_size = actual_ull;
            else sub_size = remaining.lo;
            if (sub_size == 0) break;
        }

        unsigned int zero = 0;
        die_cuda(cudaMemcpy(d_match_count, &zero, sizeof(unsigned int),
                            cudaMemcpyHostToDevice), "reset count");

        unsigned char b_start_hi;
        unsigned long long b_start_lo;
        unsigned char b_range_hi;
        unsigned long long b_range_lo;
        unsigned char b_mask_hi;
        unsigned long long b_mask_lo;
        unsigned long long bs;

        if (mode == MODE_V4) {
            bs = splitmix64_host(base_seed ^ (batch_counter++ * 0x9E3779B97F4A7C15ULL));
            b_start_hi = (unsigned char)(start_u.hi & 0xFFu); b_start_lo = start_u.lo;
            b_range_hi = (unsigned char)(range_size.hi & 0xFFu); b_range_lo = range_size.lo;
            b_mask_hi = full_mask_hi; b_mask_lo = full_mask_lo;
            switch (kpt) {
            case 1:  LAUNCH_RANDOM(1);  break;
            case 2:  LAUNCH_RANDOM(2);  break;
            case 4:  LAUNCH_RANDOM(4);  break;
            case 8:  LAUNCH_RANDOM(8);  break;
            case 16: LAUNCH_RANDOM(16); break;
            }
        } else if (mode == MODE_V3) {
            switch (kpt) {
            case 1:  LAUNCH_FEISTEL(1);  break;
            case 2:  LAUNCH_FEISTEL(2);  break;
            case 4:  LAUNCH_FEISTEL(4);  break;
            case 8:  LAUNCH_FEISTEL(8);  break;
            case 16: LAUNCH_FEISTEL(16); break;
            }
        } else if (mode == MODE_V5) {
            unsigned long long block_start_lo = start_u.lo + cursor.lo;
            unsigned char block_start_hi = (unsigned char)((start_u.hi & 0xFFu)
                                            + (cursor.hi & 0xFFu)
                                            + (block_start_lo < start_u.lo ? 1 : 0));
            unsigned long long tested_in_block = 0;
            while (tested_in_block < v5_sample && !g_stop) {
                bs = splitmix64_host(base_seed ^ (batch_counter++ * 0x9E3779B97F4A7C15ULL));
                b_start_hi = block_start_hi; b_start_lo = block_start_lo;
                b_range_hi = 0; b_range_lo = v5_block;
                b_mask_hi = block_mask_hi; b_mask_lo = block_mask_lo;
                switch (kpt) {
                case 1:  LAUNCH_RANDOM(1);  break;
                case 2:  LAUNCH_RANDOM(2);  break;
                case 4:  LAUNCH_RANDOM(4);  break;
                case 8:  LAUNCH_RANDOM(8);  break;
                case 16: LAUNCH_RANDOM(16); break;
                }
                die_cuda(cudaGetLastError(), "kernel launch");
                die_cuda(cudaDeviceSynchronize(), "kernel sync");
                tested_in_block += actual_ull;
                total += actual_ull;
                iteration += actual_ull;

                unsigned int nm = 0;
                die_cuda(cudaMemcpy(&nm, d_match_count, sizeof(unsigned int),
                                    cudaMemcpyDeviceToHost), "copy nmatches");
                if (nm > 0) {
                    if (nm > MAX_MATCHES) nm = MAX_MATCHES;
                    std::vector<unsigned long long> hk(2 * nm);
                    die_cuda(cudaMemcpy(hk.data(), d_match_keys,
                                        2 * nm * sizeof(unsigned long long),
                                        cudaMemcpyDeviceToHost), "copy keys");
                    for (unsigned int m = 0; m < nm; m++) {
                        Key128 mk; mk.hi = hk[2*m]; mk.lo = hk[2*m+1];
                        std::cout << "\n\n==================================================\n";
                        std::cout << " *** MATCH FOUND ***\n";
                        std::cout << "==================================================\n";
                        std::cout << " Private key (32B hex): " << k128_to_priv_hex64(mk) << "\n";
                        std::cout << " Private key (9B  hex): " << k128_to_hex9(mk) << "\n";
                        std::cout << " Address              : " << target_str << "\n";
                        std::cout << " Saved to             : " << FOUND_FILE << "\n";
                        std::cout << "==================================================\n";
                        // >>> SALVARE IN FISIER <<<
                        save_found_key(mk, target_str, range_str);
                    }
                    found_match = true;
                    g_stop = 1;
                    break;
                }
            }
            cursor = k128_add_u64(cursor, v5_block);
            save_checkpoint(cursor, range_str);
        } else {
            // v2
            unsigned long long b_start_lo_v2 = start_u.lo + cursor.lo;
            unsigned char b_start_hi_v2 = (unsigned char)((start_u.hi & 0xFFu)
                                        + (cursor.hi & 0xFFu)
                                        + (b_start_lo_v2 < start_u.lo ? 1 : 0));
            unsigned char      b_range_hi_v2 = 0;
            unsigned long long b_range_lo_v2 = sub_size;
            unsigned char      b_mask_hi_v2  = block_mask_hi;
            unsigned long long b_mask_lo_v2  = block_mask_lo;
            if (sub_size != actual_ull) {
                int nb = bit_length_72(0, sub_size - 1ULL);
                mask_for_bits(nb, b_mask_hi_v2, b_mask_lo_v2);
            }
            bs = splitmix64_host(base_seed ^ (batch_counter++ * 0x9E3779B97F4A7C15ULL));
            b_start_hi = b_start_hi_v2; b_start_lo = b_start_lo_v2;
            b_range_hi = b_range_hi_v2; b_range_lo = b_range_lo_v2;
            b_mask_hi = b_mask_hi_v2; b_mask_lo = b_mask_lo_v2;
            switch (kpt) {
            case 1:  LAUNCH_RANDOM(1);  break;
            case 2:  LAUNCH_RANDOM(2);  break;
            case 4:  LAUNCH_RANDOM(4);  break;
            case 8:  LAUNCH_RANDOM(8);  break;
            case 16: LAUNCH_RANDOM(16); break;
            }
        }

        if (mode != MODE_V5) {
            die_cuda(cudaGetLastError(), "kernel launch");
            die_cuda(cudaDeviceSynchronize(), "kernel sync");

            unsigned int nmatches = 0;
            die_cuda(cudaMemcpy(&nmatches, d_match_count, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost), "copy nmatches");
            total += actual_ull;
            iteration += actual_ull;

            if (nmatches > 0) {
                if (nmatches > MAX_MATCHES) nmatches = MAX_MATCHES;
                std::vector<unsigned long long> hkeys(2 * nmatches);
                die_cuda(cudaMemcpy(hkeys.data(), d_match_keys,
                                    2 * nmatches * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost), "copy keys");
                for (unsigned int m = 0; m < nmatches; m++) {
                    Key128 mk; mk.hi = hkeys[2*m]; mk.lo = hkeys[2*m+1];
                    std::cout << "\n\n==================================================\n";
                    std::cout << " *** MATCH FOUND ***\n";
                    std::cout << "==================================================\n";
                    std::cout << " Private key (32B hex): " << k128_to_priv_hex64(mk) << "\n";
                    std::cout << " Private key (9B  hex): " << k128_to_hex9(mk) << "\n";
                    std::cout << " Address              : " << target_str << "\n";
                    std::cout << " Saved to             : " << FOUND_FILE << "\n";
                    std::cout << "==================================================\n";
                    // >>> SALVARE IN FISIER <<<
                    save_found_key(mk, target_str, range_str);
                }
                found_match = true;
                g_stop = 1;
                break;
            }

            if (mode != MODE_V4) cursor = k128_add_u64(cursor, sub_size);
        }

        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - t0).count();
            double since = std::chrono::duration<double>(now - last_report).count();
            if (since >= 10.0) {
                last_rate = (dt > 0.0) ? (double)total / dt : 0.0;
                double pct = 0.0;
                if (range_size.lo > 0 || range_size.hi > 0) {
                    long double N_ld = (long double)range_size.hi * 18446744073709551616.0L
                                     + (long double)range_size.lo;
                    long double c_ld = (long double)cursor.hi * 18446744073709551616.0L
                                     + (long double)cursor.lo;
                    pct = (double)(c_ld * 100.0L / N_ld);
                }
                std::cout << "\r\033[2K"
                          << "Cursor: 0x" << k128_to_hex9(cursor)
                          << " (" << std::fixed << std::setprecision(4) << pct << "%)"
                          << " | " << std::setprecision(1) << last_rate << " keys/s"
                          << " | Total: " << fmt_num(total)
                          << " | Batches: " << fmt_num(batch_counter)
                          << std::flush;
                last_report = now;
            }
        }
    }

    if (mode == MODE_V5) save_checkpoint(cursor, range_str);

    cudaFree(d_target);
    cudaFree(d_match_keys);
    cudaFree(d_match_count);

    double tt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double avg = (tt > 0.0) ? (double)total / tt : 0.0;
    std::cout << "\r\033[2K"
              << (found_match ? "Search finished" : "Stopped")
              << " | Total: " << fmt_num(total)
              << " | Avg: " << std::fixed << std::setprecision(2)
              << avg << " keys/s\n";
    if (found_match) {
        std::cout << "Cheile gasite au fost salvate in: " << FOUND_FILE << "\n";
    }
    return 0;
}
