/*
 *
 *  Copyright (C) 2023, SToFU Systems S.L.
 *  All rights reserved.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
#include "sha512.cuh"

namespace hashes
{
    __device__ static const uint64_t K[80] = {
   UINT64_C(0x428a2f98d728ae22), UINT64_C(0x7137449123ef65cd),
   UINT64_C(0xb5c0fbcfec4d3b2f), UINT64_C(0xe9b5dba58189dbbc),
   UINT64_C(0x3956c25bf348b538), UINT64_C(0x59f111f1b605d019),
   UINT64_C(0x923f82a4af194f9b), UINT64_C(0xab1c5ed5da6d8118),
   UINT64_C(0xd807aa98a3030242), UINT64_C(0x12835b0145706fbe),
   UINT64_C(0x243185be4ee4b28c), UINT64_C(0x550c7dc3d5ffb4e2),
   UINT64_C(0x72be5d74f27b896f), UINT64_C(0x80deb1fe3b1696b1),
   UINT64_C(0x9bdc06a725c71235), UINT64_C(0xc19bf174cf692694),
   UINT64_C(0xe49b69c19ef14ad2), UINT64_C(0xefbe4786384f25e3),
   UINT64_C(0x0fc19dc68b8cd5b5), UINT64_C(0x240ca1cc77ac9c65),
   UINT64_C(0x2de92c6f592b0275), UINT64_C(0x4a7484aa6ea6e483),
   UINT64_C(0x5cb0a9dcbd41fbd4), UINT64_C(0x76f988da831153b5),
   UINT64_C(0x983e5152ee66dfab), UINT64_C(0xa831c66d2db43210),
   UINT64_C(0xb00327c898fb213f), UINT64_C(0xbf597fc7beef0ee4),
   UINT64_C(0xc6e00bf33da88fc2), UINT64_C(0xd5a79147930aa725),
   UINT64_C(0x06ca6351e003826f), UINT64_C(0x142929670a0e6e70),
   UINT64_C(0x27b70a8546d22ffc), UINT64_C(0x2e1b21385c26c926),
   UINT64_C(0x4d2c6dfc5ac42aed), UINT64_C(0x53380d139d95b3df),
   UINT64_C(0x650a73548baf63de), UINT64_C(0x766a0abb3c77b2a8),
   UINT64_C(0x81c2c92e47edaee6), UINT64_C(0x92722c851482353b),
   UINT64_C(0xa2bfe8a14cf10364), UINT64_C(0xa81a664bbc423001),
   UINT64_C(0xc24b8b70d0f89791), UINT64_C(0xc76c51a30654be30),
   UINT64_C(0xd192e819d6ef5218), UINT64_C(0xd69906245565a910),
   UINT64_C(0xf40e35855771202a), UINT64_C(0x106aa07032bbd1b8),
   UINT64_C(0x19a4c116b8d2d0c8), UINT64_C(0x1e376c085141ab53),
   UINT64_C(0x2748774cdf8eeb99), UINT64_C(0x34b0bcb5e19b48a8),
   UINT64_C(0x391c0cb3c5c95a63), UINT64_C(0x4ed8aa4ae3418acb),
   UINT64_C(0x5b9cca4f7763e373), UINT64_C(0x682e6ff3d6b2b8a3),
   UINT64_C(0x748f82ee5defb2fc), UINT64_C(0x78a5636f43172f60),
   UINT64_C(0x84c87814a1f0ab72), UINT64_C(0x8cc702081a6439ec),
   UINT64_C(0x90befffa23631e28), UINT64_C(0xa4506cebde82bde9),
   UINT64_C(0xbef9a3f7b2c67915), UINT64_C(0xc67178f2e372532b),
   UINT64_C(0xca273eceea26619c), UINT64_C(0xd186b8c721c0c207),
   UINT64_C(0xeada7dd6cde0eb1e), UINT64_C(0xf57d4f7fee6ed178),
   UINT64_C(0x06f067aa72176fba), UINT64_C(0x0a637dc5a2c898a6),
   UINT64_C(0x113f9804bef90dae), UINT64_C(0x1b710b35131c471b),
   UINT64_C(0x28db77f523047d84), UINT64_C(0x32caab7b40c72493),
   UINT64_C(0x3c9ebe0a15c9bebc), UINT64_C(0x431d67c49c100d4c),
   UINT64_C(0x4cc5d4becb3e42b6), UINT64_C(0x597f299cfc657e2a),
   UINT64_C(0x5fcb6fab3ad6faec), UINT64_C(0x6c44198c4a475817)
    };

    /* Various logical functions for calculating sha-512 hash on GPU */

#define ROR64c(x, y) \
    ( ((((x)&UINT64_C(0xFFFFFFFFFFFFFFFF))>>((uint64_t)(y)&UINT64_C(63))) | \
      ((x)<<((uint64_t)(64-((y)&UINT64_C(63)))))) & UINT64_C(0xFFFFFFFFFFFFFFFF))

#define STORE64H(x, y)                                                                     \
   { (y)[0] = (unsigned char)(((x)>>56)&255); (y)[1] = (unsigned char)(((x)>>48)&255);     \
     (y)[2] = (unsigned char)(((x)>>40)&255); (y)[3] = (unsigned char)(((x)>>32)&255);     \
     (y)[4] = (unsigned char)(((x)>>24)&255); (y)[5] = (unsigned char)(((x)>>16)&255);     \
     (y)[6] = (unsigned char)(((x)>>8)&255); (y)[7] = (unsigned char)((x)&255); }

#define LOAD64H(x, y)                                                      \
   { x = (((uint64_t)((y)[0] & 255))<<56)|(((uint64_t)((y)[1] & 255))<<48) | \
         (((uint64_t)((y)[2] & 255))<<40)|(((uint64_t)((y)[3] & 255))<<32) | \
         (((uint64_t)((y)[4] & 255))<<24)|(((uint64_t)((y)[5] & 255))<<16) | \
         (((uint64_t)((y)[6] & 255))<<8)|(((uint64_t)((y)[7] & 255))); }


#define Ch(x,y,z)       (z ^ (x & (y ^ z)))
#define Maj(x,y,z)      (((x | y) & z) | (x & y))
#define S(x, n)         ROR64c(x, n)
#define R(x, n)         (((x) &UINT64_C(0xFFFFFFFFFFFFFFFF))>>((uint64_t)n))
#define Sigma0(x)       (S(x, 28) ^ S(x, 34) ^ S(x, 39))
#define Sigma1(x)       (S(x, 14) ^ S(x, 18) ^ S(x, 41))
#define Gamma0(x)       (S(x, 1) ^ S(x, 8) ^ R(x, 7))
#define Gamma1(x)       (S(x, 19) ^ S(x, 61) ^ R(x, 6))
#ifndef MIN
#define MIN(x, y) ( ((x)<(y))?(x):(y) )
#endif

    /*
    * FUNCTION: static int __device__ __host__ sha512_compress
    *
    * ARGS:
    * sha512_context* md - Pointer to the SHA-512 context structure.
    * unsigned char* buf - Pointer to the buffer containing the data to be compressed.
    *
    * DESCRIPTION:
    * This function performs the compression step of the SHA-512 algorithm on a block of data.
    * It performs the following steps:
    * - Copies the current state values from the SHA-512 context (md) into local variables (S).
    * - Copies the input data block (buf) into an array of 80 64-bit unsigned integers (W).
    * - Fills the remaining elements of W[16..79] using bitwise operations and additions as per the SHA-512 algorithm.
    * - Performs a series of 80 rounds of SHA-512 operations (RND macro) on the state variables (S) and elements of W.
    * - Updates the state variables (md->state) by adding the values from the local variables (S).
    * This function is marked as static, which means it can only be accessed within the same source file. It can be called from both device (GPU) and host (CPU) code, as denoted by the __device__ and __host__ qualifiers.
    *
    * RETURN VALUE: int
    * Returns 0 on success, and a non-zero value if any error occurs (currently not used in the function).
    */
    static int __device__ __host__ sha512_compress(sha512_context* md, unsigned char* buf)
    {
        uint64_t S[8], W[80], t0, t1;
        int i;

        /* copy state into S */
        for (i = 0; i < 8; i++)
            S[i] = md->state[i];
        /* copy the state into 1024-bits into W[0..15] */
        for (i = 0; i < 16; i++)
            LOAD64H(W[i], buf + (8 * i));
        /* fill W[16..79] */
        for (i = 16; i < 80; i++)
            W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];

        /* Compress */
#define RND(a,b,c,d,e,f,g,h,i) \
    t0 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]; \
    t1 = Sigma0(a) + Maj(a, b, c);\
    d += t0; \
    h  = t0 + t1;

        for (i = 0; i < 80; i += 8) {
            RND(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], i + 0);
            RND(S[7], S[0], S[1], S[2], S[3], S[4], S[5], S[6], i + 1);
            RND(S[6], S[7], S[0], S[1], S[2], S[3], S[4], S[5], i + 2);
            RND(S[5], S[6], S[7], S[0], S[1], S[2], S[3], S[4], i + 3);
            RND(S[4], S[5], S[6], S[7], S[0], S[1], S[2], S[3], i + 4);
            RND(S[3], S[4], S[5], S[6], S[7], S[0], S[1], S[2], i + 5);
            RND(S[2], S[3], S[4], S[5], S[6], S[7], S[0], S[1], i + 6);
            RND(S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[0], i + 7);
        }
#undef RND
        for (i = 0; i < 8; i++)
            md->state[i] = md->state[i] + S[i];

        return 0;
    }

    /*
    * FUNCTION: int __device__ __host__ sha512_init
    *
    * ARGS:
    * sha512_context* md - Pointer to the SHA-512 context structure.
    *
    * DESCRIPTION:
    * This function initializes the SHA-512 context by setting the initial state values for the SHA-512 hash calculation.
    * It performs the following steps:
    * - Checks for a NULL pointer for the input SHA-512 context, which is an error condition.
    * - Sets the buffer length (curlen) and original message length (length) in the context to 0.
    * - Sets the initial state values (8 64-bit unsigned integers) in the context as per the SHA-512 algorithm specifications.
    * This function can be called from both device (GPU) and host (CPU) code, as denoted by the __device__ and __host__ qualifiers.
    *
    * RETURN VALUE: int
    * Returns 0 on success, and a non-zero value if any error occurs (e.g., NULL pointer for the input context).
    */
    int __device__ __host__ sha512_init(sha512_context* md)
    {
        if (md == NULL) return 1;
        md->curlen = 0;
        md->length = 0;
        md->state[0] = UINT64_C(0x6a09e667f3bcc908);
        md->state[1] = UINT64_C(0xbb67ae8584caa73b);
        md->state[2] = UINT64_C(0x3c6ef372fe94f82b);
        md->state[3] = UINT64_C(0xa54ff53a5f1d36f1);
        md->state[4] = UINT64_C(0x510e527fade682d1);
        md->state[5] = UINT64_C(0x9b05688c2b3e6c1f);
        md->state[6] = UINT64_C(0x1f83d9abfb41bd6b);
        md->state[7] = UINT64_C(0x5be0cd19137e2179);

        return 0;
    }

    /*
    * FUNCTION: int __device__ __host__ sha512_update
    *
    * ARGS:
    * sha512_context* md - Pointer to the SHA-512 context structure.
    * const uint8_t* in - Pointer to the input message buffer.
    * size_t inlen - Length of the input message buffer.
    *
    * DESCRIPTION:
    * This function updates the SHA-512 hash calculation with additional input data. It processes the input data in blocks of 128 bytes and updates the SHA-512 context accordingly.
    * It performs the following steps:
    * - Checks for NULL pointers for the input SHA-512 context and input message buffer.
    * - Checks if the current length of the message buffer in the context is greater than the size of the buffer, which is an error condition.
    * - Processes the input data in blocks of 128 bytes:
    * - If the current length of the message buffer in the context is 0 and the input data length is greater than or equal to 128 bytes, it directly compresses the input data using sha512_compress() function, updates the length of the original message, and advances the input data buffer and length.
    * - Otherwise, it copies the input data to the message buffer in the context until the buffer is full (128 bytes):
    * - If the buffer is full, it compresses the buffer using sha512_compress() function, updates the length of the original message, and resets the buffer length.
    * - Continues this process until all the input data is processed.
    * This function can be called from both device (GPU) and host (CPU) code, as denoted by the __device__ and __host__ qualifiers.
    *
    * RETURN VALUE: int
    * Returns 0 on success, and a non-zero value if any error occurs.
    */
    int __device__ __host__ sha512_update(sha512_context* md, const uint8_t* in, size_t inlen)
    {
        size_t n;
        int  err;

        /* Check if input parameters are valid */
        if (md == NULL) return 1;
        if (in == NULL) return 1;
        if (md->curlen > sizeof(md->buf)) return 1;

        /* Process input data in blocks of HASH_SIZE bytes */
        while (inlen > 0)
        {
            /* If there is enough input data and buffer is empty, directly compress the input data */
            if (md->curlen == 0 && inlen >= HASH_SIZE)
            {
                if ((err = sha512_compress(md, (unsigned char*)in)) != 0) return err;

                md->length += HASH_SIZE * 8;
                in += HASH_SIZE;
                inlen -= HASH_SIZE;
            }
            else
            {
                /* Copy input data to buffer until it is full or input data is exhausted */
                n = MIN(inlen, (HASH_SIZE - md->curlen));
                for (size_t i = 0; i < n; ++i)
                    md->buf[i + md->curlen] = in[i];

                md->curlen += n;
                in += n;
                inlen -= n;

                /* If buffer is full, compress it */
                if (md->curlen == HASH_SIZE) {
                    if ((err = sha512_compress(md, md->buf)) != 0) return err;

                    md->length += 8 * HASH_SIZE;
                    md->curlen = 0;
                }
            }
        }
        return 0;
    }

    /*
    * FUNCTION: int __device__ __host__ sha512_final
    *
    * ARGS:
    * sha512_context* md - Pointer to the SHA-512 context structure.
    * uint8_t* out - Pointer to the output buffer for storing the final SHA-512 hash.
    *
    * DESCRIPTION:
    * This function finalizes the SHA-512 hash calculation by padding the input message and storing the calculated hash in the output buffer.
    * It performs the following steps:
    * - Checks for NULL pointers for the input SHA-512 context and output buffer.
    * - Appends the '1' bit to the message buffer.
    * - If the length of the message buffer is greater than 112 bytes, it appends zeros and compresses the buffer.
    * - Appends zeros to the message buffer until it reaches a length of 120 bytes.
    * - Stores the length of the original message in big-endian format in the last 8 bytes of the buffer.
    * - Performs the final compression using sha512_compress() function.
    * - Copies the resulting hash from the SHA-512 context to the output buffer.
    *  This function can be called from both device (GPU) and host (CPU) code, as denoted by the __device__ and __host__ qualifiers.
    *
    * RETURN VALUE: int
    * Returns 0 on success, and a non-zero value if any error occurs.
    */
    int __device__ __host__ sha512_final(sha512_context* md, uint8_t* out)
    {
        /* Check if input parameters are valid */
        if (md == NULL) return 1;
        if (out == NULL) return 1;
        if (md->curlen >= sizeof(md->buf)) return 1;

        /* increase the length of the message */
        md->length += md->curlen * UINT64_C(8);
        /* append the '1' bit */
        md->buf[md->curlen++] = (unsigned char)0x80;

        /* if the length is currently above 112 bytes append zeros then compress. Then can fall back to padding zeros and length encoding like normal */
        if (md->curlen > 112) {
            while (md->curlen < HASH_SIZE)
                md->buf[md->curlen++] = (unsigned char)0;

            sha512_compress(md, md->buf);
            md->curlen = 0;
        }

        while (md->curlen < 120)
            md->buf[md->curlen++] = (unsigned char)0;

        /* store length */
        STORE64H(md->length, md->buf + 120);
        sha512_compress(md, md->buf);
        /* copy output */
        for (int i = 0; i < 8; i++)
            STORE64H(md->state[i], out + (8 * i));

        return 0;
    }

    /*
    * FUNCTION: int __device__ __host__ sha512
    *
    * ARGS:
    * const uint8_t* message - Pointer to the input message whose SHA-512 hash needs to be calculated.
    * size_t length - Length of the input message.
    * uint8_t* out - Pointer to the output buffer for storing the calculated SHA-512 hash.
    *
    * DESCRIPTION:
    * This function calculates the SHA-512 hash for the input message using the sha512_context structure and associated functions.
    * It initializes the sha512_context using sha512_init() function, updates the context with the input message using sha512_update() function, and finalizes the context to obtain the SHA-512 hash using sha512_final() function.
    * The calculated hash is stored in the output buffer pointed to by 'out'.
    * This function can be called from both device (GPU) and host (CPU) code, as denoted by the __device__ and __host__ qualifiers.
    *
    * RETURN VALUE: int
    * Returns the status of the SHA-512 calculation, where 0 indicates success, and any other value indicates an error.
    */
    int __device__ __host__ sha512(const uint8_t* message, size_t length, uint8_t* out)
    {
        sha512_context ctx;
        int status;
        if ((status = sha512_init(&ctx))) return status;
        if ((status = sha512_update(&ctx, message, length))) return status;
        if ((status = sha512_final(&ctx, out))) return status;
        return status;
    }

    /*
    * FUNCTION: std::string __host__ sha512
    *
    * ARGS:
    * const uint8_t* message - Pointer to the input message whose SHA-512 hash needs to be calculated.
    * size_t length - Length of the input data.
    *
    * DESCRIPTION:
    * This function calculates the SHA-512 hash of the input data using a GPU-based implementation.
    * It performs the following steps:
    * - Initializes a SHA-512 context structure (ctx) from the sha512GPU namespace.
    * - Updates the context with the input data using sha512GPU::sha512_update() function.
    * - Finalizes the hash calculation and stores the resulting digest in a local buffer (digest) using sha512GPU::sha512_final() function.
    * - Converts the digest from binary to hexadecimal representation and stores it in a string buffer (buf) using sprintf() function.
    * - Returns the calculated SHA-512 hash as a string.
    * This function is marked with __host__ qualifier, which means it can be called from host (CPU) code, but not from device (GPU) code.
    *
    * RETURN VALUE: std::vector<uint8_t>
    * Returns the calculated std::vector<uint8_t> as a hexadecimal bytes.
    */
    std::vector<uint8_t> __host__ sha512(const uint8_t* message, size_t length)
    {
        std::vector<uint8_t> digest(DIGEST_SIZE);
        hashes::sha512_context ctx;
        int status;
        if ((status = hashes::sha512_init(&ctx))) return digest;
        if ((status = hashes::sha512_update(&ctx, message, length))) return digest;
        if ((status = hashes::sha512_final(&ctx, digest.data()))) return digest;
        return digest;
    }

    /*
    * FUNCTION: void __global__ sha512Kernel
    *
    * ARGS:
    * char* inputs - Pointer to the input buffers in GPU memory.
    * int numInputs - Number of input buffers to process.
    * uint8_t* outputs - Pointer to the output buffer in GPU memory for storing the calculated SHA-512 hashes.
    * size_t bufferSize - Size of each input buffer.
    * int bufferLength - Length of each input buffer.
    * This function is meant to be called from host code and executed on the GPU using CUDA.
    *
    * DESCRIPTION:
    * This CUDA kernel function is launched on the GPU to calculate the SHA-512 hashes for the input buffers in parallel.
    * It calculates the global thread ID using blockIdx.x and threadIdx.x, and checks if the thread ID is within bounds of the number of input buffers.
    * If the thread ID is within bounds, it calls the sha512() function to calculate the SHA-512 hash for the corresponding input buffer, and stores the result in the output buffer in GPU memory.
    */
    void __global__ sha512Kernel(char* inputs, int numInputs, uint8_t* outputs, int bufferLength)
    {
        /* Calculate global thread ID */
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        /* Check if thread ID is within bounds and call SHA-512 function */
        if (index < numInputs)
            sha512((uint8_t*)(inputs + index * bufferLength), bufferLength, outputs + index * DIGEST_SIZE);
    }
}