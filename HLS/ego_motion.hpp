#pragma once

#include "hls_stream.h"
#include "ap_int.h"

// Include Vitis Vision headers for the functions we will use
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_warp.hpp"
#include "imgproc/xf_absdiff.hpp"
#include "core/xf_mean.hpp"

// --- Template Parameters for Consistency ---
// These should be defined in a central configuration header
// in a full project to ensure all modules use the same specs.
const int FRAME_HEIGHT = 384;
const int FRAME_WIDTH = 512;
const int PIXEL_TYPE = XF_8UC1; // Grayscale 8-bit
const int PIXEL_DEPTH = XF_8UP;

// --- Algorithm Parameters for HLS ---
// Reduced for a more efficient hardware implementation
const int NUM_CANDIDATES = 8;  // Number of parallel candidates to test
const int MAX_ITERATIONS = 10; // Number of refinement iterations

// A simple hardware-friendly pseudo-random number generator (LFSR)
// to replace np.random in a synthesizable way.
unsigned int lfsr_prng(unsigned int* seed) {
    unsigned int lfsr = *seed;
    // 16-bit LFSR with polynomial x^16 + x^14 + x^13 + x^11 + 1
    unsigned int bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
    lfsr = (lfsr >> 1) | (bit << 15);
    *seed = lfsr;
    return lfsr;
}


/**
 * @brief Top-level HLS function for ego-motion estimation.
 * * This module takes two consecutive grayscale frames and estimates the
 * camera's motion (translation dx, dy and rotation theta) between them.
 * * @param prev_frame_strm   Input stream for the previous grayscale frame.
 * @param curr_frame_strm   Input stream for the current grayscale frame.
 * @param dx_out              Pointer to store the output estimated translation in x.
 * @param dy_out              Pointer to store the output estimated translation in y.
 * @param theta_out           Pointer to store the output estimated rotation in radians.
 */
void ego_motion_hls(hls::stream<ap_axiu<8, 0, 0, 0>>& prev_frame_strm,
                    hls::stream<ap_axiu<8, 0, 0, 0>>& curr_frame_strm,
                    float* dx_out,
                    float* dy_out,
                    float* theta_out) {
// --- HLS Pragmas for Interface Definition ---
// This defines how the module connects to the outside world.
// Images arrive via AXI-Stream, results are accessed via AXI-Lite.
#pragma HLS INTERFACE axis port=prev_frame_strm
#pragma HLS INTERFACE axis port=curr_frame_strm
#pragma HLS INTERFACE s_axilite port=dx_out bundle=control
#pragma HLS INTERFACE s_axilite port=dy_out bundle=control
#pragma HLS INTERFACE s_axilite port=theta_out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Internal image buffers
    xf::cv::Mat<PIXEL_TYPE, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> prev_frame_img(FRAME_HEIGHT, FRAME_WIDTH);
    xf::cv::Mat<PIXEL_TYPE, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> curr_frame_img(FRAME_HEIGHT, FRAME_WIDTH);
    xf::cv::Mat<PIXEL_TYPE, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> transformed_img(FRAME_HEIGHT, FRAME_WIDTH);
    xf::cv::Mat<PIXEL_TYPE, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> diff_img(FRAME_HEIGHT, FRAME_WIDTH);

// Stream the input data into internal image matrices
#pragma HLS DATAFLOW
    xf::cv::AXIvideo2Mat(prev_frame_strm, prev_frame_img);
    xf::cv::AXIvideo2Mat(curr_frame_strm, curr_frame_img);

    // --- State for the Genetic Algorithm ---
    float candidates[NUM_CANDIDATES][3]; // Each row is {dx, dy, theta}
    float best_params[3] = {0.0f, 0.0f, 0.0f};
    float min_mse = 99999.0f;
    unsigned int seed = 12345; // Initial seed for PRNG

// --- Initialize First Generation of Candidates ---
// Create a diverse set of initial guesses.
INIT_CANDIDATES_LOOP:
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
#pragma HLS UNROLL
        // A more structured initialization than pure random
        candidates[i][0] = (i % 2 == 0) ? -1.5f : 1.5f; // dx
        candidates[i][1] = (i % 4 < 2) ? -1.5f : 1.5f; // dy
        candidates[i][2] = (i < 4) ? -0.02f : 0.02f; // theta
    }

// --- Main Iterative Refinement Loop ---
ITERATION_LOOP:
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        float mses[NUM_CANDIDATES];

    // --- Evaluate All Candidates in Parallel ---
    CANDIDATE_EVAL_LOOP:
        for (int i = 0; i < NUM_CANDIDATES; ++i) {
#pragma HLS PIPELINE II=1
            float M[6];
            float dx = candidates[i][0];
            float dy = candidates[i][1];
            float theta = candidates[i][2];

            // 1. Build the transformation matrix for warpAffine
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);
            M[0] = cos_theta;
            M[1] = -sin_theta;
            M[2] = dx;
            M[3] = sin_theta;
            M[4] = cos_theta;
            M[5] = dy;

            // 2. Apply transformation (HW accelerated)
            xf::cv::warpAffine<XF_INTERPOLATION_BILINEAR, XF_BORDER_CONSTANT>(curr_frame_img, transformed_img, M, 0);

            // 3. Calculate difference (HW accelerated)
            xf::cv::absdiff(prev_frame_img, transformed_img, diff_img);
            
            // 4. Calculate Mean Squared Error (HW accelerated)
            ap_uint<64> sum_of_squares = 0; // Use a wide type for accumulation
            xf::cv::mean(diff_img, &sum_of_squares); // This gives sum of pixels, need to square first.
                                                    // For a true MSE, you'd need a custom pixel-wise loop
                                                    // or a dedicated squaring function before mean.
                                                    // Here, we use a simpler Mean Absolute Error as a proxy.
            mses[i] = (float)sum_of_squares / (FRAME_WIDTH * FRAME_HEIGHT);
        }

        // --- Find Best Candidate and Update ---
        // This part is sequential but very fast as it operates on a small array.
        int best_idx = 0;
    FIND_BEST_LOOP:
        for (int i = 1; i < NUM_CANDIDATES; ++i) {
            if (mses[i] < mses[best_idx]) {
                best_idx = i;
            }
        }
        
        if (mses[best_idx] < min_mse) {
            min_mse = mses[best_idx];
            best_params[0] = candidates[best_idx][0];
            best_params[1] = candidates[best_idx][1];
            best_params[2] = candidates[best_idx][2];
        }

        // --- Generate Next Generation (Mutation) ---
        // Create new candidates by slightly changing the best one.
        float mutation_std = 0.5f / (iter + 1); // Decrease mutation over time
    MUTATION_LOOP:
        for (int i = 0; i < NUM_CANDIDATES; ++i) {
#pragma HLS UNROLL
            // Center new candidates around the current best
            float rand1 = (float)(lfsr_prng(&seed) % 100 - 50) / 50.0f; // range [-1, 1]
            float rand2 = (float)(lfsr_prng(&seed) % 100 - 50) / 50.0f;
            float rand3 = (float)(lfsr_prng(&seed) % 100 - 50) / 50.0f;
            
            candidates[i][0] = best_params[0] + mutation_std * rand1;
            candidates[i][1] = best_params[1] + mutation_std * rand2;
            candidates[i][2] = best_params[2] + (mutation_std/10.0f) * rand3; // Smaller mutation for theta
        }
    }

    // --- Output the best parameters found ---
    *dx_out = best_params[0];
    *dy_out = best_params[1];
    *theta_out = best_params[2];
}
