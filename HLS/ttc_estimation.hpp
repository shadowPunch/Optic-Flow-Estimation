#pragma once

#include "hls_stream.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_sobel.hpp"
#include "core/xf_arithm.hpp"
#include "core/xf_magnitude.hpp"
#include "core/xf_reduce.hpp"

// Use the same constants defined in the ego-motion module
#include "ego_motion.hpp" 

// --- Configuration ---
const int MAX_BBOXES = 20; // Max objects to calculate TTC for in one go

// A struct to hold bounding box data from the PS
struct Bbox_t {
    ap_uint<10> x1, y1, x2, y2; // Use fixed-size integers for HLS
};

// --- Top-Level HLS Function ---
void ttc_estimator_hls(
    xf::cv::Mat<XF_32FC2, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1>& corrected_flow,
    Bbox_t bboxes[MAX_BBOXES],
    float ttc_results[MAX_BBOXES],
    int num_boxes) {
#pragma HLS INTERFACE s_axilite port=bboxes bundle=control
#pragma HLS INTERFACE s_axilite port=ttc_results bundle=control
#pragma HLS INTERFACE s_axilite port=num_boxes bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW

    // Internal mats for processing
    xf::cv::Mat<XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> flow_u(FRAME_HEIGHT, FRAME_WIDTH);
    xf::cv::Mat<XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> flow_v(FRAME_HEIGHT, FRAME_WIDTH);
    xf::cv::Mat<XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> divergence(FRAME_HEIGHT, FRAME_WIDTH);
    
    // --- Step 1: Compute Divergence of the flow field ---
    // Split the 2-channel flow into separate u and v planes
    xf::cv::split<XF_32FC2, XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1>(corrected_flow, flow_u, flow_v);

    // Calculate derivatives: du/dx and dv/dy
    xf::cv::Mat<XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> du_dx(FRAME_HEIGHT, FRAME_WIDTH);
    xf::cv::Mat<XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> dv_dy(FRAME_HEIGHT, FRAME_WIDTH);
    
    // Using a 3x3 Sobel filter
    xf::cv::Sobel<XF_BORDER_REPLICATE, 3, 32, 32>(flow_u, du_dx, 1, 0);
    xf::cv::Sobel<XF_BORDER_REPLICATE, 3, 32, 32>(flow_v, dv_dy, 0, 1);

    // divergence = du/dx + dv/dy
    xf::cv::add<XF_CONVERT_POLICY_SATURATE, XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1>(du_dx, dv_dy, divergence);


    // --- Step 2: Loop through bounding boxes and calculate TTC for each ---
BBOX_LOOP:
    for (int i = 0; i < num_boxes; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_BBOXES
#pragma HLS PIPELINE

        Bbox_t bbox = bboxes[i];
        xf::cv::Rect roi(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

        if (roi.width <= 0 || roi.height <= 0) {
            ttc_results[i] = -1.0f; // Invalid TTC
            continue;
        }

        // --- Method 1: Divergence-based TTC ---
        // Get the ROI from the divergence map
        xf::cv::Mat<XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> div_roi = divergence(roi);

        // Calculate the mean divergence in the ROI
        float mean_div_val[1];
        xf::cv::reduce<XF_REDUCE_AVG, XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1>(div_roi, mean_div_val);
        
        float ttc_div = -1.0f;
        if (mean_div_val[0] > 1e-5) {
            ttc_div = 1.0f / (mean_div_val[0] * 30.0f); // Assuming 30 FPS
        }

        // --- Method 2: Flow Magnitude-based TTC ---
        xf::cv::Mat<XF_32FC2, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> flow_roi = corrected_flow(roi);
        xf::cv::Mat<XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1> flow_mag_roi(roi.height, roi.width);

        // Calculate magnitude of flow vectors in the ROI
        xf::cv::magnitude<XF_L2NORM, XF_32FC2, XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1>(flow_roi, flow_mag_roi);

        // Calculate mean flow magnitude
        float mean_flow_mag_val[1];
        xf::cv::reduce<XF_REDUCE_AVG, XF_32FC1, FRAME_HEIGHT, FRAME_WIDTH, XF_NPPC1>(flow_mag_roi, mean_flow_mag_val);
        
        float ttc_flow = -1.0f;
        if (mean_flow_mag_val[0] > 0.1f) {
            float obj_size_px = (roi.width > roi.height) ? roi.width : roi.height;
            // Constants from Python code
            const float FOCAL_LENGTH = 500.0f;
            const float OBJECT_SIZE_M = 1.5f;

            float estimated_dist = (OBJECT_SIZE_M * FOCAL_LENGTH) / obj_size_px;
            float velocity_m_per_sec = (mean_flow_mag_val[0] * estimated_dist * 30.0f) / FOCAL_LENGTH;

            if (velocity_m_per_sec > 0.1f) {
                ttc_flow = estimated_dist / velocity_m_per_sec;
            }
        }

        // --- Combine TTCs (simple average if both are valid) ---
        if (ttc_div > 0 && ttc_flow > 0) {
            ttc_results[i] = (ttc_div * 3.0f + ttc_flow * 2.0f) / 5.0f; // Weighted average from python code
        } else if (ttc_div > 0) {
            ttc_results[i] = ttc_div;
        } else if (ttc_flow > 0) {
            ttc_results[i] = ttc_flow;
        } else {
            ttc_results[i] = -1.0f; // No valid TTC found
        }
    }
}
