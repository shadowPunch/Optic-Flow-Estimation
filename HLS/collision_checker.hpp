#pragma once

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

// A struct to hold the final, combined data for a single object
// This would be assembled by the host application from the tracker and TTC accelerator outputs.
struct FinalTrackData {
    int id;
    cv::Rect bbox;
    std::string class_name;
    float ttc; // The final TTC value calculated by the HLS module
};

// A struct to hold information about a specific collision warning
struct CollisionWarning {
    int id;
    cv::Rect bbox;
    std::string class_name;
    float ttc;
    bool is_in_critical_roi;
};


class CollisionChecker {
public:
    // Constructor to initialize the ROI and thresholds
    CollisionChecker() {
        // These values are taken directly from your Python script's configuration
        
        // --- ROI CONFIG ---
        roi.x = 140;
        roi.y = 100;
        roi.width = 360 - 140;
        roi.height = 300 - 100;

        // --- TTC Thresholds ---
        ttc_threshold_inside_roi = 1.12f;
        ttc_threshold_outside_roi = 0.56f;
    }

    /**
     * @brief Checks a list of tracked objects for potential collisions.
     * @param tracks_with_ttc A vector of final track data including their TTC values.
     * @return A vector of CollisionWarning structs for objects that pose a risk.
     */
    std::vector<CollisionWarning> check(const std::vector<FinalTrackData>& tracks_with_ttc) {
        std::vector<CollisionWarning> warnings;

        // Loop through each tracked object
        for (const auto& track : tracks_with_ttc) {
            // A negative TTC value indicates it was invalid
            if (track.ttc <= 0) {
                continue;
            }

            bool is_critical = is_object_in_roi(track.bbox);
            float threshold = is_critical ? ttc_threshold_inside_roi : ttc_threshold_outside_roi;

            // --- The Core Collision Check ---
            if (track.ttc <= threshold) {
                warnings.push_back({
                    track.id,
                    track.bbox,
                    track.class_name,
                    track.ttc,
                    is_critical
                });
            }
        }

        return warnings;
    }

    // Public member to allow the visualization module to draw the ROI
    cv::Rect get_roi_rect() const {
        return roi;
    }

private:
    cv::Rect roi;
    float ttc_threshold_inside_roi;
    float ttc_threshold_outside_roi;

    /**
     * @brief Determines if an object's center point is inside the critical ROI.
     * @param bbox The bounding box of the object.
     * @return True if the center is inside the ROI, false otherwise.
     */
    bool is_object_in_roi(const cv::Rect& bbox) const {
        // Calculate the center point of the bounding box
        cv::Point center_point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        
        // Check if the center point is contained within the ROI rectangle
        return roi.contains(center_point);
    }
};
