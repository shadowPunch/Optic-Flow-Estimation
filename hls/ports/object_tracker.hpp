#pragma once

#include <vector>
#include <map>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/core.hpp>

// --- Placeholder for a C++ Hungarian Algorithm Library ---
// In a real project, you would include the header from a library like:
// - https://github.com/mcximing/hungarian-algorithm-cpp
// - https://github.com/daobilige-su/Hungarian_algorithm
#include "hungarian.h" // Assumed header for the algorithm

// A struct to hold detection data passed to the tracker
struct Detection {
    cv::Rect bbox;
    std::string class_name;
    float confidence;
};

// A struct to hold all data for a single tracked object
struct TrackData {
    cv::KalmanFilter kf;
    cv::Rect bbox;
    std::string class_name;
    int lost_frames = 0;
    int age = 1;
    int id;
};

class ObjectTracker {
public:
    ObjectTracker(int max_lost = 30, float iou_thresh = 0.3)
        : max_frames_to_lost(max_lost), iou_threshold(iou_thresh), next_track_id(0) {}

    // Main update function to be called each frame
    void update(const std::vector<Detection>& detections) {
        // 1. Predict new locations of existing tracks
        for (auto& pair : tracks) {
            pair.second.bbox = predict_new_location(pair.second.kf);
        }

        // 2. Associate detections with existing tracks using IoU and Hungarian Algorithm
        auto [matched_indices, unmatched_track_keys, unmatched_det_indices] =
            associate_detections(detections);

        // 3. Update matched tracks with new detection data
        for (const auto& match : matched_indices) {
            int track_id = match.first;
            int det_idx = match.second;
            const auto& det = detections[det_idx];

            tracks[track_id].bbox = det.bbox;
            tracks[track_id].class_name = det.class_name;
            tracks[track_id].age++;
            tracks[track_id].lost_frames = 0;

            // Correct the Kalman Filter with the new measurement
            cv::Mat measurement = (cv::Mat_<float>(2, 1) << det.bbox.x + det.bbox.width / 2.0f,
                                   det.bbox.y + det.bbox.height / 2.0f);
            tracks[track_id].kf.correct(measurement);
        }

        // 4. Update unmatched tracks (mark as lost)
        for (int track_id : unmatched_track_keys) {
            tracks[track_id].lost_frames++;
        }

        // 5. Create new tracks for unmatched detections
        for (int det_idx : unmatched_det_indices) {
            create_new_track(detections[det_idx]);
        }

        // 6. Remove tracks that have been lost for too long
        remove_lost_tracks();
    }

    // Public member to access the tracks
    std::map<int, TrackData> tracks;

private:
    int max_frames_to_lost;
    float iou_threshold;
    int next_track_id;

    float calculate_iou(const cv::Rect& boxA, const cv::Rect& boxB) {
        float interArea = (boxA & boxB).area();
        float unionArea = boxA.area() + boxB.area() - interArea;
        return (unionArea > 0) ? (interArea / unionArea) : 0;
    }

    void create_new_track(const Detection& det) {
        cv::KalmanFilter kf(4, 2, 0, CV_32F);

        // Setup Kalman Filter matrices
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
        cv::setIdentity(kf.measurementMatrix);
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0));

        // Initialize state
        float cx = det.bbox.x + det.bbox.width / 2.0f;
        float cy = det.bbox.y + det.bbox.height / 2.0f;
        kf.statePost = (cv::Mat_<float>(4, 1) << cx, cy, 0, 0);

        tracks[next_track_id] = {kf, det.bbox, det.class_name, 0, 1, next_track_id};
        next_track_id++;
    }

    cv::Rect predict_new_location(cv::KalmanFilter& kf) {
        cv::Mat prediction = kf.predict();
        float cx = prediction.at<float>(0);
        float cy = prediction.at<float>(1);
        // Assuming bbox size remains constant for prediction
        cv::Rect predicted_box = tracks[kf.statePost.at<int>(0)].bbox; // This needs a better way to get width/height
        predicted_box.x = cx - predicted_box.width / 2.0f;
        predicted_box.y = cy - predicted_box.height / 2.0f;
        return predicted_box;
    }

    void remove_lost_tracks() {
        for (auto it = tracks.cbegin(); it != tracks.cend();) {
            if (it->second.lost_frames > max_frames_to_lost) {
                it = tracks.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // This function replaces the scipy call
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    associate_detections(const std::vector<Detection>& detections) {
        if (tracks.empty()) {
            std::vector<int> unmatched_dets;
            for(int i = 0; i < detections.size(); ++i) unmatched_dets.push_back(i);
            return {{}, {}, unmatched_dets};
        }

        std::vector<int> track_ids;
        for(const auto& pair : tracks) track_ids.push_back(pair.first);

        // Create cost matrix (negative IoU)
        std::vector<std::vector<double>> cost_matrix(tracks.size(), std::vector<double>(detections.size()));
        for (size_t i = 0; i < tracks.size(); ++i) {
            for (size_t j = 0; j < detections.size(); ++j) {
                cost_matrix[i][j] = -calculate_iou(tracks[track_ids[i]].bbox, detections[j].bbox);
            }
        }
        
        // --- Call the C++ Hungarian Algorithm Library ---
        HungarianAlgorithm H; // From the assumed library
        std::vector<int> assignment;
        H.Solve(cost_matrix, assignment);
        // --- End of Library Call ---

        std::vector<std::pair<int, int>> matched_indices;
        std::vector<bool> matched_dets(detections.size(), false);
        std::vector<bool> matched_tracks(tracks.size(), false);
        
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (assignment[i] != -1) {
                // Check if the match is above the IoU threshold
                if (-cost_matrix[i][assignment[i]] >= iou_threshold) {
                    matched_indices.push_back({track_ids[i], assignment[i]});
                    matched_tracks[i] = true;
                    matched_dets[assignment[i]] = true;
                }
            }
        }

        std::vector<int> unmatched_track_keys;
        for(size_t i=0; i < tracks.size(); ++i) {
            if(!matched_tracks[i]) unmatched_track_keys.push_back(track_ids[i]);
        }

        std::vector<int> unmatched_det_indices;
        for(size_t i=0; i < detections.size(); ++i) {
            if(!matched_dets[i]) unmatched_det_indices.push_back(i);
        }

        return {matched_indices, unmatched_track_keys, unmatched_det_indices};
    }
};
