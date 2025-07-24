# Real-Time Collision Avoidance System with Optical Flow Estimation

**Status**: Under Development

---

## Overview

This project implements a real-time collision detection system that integrates deep learning-based object detection with optical flow, Kalman filtering, and Time-to-Collision (TTC) estimation.

The system is designed for high-performance edge applications such as autonomous vehicles, mobile robots, and real-time monitoring systems. It processes incoming video streams to identify objects, track their motion, and predict possible collisions based on motion dynamics and scene understanding.

---

## System Architecture

```text
Video Stream
    ↓
Object Detection (YOLOv9t)
    ↓
Object Tracking (Kalman Filter + Hungarian Matching)
    ↓
Optical Flow (PWCNet via PTLFlow)
    ↓
Ego Motion Correction
    ↓
TTC Estimation (Divergence / Flow / Looming)
    ↓
Collision Logic & ROI Check
    ↓
Annotated Output Frame


