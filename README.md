# Real-Time Collision Avoidance System with Optical Flow Estimation

**Status**: Under Development

---

## Overview

This project implements a real-time collision avoidance system designed for autonomous vehicles. It leverages deep learning models for accurate environment perception and is accelerated on an AMD Kria KV260 FPGA to meet the demanding performance requirements of real-time applications.

The core of the system is a computer vision pipeline that performs the following tasks:
- **Object Detection and Segmentation:** Using **YOLOv9T** to identify and locate objects like cars and pedestrians.
- **Motion Estimation:** Employing **PWC-Net** to compute dense optical flow, which describes the motion of objects between frames.
- **Ego-Motion Compensation:** Utilizing the **GENEVO** algorithm to distinguish between the vehicle's own movement and the independent motion of other objects.
- **Time to Collision (TTC) Estimation:** Calculating the TTC for detected objects to predict potential collisions.

The entire pipeline is optimized for deployment on the AMD Kria platform, aiming for a processing speed of 30-40ms per frame.

## System Pipeline

The collision avoidance system processes video data through the following pipeline:

1.  **Video Input:** The system takes a live video stream (e.g., 30 FPS) from a vehicle-mounted camera.
2.  **Ego-Motion Correction:** The GENEVO algorithm is applied to compensate for the vehicle's own motion, ensuring that the optical flow analysis focuses on the movement of external objects.
3.  **Object Detection & Segmentation:** YOLOv9T processes each frame to detect and draw bounding boxes around relevant objects.
4.  **Dense Optical Flow:** PWC-Net calculates a dense optical flow field for the entire frame, providing a motion vector for each pixel.
5.  **Time to Collision (TTC) Calculation:** For each detected object, a custom algorithm calculates the TTC based on the divergence of flow vectors and their magnitude. An Echo State Network (ESN) is also being developed for more robust TTC prediction.

   
## Hardware and Software

**Hardware:**
* **FPGA:** AMD Kria KV260 Vision AI Starter Kit

**Software & Frameworks:**
* **Operating System:** PetaLinux
* **Language:** Python, C++
* **Frameworks:** PyTorch, CUDA
* **FPGA Development:** Vitis HLS, Vitis AI

## 🤖 Models and Algorithms

* **Object Detection:** **YOLOv9T** is chosen for its excellent balance of speed and accuracy.
* **Optical Flow:** **PWC-Net** is used for its efficiency and high-quality flow estimations.
* **Ego-Motion Correction:** Based on the paper: [GENetic Ego-Motion Estimation with Visual Odometry (GENEVO)](https://doi.org/10.3390/a18010019).
* **Time to Collision (TTC) Estimation:**
    * A custom algorithm that uses a weighted average of divergence and flow vector magnitude.
    * An **Echo State Network (ESN) combined with an MLP** is under development for enhanced performance on temporal data.

## Current Status & Performance

The software pipeline has been successfully implemented and tested. However, the current performance is approximately **120ms per frame**, which is short of the **30-40ms** target required for real-time operation.

The current focus is on hardware acceleration using Vitis HLS to optimize the most computationally intensive parts of the pipeline:
* **Dense optical flow computation**
* **Image derivative calculations**
* **Object detection pipeline**

We are actively exploring pipelining and parallelizing these operations to offload them from the CPU to the FPGA fabric.


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
```
---

## Future Work

* Complete the HLS optimization of the optical flow and object detection modules.
* Integrate the trained ESN + MLP model for TTC estimation into the main pipeline.
* Conduct extensive testing with real-world driving data.
* Further refine the algorithms to improve accuracy and robustness.

## Other Contributors



* Shyam B Ganesh (https://github.com/sh-yamm)
* Tanmay S Kushwaha (https://github.com/Tanmay-S-Kushwaha)
* Prateek Ratan (https://github.com/Pratan1)
* Daksh Pandey (https://github.com/D1729)

## Acknowledgments

* **PWC-Net Implementation:** [https://github.com/hmorimitsu/ptiflow](https://github.com/hmorimitsu/ptiflow)
* **EvTTC Dataset:** [https://nail-hnu.github.io/EvTTC/](https://nail-hnu.github.io/EvTTC/)

