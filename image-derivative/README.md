# AMD Hardware Progress Report

## Date 14-07-2025

Current focus has shifted to implementing HLS (High-Level Synthesis) optimizations for the image processing pipeline. The work is concentrated on accelerating the optical flow computation and object detection components to achieve the target 30-40ms performance requirement.

Key developments:
- Exploring Vitis HLS for implementing custom hardware accelerators for dense optical flow computation
- Developing image derivative vitis kernel module using HLS for improved performance in edge detection and feature extraction
- Working on optimizing the PWC-Net implementation for FPGA deployment

Hardware acceleration targets:
- Dense optical flow computation (currently the bottleneck at 120ms)
- Image preprocessing and derivative calculations
- Object detection pipeline optimization

The software pipeline remains stable with the Echo State Network for TTC estimation showing promising results on the EvTTC dataset.

---

## Date 23-06-2025

We successfully ran sample model inference on Kria through Vitis Ai Library. We are now working on implementing our segmentation model. Also we are clear with the hardware development flow now, so it would be easier to work from now on.

On the software side the model is being trained for time to collision estimation. The current dataset used for it is the one mentioned last week and performance is to be evaluated and integration with the current pipeline is to be experimented.

## Date 15-06-2025

Hardware is set up, but we had been facing issues with running Ubuntu on the fpga so we plan to shift to peta linux. Honestly speaking, not much work was done on the hardware side since the people on campus left for home and the other set of students arrived this week. We need to compensate for this in the coming week.

An effort has been put in creating a dataset for Time to collision calculation using a DL Model using an Echo State Network. This particular architecture has been chosen due to its model simplicity despite performing predictions on temporal data. Also this type of model is preferred for real time applications of DL models for temporal predictions. This is preferred over GRUs, LSTMs and Vision Transformers for our use case.

The dataset for training this model was made using already existing kaggle dataset which has dashcam camera feed , the dataset was prepared using Param Ganga and then after analysing the dataset it was found that it is not very efficient, continuing with it would produce an inefficient model. 

So, a new dataset from https://nail-hnu.github.io/EvTTC/competition/ has been used which seems very promising. We are near to converting this into a suitable format for our use case.

The architecture for the echo state network is ready and also an additional Multi layer perceptron (MLP) layer is added to the final layers for enhanced non linearity for robust performance for our case, with this the software pipeline would be in a good state of completion. 

The above is a simple representation of the model architecture with 17 input features, all based on dense optic flow of objects.

**Summary:**

We had prepared the dataset and the model architecture for the pipeline and the pipeline is ready. Now only tweaks and updates need to be made for better performance. On the hardware side a few basic models were tried out but we were facing issues due to compatibility of software and lack of proper documentation and resources for ubuntu, so work needs to be shifted to peta linux.
This week's software progress was good but on the hardware side not much work has been done due to travelling of our team members mainly and it must be compensated in the upcoming weeks.

## Date 06-06-2025

Tried out Yolo NAS for slightly better performance but the model is not suitable for running due to libraries being obsolete.
Hardware was set up and all basic updates and packages were installed and the kria board is functioning well and UART connection with kria has been implemented to run it in the command line interface to save resources.
A model has been created for Time to collision (TTC) estimation which both increases robustness and accuracy while decreasing computational load since a lightweight architecture is designed.

The issues we had noticed are:
- The kria board only supports ethernet input and that has become an issue as working lan ports for internet connection are not easily available and even if it is available the connection is being very unstable and unreliable. So we are using wifi sharing from laptop to the kria board as of now but a wifi adapter is needed.

The below image describes the pipeline to create input data for training a TIme to Collision estimation DL Model:

The model for TTC calculation needs to be trained by giving optic flow frames as input. So with the dataset we have it is first necessary to calculate optic flow frames for each frame for the entire dataset which might take days in our hardware, we would need a powerful gpu to calculate and store these optic flow frames using which models can be trained and tested quickly. This is a one time action to be performed.

## Date 25-05-2025

A working pipeline has been implemented using GENetic Ego-Motion Estimation with Visual Odometry(GENEVO) for reducing the impact of ego motion error 

Yolo V9T is used for object detection and segmentation. This model is the best tradeoff between speed and accuracy.

PWC net is used to then compute dense optic flow for each pair of frames

A custom algorithm is used for Time To Collision(TTC) calculation using divergence of flow vectors and flow vector magnitude

This pipeline is successfully implemented in python using cuda and pytorch, the current shortcomings are that it takes around 120ms to compute each frame whereas the required performance is 30-40 ms for real time performance.

**Base Pipeline:**
The pipeline is pretty straightforward. It basically takes in data from video(30fps) frame by frame, then applies ego motion correction then does object segmentation, detection and applies bounding boxes on objects, then applies dense optic flow for the entire frame, then for each box, it calculates ttc using a divergence and flow vector magnitude based formula

**References:**
- PWC Net implementation: https://github.com/hmorimitsu/ptlflow
- Yolo model: Yolo v9t
- Ego motion correction: https://doi.org/10.3390/a18010019
- Time to collision algorithm: weighted average of divergence-based and flow vector magnitude-based algorithms

---

## Current Status Summary

**Hardware:** Kria board operational with Vitis AI Library integration. Focus on HLS optimization for performance improvements.

**Software:** Echo State Network for TTC estimation trained on EvTTC dataset. Pipeline stable but requires hardware acceleration to meet real-time requirements.

**Performance Target:** 30-40ms per frame (currently 120ms)

**Next Steps:** Complete HLS implementation for optical flow acceleration and integrate with existing pipeline.