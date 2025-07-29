#ifndef IMAGE_DERIVATIVE_HPP
#define IMAGE_DERIVATIVE_HPP

#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#define WIDTH 1920
#define HEIGHT 1080
#define CHANNELS 1

typedef ap_uint<8> pixel_t;
typedef ap_int<16> derivative_t;
typedef hls::stream<ap_axiu<8,1,1,1>> axis_stream_8;
typedef hls::stream<ap_axiu<16,1,1,1>> axis_stream_16;

void image_derivative_kernel(
    axis_stream_8& input_stream,
    axis_stream_16& dx_stream,
    axis_stream_16& dy_stream,
    int width,
    int height
);

#endif
