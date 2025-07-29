#include "image_derivative.hpp"
#include "hls_math.h"

void image_derivative_kernel(
    axis_stream_8& input_stream,
    axis_stream_16& dx_stream,
    axis_stream_16& dy_stream,
    int width,
    int height
) {
    #pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=dx_stream
    #pragma HLS INTERFACE axis port=dy_stream
    #pragma HLS INTERFACE s_axilite port=width
    #pragma HLS INTERFACE s_axilite port=height
    #pragma HLS INTERFACE s_axilite port=return

    static pixel_t line_buffer[3][WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    #pragma HLS RESOURCE variable=line_buffer core=RAM_S2P_BRAM

    static pixel_t window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    pixel_loop: for(int y = 0; y < height; y++) {
        #pragma HLS LOOP_TRIPCOUNT min=1080 max=1080 avg=1080
        
        col_loop: for(int x = 0; x < width; x++) {
            #pragma HLS LOOP_TRIPCOUNT min=1920 max=1920 avg=1920
            #pragma HLS PIPELINE II=1

            ap_axiu<8,1,1,1> input_pixel;
            pixel_t current_pixel = 0;
            
            if(y < height && x < width) {
                input_stream >> input_pixel;
                current_pixel = input_pixel.data;
            }

            if(y >= 2 && x >= 2) {
                window[0][0] = window[0][1]; window[0][1] = window[0][2]; window[0][2] = line_buffer[0][x];
                window[1][0] = window[1][1]; window[1][1] = window[1][2]; window[1][2] = line_buffer[1][x];
                window[2][0] = window[2][1]; window[2][1] = window[2][2]; window[2][2] = line_buffer[2][x];

                derivative_t dx = (derivative_t)(window[0][2] - window[0][0]) + 
                                 2*(derivative_t)(window[1][2] - window[1][0]) + 
                                 (derivative_t)(window[2][2] - window[2][0]);

                derivative_t dy = (derivative_t)(window[2][0] - window[0][0]) + 
                                 2*(derivative_t)(window[2][1] - window[0][1]) + 
                                 (derivative_t)(window[2][2] - window[0][2]);

                ap_axiu<16,1,1,1> dx_out, dy_out;
                dx_out.data = dx;
                dy_out.data = dy;
                dx_out.last = (x == width-1) && (y == height-1);
                dy_out.last = (x == width-1) && (y == height-1);
                dx_out.keep = 0xFFFF;
                dy_out.keep = 0xFFFF;

                dx_stream << dx_out;
                dy_stream << dy_out;
            }

            line_buffer[0][x] = line_buffer[1][x];
            line_buffer[1][x] = line_buffer[2][x];
            line_buffer[2][x] = current_pixel;
        }
    }
}
