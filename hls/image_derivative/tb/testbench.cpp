#include "../src/image_derivative.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

#define TEST_WIDTH 640
#define TEST_HEIGHT 480

void generate_test_image(pixel_t image[TEST_HEIGHT][TEST_WIDTH]) {
    for(int y = 0; y < TEST_HEIGHT; y++) {
        for(int x = 0; x < TEST_WIDTH; x++) {
            image[y][x] = (pixel_t)(128 + 50 * sin(x * 0.02) * cos(y * 0.02));
        }
    }
}

void golden_reference(pixel_t image[TEST_HEIGHT][TEST_WIDTH], 
                     derivative_t dx_ref[TEST_HEIGHT][TEST_WIDTH],
                     derivative_t dy_ref[TEST_HEIGHT][TEST_WIDTH]) {
    for(int y = 1; y < TEST_HEIGHT-1; y++) {
        for(int x = 1; x < TEST_WIDTH-1; x++) {
            dx_ref[y][x] = (derivative_t)(image[y-1][x+1] - image[y-1][x-1]) +
                          2*(derivative_t)(image[y][x+1] - image[y][x-1]) +
                          (derivative_t)(image[y+1][x+1] - image[y+1][x-1]);
            
            dy_ref[y][x] = (derivative_t)(image[y+1][x-1] - image[y-1][x-1]) +
                          2*(derivative_t)(image[y+1][x] - image[y-1][x]) +
                          (derivative_t)(image[y+1][x+1] - image[y-1][x+1]);
        }
    }
}

int main() {
    pixel_t test_image[TEST_HEIGHT][TEST_WIDTH];
    derivative_t dx_ref[TEST_HEIGHT][TEST_WIDTH];
    derivative_t dy_ref[TEST_HEIGHT][TEST_WIDTH];
    derivative_t dx_result[TEST_HEIGHT][TEST_WIDTH];
    derivative_t dy_result[TEST_HEIGHT][TEST_WIDTH];

    axis_stream_8 input_stream;
    axis_stream_16 dx_stream, dy_stream;

    generate_test_image(test_image);
    golden_reference(test_image, dx_ref, dy_ref);

    for(int y = 0; y < TEST_HEIGHT; y++) {
        for(int x = 0; x < TEST_WIDTH; x++) {
            ap_axiu<8,1,1,1> input_pixel;
            input_pixel.data = test_image[y][x];
            input_pixel.last = (x == TEST_WIDTH-1) && (y == TEST_HEIGHT-1);
            input_pixel.keep = 0xFF;
            input_stream << input_pixel;
        }
    }

    image_derivative_kernel(input_stream, dx_stream, dy_stream, TEST_WIDTH, TEST_HEIGHT);

    int output_count = 0;
    while(!dx_stream.empty() && !dy_stream.empty()) {
        ap_axiu<16,1,1,1> dx_out, dy_out;
        dx_stream >> dx_out;
        dy_stream >> dy_out;
        
        int y = (output_count / (TEST_WIDTH-2)) + 2;
        int x = (output_count % (TEST_WIDTH-2)) + 2;
        
        if(y < TEST_HEIGHT && x < TEST_WIDTH) {
            dx_result[y][x] = dx_out.data;
            dy_result[y][x] = dy_out.data;
        }
        output_count++;
    }

    int errors = 0;
    for(int y = 2; y < TEST_HEIGHT-2; y++) {
        for(int x = 2; x < TEST_WIDTH-2; x++) {
            if(abs(dx_result[y][x] - dx_ref[y][x]) > 1 || 
               abs(dy_result[y][x] - dy_ref[y][x]) > 1) {
                errors++;
                if(errors < 10) {
                    std::cout << "Error at (" << x << "," << y << "): ";
                    std::cout << "dx expected=" << dx_ref[y][x] << " got=" << dx_result[y][x];
                    std::cout << " dy expected=" << dy_ref[y][x] << " got=" << dy_result[y][x] << std::endl;
                }
            }
        }
    }

    if(errors == 0) {
        std::cout << "TEST PASSED: All derivatives computed correctly" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAILED: " << errors << " errors found" << std::endl;
        return 1;
    }
}