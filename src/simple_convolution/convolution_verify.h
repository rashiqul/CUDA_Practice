#ifndef CONVOLUTION_VERIFY_H
#define CONVOLUTION_VERIFY_H

#define DATA_DIRECTORY_1D "src/simple_convolution/data/1d"
#define DATA_DIRECTORY_2D "src/simple_convolution/data/2d"
#define DATA_DIRECTORY_3D "src/simple_convolution/data/3d"

void verify_1D_basic_conv(const float *output, int width);
void verify_1D_tiled_conv(const float *output, int width);
void verify_2D_basic_conv(const float *output, int rows, int cols);
void verify_2D_tiled_conv(const float *output, int rows, int cols);
void verify_3D_basic_conv(const float *output, int depth, int rows, int cols);
void verify_3D_tiled_conv(const float *output, int depth, int rows, int cols);

#endif // CONVOLUTION_VERIFY_H
