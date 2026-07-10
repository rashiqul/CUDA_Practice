#ifndef CONVOLUTION_VERIFY_H
#define CONVOLUTION_VERIFY_H

#define DATA_DIRECTORY_1D "src/simple_convolution/data/1d"
#define DATA_DIRECTORY_2D "src/simple_convolution/data/2d"

void verify_1D_conv(const float *output, int width);
void verify_2D_conv(const float *output, int rows, int cols);

#endif // CONVOLUTION_VERIFY_H
