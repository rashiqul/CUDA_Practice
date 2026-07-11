#include "convolution_verify.h"

#include <cmath>
#include <cstdio>
#include <wb.h>

static void write_result_1d(const float *output, int width, const char *result_path)
{
    FILE *fp = fopen(result_path, "w");
    if (fp == nullptr)
    {
        wbLog(ERROR, "Could not open result file for writing: ", result_path);
        return;
    }
    fprintf(fp, "%d\n", width);
    for (int i = 0; i < width; i++)
    {
        fprintf(fp, "%f\n", output[i]);
    }
    fclose(fp);
    wbLog(TRACE, "1D result written to ", result_path);
}

static void write_result_2d(const float *output, int rows, int cols, const char *result_path)
{
    FILE *fp = fopen(result_path, "w");
    if (fp == nullptr)
    {
        wbLog(ERROR, "Could not open result file for writing: ", result_path);
        return;
    }
    fprintf(fp, "%d\n%f\n%f\n", rows * cols + 2, (float)rows, (float)cols);
    for (int i = 0; i < rows * cols; i++)
    {
        fprintf(fp, "%f\n", output[i]);
    }
    fclose(fp);
    wbLog(TRACE, "2D result written to ", result_path);
}

static void compare_1d(const float *output, int width, const char *label)
{
    int expectedLength;
    float *expectedOutput = (float *)wbImport(DATA_DIRECTORY_1D "/output.dat", &expectedLength);

    bool passed = true;
    for (int i = 0; i < width; i++)
    {
        float diff = fabsf(output[i] - expectedOutput[i]);
        if (diff > 1e-4f)
        {
            wbLog(ERROR, "Mismatch at i=", i, ": got ", output[i], " expected ", expectedOutput[i]);
            passed = false;
        }
    }
    wbLog(TRACE, label, passed ? " PASSED" : " FAILED");

    free(expectedOutput);
}

static void compare_2d(const float *output, int rows, int cols, const char *label)
{
    int expectedLength;
    float *expectedOutput = (float *)wbImport(DATA_DIRECTORY_2D "/output.dat", &expectedLength);

    // expectedOutput[0] = rows header, [1] = cols header; data starts at [2]
    const float *expectedData = expectedOutput + 2;

    bool passed = true;
    for (int i = 0; i < rows * cols; i++)
    {
        float diff = fabsf(output[i] - expectedData[i]);
        if (diff > 1e-4f)
        {
            wbLog(ERROR, "Mismatch at [", i / cols, "][", i % cols, "]: got ", output[i], " expected ", expectedData[i]);
            passed = false;
        }
    }
    wbLog(TRACE, label, passed ? " PASSED" : " FAILED");

    free(expectedOutput);
}

void verify_1D_basic_conv(const float *output, int width)
{
    write_result_1d(output, width, DATA_DIRECTORY_1D "/output_1D_conv_basic.dat");
    compare_1d(output, width, "1D basic convolution");
}

void verify_1D_tiled_conv(const float *output, int width)
{
    write_result_1d(output, width, DATA_DIRECTORY_1D "/output_1D_conv_tiled.dat");
    compare_1d(output, width, "1D tiled convolution");
}

void verify_2D_basic_conv(const float *output, int rows, int cols)
{
    write_result_2d(output, rows, cols, DATA_DIRECTORY_2D "/output_2D_conv_basic.dat");
    compare_2d(output, rows, cols, "2D basic convolution");
}

void verify_2D_tiled_conv(const float *output, int rows, int cols)
{
    write_result_2d(output, rows, cols, DATA_DIRECTORY_2D "/output_2D_conv_tiled.dat");
    compare_2d(output, rows, cols, "2D tiled convolution");
}

static void write_result_3d(const float *output, int depth, int rows, int cols, const char *result_path)
{
    FILE *fp = fopen(result_path, "w");
    if (fp == nullptr)
    {
        wbLog(ERROR, "Could not open result file for writing: ", result_path);
        return;
    }
    fprintf(fp, "%d\n%f\n%f\n%f\n", depth * rows * cols + 3, (float)depth, (float)rows, (float)cols);
    for (int i = 0; i < depth * rows * cols; i++)
    {
        fprintf(fp, "%f\n", output[i]);
    }
    fclose(fp);
    wbLog(TRACE, "3D result written to ", result_path);
}

static void compare_3d(const float *output, int depth, int rows, int cols, const char *label)
{
    int expectedLength;
    float *expectedOutput = (float *)wbImport(DATA_DIRECTORY_3D "/output.dat", &expectedLength);

    // expectedOutput[0..2] = depth/rows/cols headers; data starts at [3]
    const float *expectedData = expectedOutput + 3;

    bool passed = true;
    for (int d = 0; d < depth; d++)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int idx = d * rows * cols + r * cols + c;
                float diff = fabsf(output[idx] - expectedData[idx]);
                if (diff > 1e-4f)
                {
                    wbLog(ERROR, "Mismatch at [", d, "][", r, "][", c, "]: got ", output[idx], " expected ", expectedData[idx]);
                    passed = false;
                }
            }
        }
    }
    wbLog(TRACE, label, passed ? " PASSED" : " FAILED");

    free(expectedOutput);
}

void verify_3D_basic_conv(const float *output, int depth, int rows, int cols)
{
    write_result_3d(output, depth, rows, cols, DATA_DIRECTORY_3D "/output_3D_conv_basic.dat");
    compare_3d(output, depth, rows, cols, "3D basic convolution");
}

void verify_3D_tiled_conv(const float *output, int depth, int rows, int cols)
{
    write_result_3d(output, depth, rows, cols, DATA_DIRECTORY_3D "/output_3D_conv_tiled.dat");
    compare_3d(output, depth, rows, cols, "3D tiled convolution");
}
