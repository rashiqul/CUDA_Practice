#include "convolution_verify.h"

#include <cmath>
#include <cstdio>
#include <wb.h>

static void write_result_1d(const float *output, int width)
{
    FILE *fp = fopen(DATA_DIRECTORY_1D "/result.dat", "w");
    if (fp == nullptr)
    {
        wbLog(ERROR, "Could not open 1D result.dat for writing");
        return;
    }
    fprintf(fp, "%d\n", width);
    for (int i = 0; i < width; i++)
    {
        fprintf(fp, "%f\n", output[i]);
    }
    fclose(fp);
    wbLog(TRACE, "1D result written to " DATA_DIRECTORY_1D "/result.dat");
}

static void write_result_2d(const float *output, int rows, int cols)
{
    FILE *fp = fopen(DATA_DIRECTORY_2D "/result.dat", "w");
    if (fp == nullptr)
    {
        wbLog(ERROR, "Could not open 2D result.dat for writing");
        return;
    }
    fprintf(fp, "%d\n%f\n%f\n", rows * cols + 2, (float)rows, (float)cols);
    for (int i = 0; i < rows * cols; i++)
    {
        fprintf(fp, "%f\n", output[i]);
    }
    fclose(fp);
    wbLog(TRACE, "2D result written to " DATA_DIRECTORY_2D "/result.dat");
}

void verify_1D_conv(const float *output, int width)
{
    write_result_1d(output, width);

    int expectedLength;
    float *expectedOutput = (float *)wbImport(DATA_DIRECTORY_1D "/output.dat", &expectedLength);

    // No dimension header in output.dat; values start at index 0
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
    wbLog(TRACE, "1D convolution ", passed ? "PASSED" : "FAILED");

    free(expectedOutput);
}

void verify_2D_conv(const float *output, int rows, int cols)
{
    write_result_2d(output, rows, cols);

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
    wbLog(TRACE, "2D convolution ", passed ? "PASSED" : "FAILED");

    free(expectedOutput);
}
