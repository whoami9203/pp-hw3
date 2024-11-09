#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_texture_types.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

// clang-format off
__constant__ int mask[MASK_N][MASK_X][MASK_Y] = {
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0},
     {  2,  8, 12,  8,  2},
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1},
     { -4, -8,  0,  8,  4},
     { -6,-12,  0, 12,  6},
     { -4, -8,  0,  8,  4},
     { -1, -2,  0,  2,  1}}
};
// clang-format on

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
    unsigned* channels) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    if (fread(sig, 1, 8, infile) <= 0) {
        printf("read fail");
        exit(1);
    }
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
    const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// int index = threadIdx.x + blockIdx.x * blockDim.x;
// int stride = blockDim.x * gridDim.x;
// cudaMemPrefetchAsync(a, size, deviceId);
// cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

__global__ void sobel(cudaTextureObject_t texRef, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int x, y, v, u;
    int R, G, B;
    int val[MASK_N * 3] = {0};
    int total_pixel = width * height;
    uchar4 pixel_val;

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;    

    while(index < total_pixel){
        y = index / width;
        x = index % width;

        val[0] = 0;
        val[1] = 0;
        val[2] = 0;
        val[3] = 0;
        val[4] = 0;
        val[5] = 0;
        
        for (v = 0; v < 5; ++v) {
            for (u = 0; u < 5; ++u) {
                pixel_val = tex1D<uchar4>(texRef, (width+5) * (y + v) + (x + u));
                R = pixel_val.z;
                G = pixel_val.y;
                B = pixel_val.x;

                val[2] += R * mask[0][u][v];
                val[1] += G * mask[0][u][v];
                val[0] += B * mask[0][u][v];

                val[5] += R * mask[1][u][v];
                val[4] += G * mask[1][u][v];
                val[3] += B * mask[1][u][v];
            }
        }

        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;
        
        totalR += val[2] * val[2] + val[5] * val[5];
        totalG += val[1] * val[1] + val[4] * val[4];
        totalB += val[0] * val[0] + val[3] * val[3];

        totalR = sqrt(totalR) / SCALE;
        totalG = sqrt(totalG) / SCALE;
        totalB = sqrt(totalB) / SCALE;
        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;

        index += stride;
    }
}

int main(int argc, char** argv) {
    assert(argc == 3);

    auto start_all = std::chrono::high_resolution_clock::now();

    cudaError_t err;
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;
    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    unsigned height, width, channels;
    unsigned char* src_img = NULL;

    read_png(argv[1], &src_img, &height, &width, &channels);
    assert(channels == 3);

    unsigned char* dst_img;
    err = cudaMallocManaged(&dst_img, height * width * channels * sizeof(unsigned char));
    fprintf(stderr, "dst_img error: %s\n", cudaGetErrorString(err));

    auto start_copy = std::chrono::high_resolution_clock::now();

    uchar4* mod_src_img_uchar4 = (uchar4*)calloc((height + 5) * (width + 5), sizeof(uchar4));

    int num = width * channels * sizeof(unsigned char);
    for(int i=0; i<height; ++i){
        memcpy(mod_src_img_uchar4 + ((i+2) * (width+5) + 2), src_img + channels * i * width, num);
    }

    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    err = cudaMallocArray(&cuArray, &channelDesc, (width+5) * (height+5));
    fprintf(stderr, "Malloc error: %s\n", cudaGetErrorString(err));

    err = cudaMemcpyToArray(cuArray, 0, 0, mod_src_img_uchar4, (width+5) * (height+5) * sizeof(uchar4), cudaMemcpyHostToDevice);
    fprintf(stderr, "Memcpy error: %s\n", cudaGetErrorString(err));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texRef;
    cudaCreateTextureObject(&texRef, &resDesc, &texDesc, NULL);

    // cudaMemPrefetchAsync(mod_src_img_h, (height+5) * (width+5) * channels * sizeof(unsigned char), deviceId);

    auto end_copy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_copy - start_copy;
    std::cout << "Image Copy Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    auto start_sobel = std::chrono::high_resolution_clock::now();
    sobel<<<numberOfBlocks, threadsPerBlock>>>(texRef, dst_img, height, width, channels);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

    auto end_sobel = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_sobel - start_sobel;
    std::cout << "Sobel Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    cudaMemPrefetchAsync(dst_img, height * width * channels * sizeof(unsigned char), cudaCpuDeviceId);

    write_png(argv[2], dst_img, height, width, channels);

    // free(src_img);
    // fprintf(stderr, "free src_img\n");
    cudaFree(dst_img);
    cudaFree(mod_src_img_uchar4);

    auto end_all = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_sobel - start_sobel;
    std::cout << "Total Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    return 0;
}
