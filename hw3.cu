#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <chrono>

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

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels, int TILE_WIDTH) {
    
    //extern __shared__ unsigned char sharedMem[];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("In kernel.\n");
    }
    return;
    /*
    int vv, uu, v, u, i;
    int R, G, B;
    int val[MASK_N * 3] = {0};
    int cur_mem_row = 0;
    int end_mem_row = 4;

    int numOfRowChunk = (height + 1279) / 1280;
    int numOfColChunk = (width + 6404) / 6400;
    int rows_per_block = (height + numOfRowChunk - 1) / numOfRowChunk;
    int chunk_index = blockIdx.x % numOfColChunk;

    int start_column = chunk_index * TILE_WIDTH;
    int end_column = start_column + TILE_WIDTH > (width+5) ? width+5 : start_column + TILE_WIDTH;

    int start_row = blockIdx.x / numOfColChunk * rows_per_block;
    int end_row = start_row + rows_per_block > height ? height : start_row + rows_per_block;

    int len = end_column - start_column + 1;
    int stride = blockDim.x;

    if (start_row < end_row){
        for(i=threadIdx.x; i<len; i+=stride){
            for(int j=0; j<5; ++j){
                sharedMem[channels * (j * TILE_WIDTH + i) + 0] = s[channels * ((start_row+j)*(width+5) + chunk_index*TILE_WIDTH + i) + 0];
                sharedMem[channels * (j * TILE_WIDTH + i) + 1] = s[channels * ((start_row+j)*(width+5) + chunk_index*TILE_WIDTH + i) + 1];
                sharedMem[channels * (j * TILE_WIDTH + i) + 2] = s[channels * ((start_row+j)*(width+5) + chunk_index*TILE_WIDTH + i) + 2];
            }
        }
    }

    __syncthreads();      
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Shared Memory First Load: R=%d, G=%d, B=%d\n", 
            sharedMem[0], sharedMem[1], sharedMem[2]);
    }

    while(start_row < end_row){
        for(int index=threadIdx.x; index<len; index+=stride){
            val[0] = 0;
            val[1] = 0;
            val[2] = 0;
            val[3] = 0;
            val[4] = 0;
            val[5] = 0;

            for (v = 0; v < 5; ++v) {
                for (u = 0; u < 5; ++u) {
                    vv = (v + cur_mem_row + index) % 5;
                    uu = (u + cur_mem_row + index) % 5;

                    R = sharedMem[channels * ((TILE_WIDTH) * (vv) + (uu)) + 2];
                    G = sharedMem[channels * ((TILE_WIDTH) * (vv) + (uu)) + 1];
                    B = sharedMem[channels * ((TILE_WIDTH) * (vv) + (uu)) + 0];

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
            t[channels * (width * start_row + chunk_index * TILE_WIDTH + index) + 2] = cR;
            t[channels * (width * start_row + chunk_index * TILE_WIDTH + index) + 1] = cG;
            t[channels * (width * start_row + chunk_index * TILE_WIDTH + index) + 0] = cB; 
        }

        start_row++;
        if(start_row < end_row){
            cur_mem_row = (cur_mem_row + 1) % 5;
            end_mem_row = (end_mem_row + 1) % 5;
            for(int i=threadIdx.x; i<len; i+=stride){
                sharedMem[channels * (end_mem_row * TILE_WIDTH + i) + 0] = s[channels * ((start_row+4)*(width+5) + chunk_index*TILE_WIDTH + i) + 0];
                sharedMem[channels * (end_mem_row * TILE_WIDTH + i) + 1] = s[channels * ((start_row+4)*(width+5) + chunk_index*TILE_WIDTH + i) + 1];
                sharedMem[channels * (end_mem_row * TILE_WIDTH + i) + 2] = s[channels * ((start_row+4)*(width+5) + chunk_index*TILE_WIDTH + i) + 2];
            }

            __syncthreads();
        }
    } */
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
    cudaMallocManaged(&dst_img, height * width * channels * sizeof(unsigned char));

    auto start_copy = std::chrono::high_resolution_clock::now();

    unsigned char* mod_src_img;
    cudaMallocManaged(&mod_src_img, (height+5) * (width+5) * channels * sizeof(unsigned char));

    // memset(mod_src_img, 0, (height+5) * (width+5) * channels * sizeof(unsigned char));
    for(int i=0; i<height+5; ++i){
        if(i < 2 || i >= height + 2){
            memset(mod_src_img, 0, (width+5) * channels);
        }
        else{
            for(int j=0; j<3; ++j){
                mod_src_img[channels * (i * (width+5) + 0) + j] = 0;
                mod_src_img[channels * (i * (width+5) + 1) + j] = 0;
                mod_src_img[channels * (i * (width+5) + width+2) + j] = 0;
                mod_src_img[channels * (i * (width+5) + width+3) + j] = 0;
                mod_src_img[channels * (i * (width+5) + width+4) + j] = 0;
            }
        }
    }

    int num = width * channels * sizeof(unsigned char);
    for(int i=0; i<height; ++i){
        memcpy(mod_src_img + channels * ((i+2) * (width+5) + 2), src_img + channels * i * width, num);
    }

    cudaMemPrefetchAsync(mod_src_img, (height+5) * (width+5) * channels * sizeof(unsigned char), deviceId);

    auto end_copy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_copy - start_copy;
    std::cout << "Image Copy Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    auto start_sobel = std::chrono::high_resolution_clock::now();
    
    int numOfColumn = (width + 5 + 6400 - 1) / 6400;
    int columnWidth = (width + 5 + numOfColumn - 1) / numOfColumn;
    int sharedMemSize = 5 * columnWidth * channels * sizeof(unsigned char);
    fprintf(stderr, "shared memory size: %d\n", sharedMemSize);

    sobel<<<numberOfBlocks, threadsPerBlock, sharedMemSize>>>(mod_src_img, dst_img, height, width, channels, columnWidth);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

    auto end_sobel = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_sobel - start_sobel;
    std::cout << "Sobel Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    cudaMemPrefetchAsync(dst_img, height * width * channels * sizeof(unsigned char), cudaCpuDeviceId);

    auto start_write = std::chrono::high_resolution_clock::now();

    write_png(argv[2], dst_img, height, width, channels);

    auto end_write = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_write - start_write;
    std::cout << "Write Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    // free(src_img);
    // fprintf(stderr, "free src_img\n");
    cudaFree(dst_img);
    cudaFree(mod_src_img);

    auto end_all = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end_all - start_all;
    std::cout << "Total Time: " << elapsed_seconds.count() * 1000.0 << " ms" << std::endl;

    return 0;
}
