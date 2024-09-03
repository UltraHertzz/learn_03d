#include <torch/extension.h>

__global__ void inverse_projection_kernel(
    const float* __restrict__ K, 
    const float* __restrict__ depth_image, 
    const unsigned char* __restrict__ rgb_image, 
    const int* __restrict__ instance_mask, 
    float* points, 
    unsigned char* colors, 
    int* ids, 
    int width, 
    int height) {

    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u < width && v < height) {
        float fx = K[0];
        float fy = K[4];
        float cx = K[2];
        float cy = K[5];

        int idx = v * width + u;
        float Z = depth_image[idx];

        if (Z > 0) {
            float x_n = (u - cx) / fx;
            float y_n = (v - cy) / fy;

            float X = x_n * Z;
            float Y = y_n * Z;

            points[3 * idx] = X;
            points[3 * idx + 1] = Y;
            points[3 * idx + 2] = Z;

            colors[3 * idx] = rgb_image[3 * idx];
            colors[3 * idx + 1] = rgb_image[3 * idx + 1];
            colors[3 * idx + 2] = rgb_image[3 * idx + 2];

            if (instance_mask != nullptr) {
                ids[idx] = instance_mask[idx];
            }
        }
    }
}

void inverse_projection_cuda(
    at::Tensor K, 
    at::Tensor depth_image, 
    at::Tensor rgb_image, 
    at::Tensor instance_mask, 
    at::Tensor points, 
    at::Tensor colors, 
    at::Tensor ids, 
    int width, 
    int height) {
    
    const int threads = 16;
    const dim3 blocks((width + threads - 1) / threads, (height + threads - 1) / threads);
    const dim3 threadsPerBlock(threads, threads);

    inverse_projection_kernel<<<blocks, threadsPerBlock>>>(
        K.data_ptr<float>(), 
        depth_image.data_ptr<float>(), 
        rgb_image.data_ptr<unsigned char>(), 
        instance_mask.defined() ? instance_mask.data_ptr<int>() : nullptr, 
        points.data_ptr<float>(), 
        colors.data_ptr<unsigned char>(), 
        ids.defined() ? ids.data_ptr<int>() : nullptr, 
        width, 
        height
    );
}
