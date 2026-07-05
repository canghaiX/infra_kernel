/*
练习项目 4：CUDA 实现简化版 vector add、reduction、softmax。

这份代码刻意保持简单：
- vector add: 一个 thread 处理一个元素。
- reduction: 一个 block 先做局部求和，host 再汇总 partial sums。
- softmax: 一个 block 处理一行，先 max reduction，再 sum reduction。
*/

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void reduce_sum_kernel(const float* x, float* partial_sums, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float local = 0.0f;
    if (i < n) {
        local += x[i];
    }
    if (i + blockDim.x < n) {
        local += x[i + blockDim.x];
    }
    shared[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

__global__ void row_softmax_kernel(const float* x, float* out, int rows, int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, x[row * cols + col]);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(x[row * cols + col] - row_max);
    }
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = shared[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        out[row * cols + col] = expf(x[row * cols + col] - row_max) / row_sum;
    }
}

std::vector<float> make_data(int n, float scale) {
    std::vector<float> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = std::sin(i * scale);
    }
    return data;
}

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

void print_error(const std::string& name, const std::vector<float>& got, const std::vector<float>& expected) {
    std::cout << name << " max_error=" << max_abs_error(got, expected)
              << ", elements=" << got.size() << std::endl;
}

void demo_vector_add() {
    int n = 10000;
    auto a = make_data(n, 0.01f);
    auto b = make_data(n, 0.02f);
    std::vector<float> c(n), expected(n);
    for (int i = 0; i < n; ++i) {
        expected[i] = a[i] + b[i];
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, n * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    print_error("vector_add", c, expected);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

void demo_reduction() {
    int n = 10000;
    auto x = make_data(n, 0.001f);
    float expected = 0.0f;
    for (float v : x) {
        expected += v;
    }

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    std::vector<float> partial(blocks);

    float *d_x, *d_partial;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_x, d_partial, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float got = 0.0f;
    for (float v : partial) {
        got += v;
    }

    std::cout << "reduction max_error=" << std::fabs(got - expected)
              << ", elements=" << n << std::endl;
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_partial));
}

void demo_softmax() {
    int rows = 8;
    int cols = 257;
    auto x = make_data(rows * cols, 0.01f);
    std::vector<float> out(rows * cols), expected(rows * cols);

    for (int r = 0; r < rows; ++r) {
        float row_max = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            row_max = std::max(row_max, x[r * cols + c]);
        }
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            row_sum += std::exp(x[r * cols + c] - row_max);
        }
        for (int c = 0; c < cols; ++c) {
            expected[r * cols + c] = std::exp(x[r * cols + c] - row_max) / row_sum;
        }
    }

    float *d_x, *d_out;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, out.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    row_softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(d_x, d_out, rows, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    print_error("softmax", out, expected);
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_out));
}

int main() {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "device: " << prop.name << std::endl;

    demo_vector_add();
    demo_reduction();
    demo_softmax();

    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "CUDA simplified practice finished." << std::endl;
    return 0;
}
