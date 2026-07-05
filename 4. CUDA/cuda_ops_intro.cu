/*
CUDA 算子入门。

这份代码用纯 CUDA C++ 演示：

1. vector add
2. matrix transpose
3. reduction
4. row softmax
5. RMSNorm
6. naive matmul
7. shared memory tiled matmul

重点：
- thread/block/grid 如何映射到数据。
- global memory 如何读写。
- shared memory 如何缓存 tile。
- block 内 parallel reduction 如何写。
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

constexpr int kTile = 16;

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // 一维 grid：
    // 每个 thread 负责一个元素 c[i] = a[i] + b[i]。
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    // shared memory tile 多加 1 列，常用于减少 bank conflict。
    __shared__ float tile[kTile][kTile + 1];

    int x = blockIdx.x * kTile + threadIdx.x;
    int y = blockIdx.y * kTile + threadIdx.y;

    // 先从 global memory 连续读入 shared memory。
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // 再把 tile 转置后写出。
    int out_x = blockIdx.y * kTile + threadIdx.x;
    int out_y = blockIdx.x * kTile + threadIdx.y;
    if (out_x < rows && out_y < cols) {
        output[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void reduce_sum_kernel(const float* input, float* block_sums, int n) {
    // 一个 block 先规约出一个 partial sum，最后 host 再把 partial sum 加起来。
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (i < n) {
        sum += input[i];
    }
    if (i + blockDim.x < n) {
        sum += input[i + blockDim.x];
    }
    shared[tid] = sum;
    __syncthreads();

    // block 内 parallel reduction。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0];
    }
}

__global__ void row_softmax_kernel(const float* input, float* output, int rows, int cols) {
    // 一个 block 处理一行，适合演示 softmax 的两次 reduction。
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row * cols + col]);
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
        local_sum += expf(input[row * cols + col] - row_max);
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
        output[row * cols + col] = expf(input[row * cols + col] - row_max) / row_sum;
    }
}

__global__ void rmsnorm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int rows,
    int hidden,
    float eps
) {
    // 一个 block 处理一个 token 的 hidden 向量。
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_sum_sq = 0.0f;
    for (int col = tid; col < hidden; col += blockDim.x) {
        float x = input[row * hidden + col];
        local_sum_sq += x * x;
    }
    shared[tid] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared[0] / hidden + eps);

    for (int col = tid; col < hidden; col += blockDim.x) {
        output[row * hidden + col] = input[row * hidden + col] * inv_rms * weight[col];
    }
}

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 每个 thread 计算 C 的一个元素。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // shared memory 版本：
    // 每个 block 计算 C 的一个 tile。
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];

    int row = blockIdx.y * kTile + threadIdx.y;
    int col = blockIdx.x * kTile + threadIdx.x;

    float acc = 0.0f;

    for (int tile = 0; tile < (K + kTile - 1) / kTile; ++tile) {
        int a_col = tile * kTile + threadIdx.x;
        int b_row = tile * kTile + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < kTile; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

std::vector<float> make_data(int n, float scale = 0.01f) {
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
    for (int i = 0; i < n; ++i) expected[i] = a[i] + b[i];

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

void demo_transpose() {
    int rows = 31;
    int cols = 47;
    auto input = make_data(rows * cols, 0.03f);
    std::vector<float> output(rows * cols), expected(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            expected[c * rows + r] = input[r * cols + c];
        }
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(kTile, kTile);
    dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    transpose_kernel<<<grid, block>>>(d_input, d_output, rows, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    print_error("transpose", output, expected);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

void demo_reduction() {
    int n = 10000;
    auto input = make_data(n, 0.001f);
    float expected = 0.0f;
    for (float x : input) expected += x;

    int threads = 256;
    int blocks = (n + threads * 2 - 1) / (threads * 2);
    std::vector<float> partial(blocks);

    float *d_input, *d_partial;
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partial, blocks * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_partial, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    float got = 0.0f;
    for (float x : partial) got += x;
    std::cout << "reduction max_error=" << std::fabs(got - expected) << ", elements=" << n << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_partial));
}

void demo_softmax() {
    int rows = 8;
    int cols = 257;
    auto input = make_data(rows * cols, 0.01f);
    std::vector<float> output(rows * cols), expected(rows * cols);

    for (int r = 0; r < rows; ++r) {
        float row_max = -INFINITY;
        for (int c = 0; c < cols; ++c) row_max = std::max(row_max, input[r * cols + c]);
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c) row_sum += std::exp(input[r * cols + c] - row_max);
        for (int c = 0; c < cols; ++c) expected[r * cols + c] = std::exp(input[r * cols + c] - row_max) / row_sum;
    }

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    row_softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(d_input, d_output, rows, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    print_error("softmax", output, expected);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

void demo_rmsnorm() {
    int rows = 16;
    int hidden = 256;
    float eps = 1e-6f;
    auto input = make_data(rows * hidden, 0.01f);
    auto weight = make_data(hidden, 0.02f);
    std::vector<float> output(rows * hidden), expected(rows * hidden);

    for (int r = 0; r < rows; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < hidden; ++c) {
            float x = input[r * hidden + c];
            sum_sq += x * x;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden + eps);
        for (int c = 0; c < hidden; ++c) {
            expected[r * hidden + c] = input[r * hidden + c] * inv_rms * weight[c];
        }
    }

    float *d_input, *d_weight, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    rmsnorm_kernel<<<rows, threads, threads * sizeof(float)>>>(d_input, d_weight, d_output, rows, hidden, eps);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    print_error("rmsnorm", output, expected);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_output));
}

void demo_matmul() {
    int M = 33;
    int N = 35;
    int K = 37;
    auto A = make_data(M * K, 0.01f);
    auto B = make_data(K * N, 0.02f);
    std::vector<float> naive(M * N), tiled(M * N), expected(M * N);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            expected[m * N + n] = acc;
        }
    }

    float *d_A, *d_B, *d_naive, *d_tiled;
    CHECK_CUDA(cudaMalloc(&d_A, A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_naive, naive.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tiled, tiled.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(kTile, kTile);
    dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_naive, M, N, K);
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_tiled, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(naive.data(), d_naive, naive.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tiled.data(), d_tiled, tiled.size() * sizeof(float), cudaMemcpyDeviceToHost));

    print_error("matmul_naive", naive, expected);
    print_error("matmul_tiled", tiled, expected);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_naive));
    CHECK_CUDA(cudaFree(d_tiled));
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cout << "device: " << prop.name << std::endl;

    demo_vector_add();
    demo_transpose();
    demo_reduction();
    demo_softmax();
    demo_rmsnorm();
    demo_matmul();

    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "CUDA examples finished." << std::endl;
    return 0;
}
