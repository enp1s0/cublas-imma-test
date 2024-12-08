#include <iostream>
#include <chrono>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>

template <class T>
std::size_t round_up(const std::size_t n) {
  if (sizeof(T) >= 4) {
    return n;
  }
  const auto ns = 4 / sizeof(T);
  return ((n + ns - 1) / ns) * ns;
}

template <class INPUT_T, class OUTPUT_T>
void run_cublas(
    const std::size_t m_,
    const std::size_t n_,
    const std::size_t k_,
    const unsigned num_tests
    ) {
  auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

  const auto m = round_up<INPUT_T>(m_);
  const auto n = round_up<INPUT_T>(n_);
  const auto k = round_up<INPUT_T>(k_);

  auto mat_a_uptr = cutf::memory::get_device_unique_ptr<INPUT_T>(m * k);
  auto mat_b_uptr = cutf::memory::get_device_unique_ptr<INPUT_T>(k * n);
  auto mat_c_uptr = cutf::memory::get_device_unique_ptr<OUTPUT_T>(m * n);

  cublasComputeType_t compute_type;
  cudaDataType_t input_data_type, output_data_type;
  if (std::is_same<INPUT_T, std::int8_t>::value && std::is_same<OUTPUT_T, std::int32_t>::value) {
    compute_type = CUBLAS_COMPUTE_32I;
    input_data_type = CUDA_R_8I;
    output_data_type = CUDA_R_32I;
  }
  if (std::is_same<INPUT_T, half       >::value && std::is_same<OUTPUT_T, float       >::value) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    input_data_type = CUDA_R_16F;
    output_data_type = CUDA_R_32F;
  }

  cudaDeviceSynchronize();
  const auto start_closk = std::chrono::system_clock::now();

  for (unsigned i = 0; i < num_tests; i++) {
    std::int32_t alpha = 1, beta = 0;
    CUTF_CHECK_ERROR(cublasGemmEx(
            *cublas_handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            mat_a_uptr.get(), input_data_type, k,
            mat_b_uptr.get(), input_data_type, k,
            &beta,
            mat_c_uptr.get(), output_data_type, m,
            compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
  }

  cudaDeviceSynchronize();
  const auto end_closk = std::chrono::system_clock::now();

  const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_closk - start_closk).count() * 1e-9;
  const auto num_ops = 2lu * m * n * k * num_tests;
  const auto throughput = num_ops / elapsed_time;
  std::printf("%lu,%lu,%lu,%e\n",
              m_, n_, k_,
              throughput * 1e-12
             );
}

int main(int argc, char **argv) {
  if (argc < 1 + 1 + 3 + 1) {
    std::fprintf(stderr, "Usage: %s [mode] [m] [n] [k] [test_count]\n", argv[0]);
    std::fprintf(stderr, "- mode : I8I32 F16F32\n");
    return 1;
  }
  const std::string mode = argv[1];
  const auto m = std::stoul(argv[2]);
  const auto n = std::stoul(argv[3]);
  const auto k = std::stoul(argv[4]);
  const auto num_tests  = std::stoul(argv[5]);

  std::printf("%s,", mode.c_str());
  if (mode == "I8I32") {
    run_cublas<signed char, int>(m, n, k, num_tests);
  } else if (mode == "F16F32") {
    run_cublas<half, float>(m, n, k, num_tests);
  }
}
