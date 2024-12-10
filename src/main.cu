#include <iostream>
#include <chrono>
#include <cuda_fp8.h>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/cublaslt.hpp>
#include <cutf/type.hpp>

template <class T>
std::size_t round_up(const std::size_t n) {
  if (sizeof(T) >= 4) {
    return n;
  }
  const auto ns = 4 / sizeof(T);
  return ((n + ns - 1) / ns) * ns;
}

void print_result(
    const std::size_t m,
    const std::size_t n,
    const std::size_t k,
    const double elapsed_time,
    const std::uint32_t num_tests
    ) {
  const auto num_ops = 2lu * m * n * k * num_tests;
  const auto throughput = num_ops / elapsed_time;
  std::printf("%lu,%lu,%lu,%e\n",
              m, n, k,
              throughput * 1e-12
             );
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

  const OUTPUT_T alpha = 1, beta = 0;
  const auto gemm_func = [&]() {
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
  };
  gemm_func();

  cudaDeviceSynchronize();
  const auto start_closk = std::chrono::system_clock::now();

  for (unsigned i = 0; i < num_tests; i++) {
    gemm_func();
  }

  cudaDeviceSynchronize();
  const auto end_closk = std::chrono::system_clock::now();

  const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_closk - start_closk).count() * 1e-9;
  print_result(m_, n_, k_, elapsed_time, num_tests);
}

template <class INPUT_T, class OUTPUT_T>
void run_cublaslt(
    const std::size_t m_,
    const std::size_t n_,
    const std::size_t k_,
    const unsigned num_tests
    ) {
  const auto m = round_up<INPUT_T>(m_);
  const auto n = round_up<INPUT_T>(n_);
  const auto k = round_up<INPUT_T>(k_);

  auto mat_a_uptr = cutf::memory::get_device_unique_ptr<INPUT_T>(m * k);
  auto mat_b_uptr = cutf::memory::get_device_unique_ptr<INPUT_T>(k * n);
  auto mat_c_uptr = cutf::memory::get_device_unique_ptr<OUTPUT_T>(m * n);

  auto cublaslt_handle = cutf::cublaslt::create_handle_unique_ptr();

  const cublasOperation_t trans_A = CUBLAS_OP_T;
  const cublasOperation_t trans_B = CUBLAS_OP_N;

  auto a_desc_uptr = cutf::cublaslt::create_matrix_layout_uptr(
      k_, m_, k, mat_a_uptr.get()
      );
  auto b_desc_uptr = cutf::cublaslt::create_matrix_layout_uptr(
      k_, n_, k, mat_b_uptr.get()
      );
  auto c_desc_uptr = cutf::cublaslt::create_matrix_layout_uptr(
      m_, n_, m, mat_c_uptr.get()
      );

  auto cublaslt_op_desc = cutf::cublaslt::create_matmul_desc_unique_ptr(
      CUBLAS_COMPUTE_32F,
      cutf::type::get_data_type<OUTPUT_T>()
      );
  CUTF_CHECK_ERROR(cublasLtMatmulDescSetAttribute(*cublaslt_op_desc.get(), CUBLASLT_MATMUL_DESC_TRANSA, &trans_A, sizeof(trans_A)));
  CUTF_CHECK_ERROR(cublasLtMatmulDescSetAttribute(*cublaslt_op_desc.get(), CUBLASLT_MATMUL_DESC_TRANSB, &trans_B, sizeof(trans_B)));


  auto cublaslt_preference_uptr = cutf::cublaslt::create_preference_unique_ptr();
  const std::size_t worksize = 4lu << 20;
  CUTF_CHECK_ERROR(cublasLtMatmulPreferenceSetAttribute(
          *cublaslt_preference_uptr.get(),
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &worksize,
          sizeof(worksize)
          ));

  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  CUTF_CHECK_ERROR(cublasLtMatmulAlgoGetHeuristic(
          *cublaslt_handle.get(),
          *cublaslt_op_desc.get(),
          *a_desc_uptr.get(),
          *b_desc_uptr.get(),
          *c_desc_uptr.get(),
          *c_desc_uptr.get(),
          *cublaslt_preference_uptr.get(),
          1,
          &heuristic_result,
          &returned_results
          ));
  if (returned_results == 0) {
    std::fprintf(stderr, "Error in cublasLtMatmulAlgoGetHeuristic\n");
    return;
  }

  auto workspace_uptr = cutf::memory::get_device_unique_ptr<std::uint8_t>(worksize);


  const OUTPUT_T alpha = 1, beta = 0;
  const auto gemm_func = [&]() {
    CUTF_CHECK_ERROR(cublasLtMatmul(
            *cublaslt_handle.get(),
            *cublaslt_op_desc.get(),
            &alpha,
            mat_a_uptr.get(), *a_desc_uptr.get(),
            mat_b_uptr.get(), *b_desc_uptr.get(),
            &beta,
            mat_c_uptr.get(), *c_desc_uptr.get(),
            mat_c_uptr.get(), *c_desc_uptr.get(),
            &heuristic_result.algo,
            workspace_uptr.get(),
            worksize,
            0
            ));
  };
  gemm_func();

  cudaDeviceSynchronize();
  const auto start_closk = std::chrono::system_clock::now();

  for (unsigned i = 0; i < num_tests; i++) {
    gemm_func();
  }

  cudaDeviceSynchronize();
  const auto end_closk = std::chrono::system_clock::now();

  const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_closk - start_closk).count() * 1e-9;
  print_result(m_, n_, k_, elapsed_time, num_tests);
}

int main(int argc, char **argv) {
  if (argc < 1 + 1 + 3 + 1) {
    std::fprintf(stderr, "Usage: %s [mode] [m] [n] [k] [test_count]\n", argv[0]);
    std::fprintf(stderr, "- mode : I8I32 F16F32 E4M3F32\n");
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
    run_cublaslt<half, float>(m, n, k, num_tests);
  //} else if (mode == "E5M2F32") {
  //  run_cublaslt<__nv_fp8_e5m2, float>(m, n, k, num_tests);
  } else if (mode == "E4M3F32") {
    run_cublaslt<__nv_fp8_e4m3, float>(m, n, k, num_tests);
  }
}
