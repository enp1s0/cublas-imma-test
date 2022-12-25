#include <iostream>
#include <chrono>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>

void run(
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const unsigned num_tests
		) {
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	auto mat_a_uptr = cutf::memory::get_device_unique_ptr<std::int8_t>(m * k);
	auto mat_b_uptr = cutf::memory::get_device_unique_ptr<std::int8_t>(k * n);
	auto mat_c_uptr = cutf::memory::get_device_unique_ptr<std::int32_t>(m * n);

	cudaDeviceSynchronize();
	const auto start_closk = std::chrono::system_clock::now();

	for (unsigned i = 0; i < num_tests; i++) {
		std::int32_t alpha = 1, beta = 0;
		CUTF_CHECK_ERROR(cublasGemmEx(
					*cublas_handle.get(),
					CUBLAS_OP_T, CUBLAS_OP_N,
					m, n, k,
					&alpha,
					mat_a_uptr.get(), CUDA_R_8I, k,
					mat_b_uptr.get(), CUDA_R_8I, k,
					&beta,
					mat_c_uptr.get(), CUDA_R_32I, m,
					CUBLAS_COMPUTE_32F,
					CUBLAS_GEMM_DEFAULT_TENSOR_OP
					));
	}

	cudaDeviceSynchronize();
	const auto end_closk = std::chrono::system_clock::now();

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_closk - start_closk).count() * 1e-9;
	const auto num_ops = 2lu * m * n * k * num_tests;
	const auto throughput = num_ops / elapsed_time;
	std::printf("%lu,%lu,%lu,%e\n",
			m, n, k,
			throughput * 1e-12
			);
}

int main(int argc, char **argv) {
	if (argc < 1 + 3 + 1) {
		std::fprintf(stderr, "Usage: %s [m] [n] [k] [test_count]\n", argv[0]);
	}
	const auto m = std::stoul(argv[1]);
	const auto n = std::stoul(argv[2]);
	const auto k = std::stoul(argv[3]);
	const auto num_tests  = std::stoul(argv[4]);

	run(m, n, k, num_tests);
}
