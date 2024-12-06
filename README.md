# cuBLAS IMMA performance benchmark

## Build
```bash
git clone https://github.com/enp1s0/cublas-imma-test --recursive
cd cublas-imma-test
make -j
```

## Usage
```bash
cublas-imma.test [mode] [m] [n] [k] [test_count]
  - mode : I8I32 F16F32

# e.g.
./cublas-imma.test I8I32 2048 1024 1024 10
```
