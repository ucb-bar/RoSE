# the first argument is the benchmark name
# the second argument is the number of DMA inflight requests

benchmark=$1
num_dma_inflight_requests=(32)

if [ -z "$benchmark" ]; then
    echo "Benchmark name is required"
    exit 1
fi

chipyard_path=/scratch/iansseijelly/tacit-chipyard
binary_staging_path=/scratch/iansseijelly/test-staging

# first, get the baseline result
echo "Running $benchmark with no tracing"
# run the benchmark
pushd $chipyard_path/software/baremetal-ide
rm -rf build
cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake
cmake --build ./build/ --target $benchmark
cp ./build/examples/embench/$benchmark.elf $binary_staging_path/$benchmark-no-tracing.elf
popd

for num_dma_inflight in ${num_dma_inflight_requests[@]}; do
    echo "Running $benchmark with $num_dma_inflight DMA inflight requests"
    # run the benchmark
    pushd $chipyard_path/software/baremetal-ide
    rm -rf build
    cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -D EMBENCH_ENABLE_TRACE_DMA=1 -D EMBENCH_DMA_INFLIGHT_REQUESTS=$num_dma_inflight
    cmake --build ./build/ --target $benchmark
    cp ./build/examples/embench/$benchmark.elf $binary_staging_path/$benchmark-dma-inflight-requests-$num_dma_inflight.elf
    popd
done

pushd $chipyard_path/sims/vcs
make run-binary CONFIG=TacitRocketRawByteConfig BINARY=$binary_staging_path/$benchmark-no-tracing.elf LOADMEM=1 TIMEOUT_CYCLES=15000000000 &
for num_dma_inflight in ${num_dma_inflight_requests[@]}; do
    make run-binary CONFIG=TacitRocketRawByteConfig BINARY=$binary_staging_path/$benchmark-dma-inflight-requests-$num_dma_inflight.elf LOADMEM=1 TIMEOUT_CYCLES=15000000000 &
done
popd

echo "Waiting for all benchmarks to finish"
wait