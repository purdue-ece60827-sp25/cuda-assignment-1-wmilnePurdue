
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        y[i] += scale * x[i];
    }
}

int runGpuSaxpy(int vectorSize) {

    if (vectorSize == 0) {
        std::cerr << "Error: vectorSize cannot be 0" << std::endl;
        return -1; 
    }

    float *x, *y, *d_x, *d_y;
    float scale;

    // Host memory allocation
    x = (float*)malloc(vectorSize * sizeof(float));
    y = (float*)malloc(vectorSize * sizeof(float));

    // Device memory allocation
    cudaError_t err_x = cudaMalloc(&d_x, vectorSize * sizeof(float));
    cudaError_t err_y = cudaMalloc(&d_y, vectorSize * sizeof(float));
    gpuAssert(err_x, __FILE__, __LINE__, true);
    gpuAssert(err_y, __FILE__, __LINE__, true);


    // Initialize vectors and scale on the host
    vectorInit(x, vectorSize);
    vectorInit(y, vectorSize);
    std::random_device rd;  // Create a new random device each time
    std::mt19937 gen(rd()); // Create and seed a new engine each time
    std::uniform_real_distribution<> distrib(MIN_VAL, MAX_VAL); // The distribution    
    scale = (float)distrib(gen);

    // Decided to use casting method posted by TA - Convert to integers and back to floats
    for (int i = 0; i < vectorSize; ++i) {
        x[i] = (float)(int)x[i]; 
        y[i] = (float)(int)y[i]; 
    }
    scale = (float)(int)scale;  

    //copy for CPU verification
    float* y_cpu = (float*)malloc(vectorSize * sizeof(float));
    memcpy(y_cpu, y, vectorSize * sizeof(float));

	if (x == NULL || y == NULL || y_cpu == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	#ifndef DEBUG_PRINT_DISABLE 
        std::cout << "\n Adding vectors : \n";
        std::cout << " scale = " << std::fixed << scale << "\n"; 
        std::cout << " x = { ";
        for (int i = 0; i < std::min(3, vectorSize); ++i) {
            std::cout << std::fixed << x[i] << ", "; 
        }
        std::cout << " ... }\n";
        std::cout << " y = { ";
        for (int i = 0; i < std::min(3, vectorSize); ++i) {
            std::cout << std::fixed << y[i] << ", "; 
        }
        std::cout << " ... }\n";
	#endif    

    // Query device properties and determine block size
    int deviceId = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    //int blockSize = std::min(maxThreadsPerBlock, 256);
    int blockSize = maxThreadsPerBlock;

    // Determine number of blocks needed
    int numBlocks = (vectorSize + blockSize - 1) / blockSize;

    // Check grid dimensions
    int maxGridDimX = prop.maxGridSize[0]; // Access max grid dim - X

    if (numBlocks > maxGridDimX) {
        std::cerr << "Error: Number of blocks exceeds maximum grid dimension" << std::endl;
        // Free allocated memory before returning error
        free(x);
        free(y);
        free(y_cpu);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1; // Return error code
    }

    // Copy data to device
    gpuAssert(cudaMemcpy(d_x, x, vectorSize * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__, true);
    gpuAssert(cudaMemcpy(d_y, y, vectorSize * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__, true);

    // Kernel launch
    saxpy_gpu<<<numBlocks, blockSize>>>(d_x, d_y, scale, vectorSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__, true);

    // Copy results back to host
    gpuAssert(cudaMemcpy(y, d_y, vectorSize * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__, true);

    int errorCount = verifyVector(x, y_cpu, y, scale, vectorSize);

    #ifndef DEBUG_PRINT_DISABLE  
        std::cout << "\nGPU and CPU Results:\n";
        std::cout << "y (GPU) = {";
        for (int i = 0; i < std::min(3, vectorSize); ++i) {
            std::cout << std::fixed << y[i] << ", ";
        }
        std::cout << " ... }\n";
        std::cout << "Found " << errorCount << " / " << vectorSize << " possible errors \n";
    #endif

    // Free memory
    free(x);
    free(y);
    free(y_cpu);
    gpuAssert(cudaFree(d_x), __FILE__, __LINE__, true);
    gpuAssert(cudaFree(d_y), __FILE__, __LINE__, true);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t rng;
    curand_init(clock64(), threadId, 0, &rng);

    uint64_t hitCount = 0;
    for (uint64_t i = 0; i < sampleSize; ++i) {
        float x = curand_uniform(&rng);
        float y = curand_uniform(&rng);
        if (x * x + y * y < 1) {
            ++hitCount;
        }
    }
    pSums[threadId] = hitCount;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = threadId * reduceSize;
    uint64_t end = min(start + reduceSize, pSumSize);

    uint64_t sum = 0;
    for (uint64_t i = start; i < end; ++i) {
        sum += pSums[i];
    }
    totals[threadId] = sum;
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;
    uint64_t *d_pSums, *d_totals;

    gpuAssert(cudaMalloc(&d_pSums, generateThreadCount * sizeof(uint64_t)), __FILE__, __LINE__, true);
    gpuAssert(cudaMalloc(&d_totals, reduceThreadCount * sizeof(uint64_t)), __FILE__, __LINE__, true);

    dim3 blockDim(256); 
    dim3 gridDimGenerate((generateThreadCount + blockDim.x - 1) / blockDim.x);
    dim3 gridDimReduce((reduceThreadCount + blockDim.x - 1) / blockDim.x);

    generatePoints<<<gridDimGenerate, blockDim>>>(d_pSums, generateThreadCount, sampleSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__, true);

    reduceCounts<<<gridDimReduce, blockDim>>>(d_pSums, d_totals, generateThreadCount, reduceSize);
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__, true);

    uint64_t *h_totals = new uint64_t[reduceThreadCount];
    gpuAssert(cudaMemcpy(h_totals, d_totals, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost), __FILE__, __LINE__, true);

    uint64_t totalHits = 0;
    for (uint64_t i = 0; i < reduceThreadCount; ++i) {
        totalHits += h_totals[i];
    }

    approxPi = (double)totalHits / (generateThreadCount * sampleSize) * 4.0;

    delete[] h_totals;
    gpuAssert(cudaFree(d_pSums), __FILE__, __LINE__, true);
    gpuAssert(cudaFree(d_totals), __FILE__, __LINE__, true);

    return approxPi;
}
