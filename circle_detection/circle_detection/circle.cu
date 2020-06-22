// std::system includes
#include <memory>
#include <iostream>
#include <stdio.h>
#include <time.h>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

//#include <helper_cuda.h>

#define threads 2048*2048
#define threads_per_block 1024
#define num_vert 629

/* GPU Device kernel for an individual Ransac interation. 
   d_centerx: output array of x coordinates of circle centers for each executing thread.
   d_centery: output array of y coordinates of circle centers for each executing thread.
   d_centerrad: output array of radius lengths for each executing thread.
   d_eval: output array of evaluation of each thread's circle, i.e: how many points does it go across.
   d_vert_x: input array of the x coordinates of the given contour points.
   d_vert_y: input array of the y cooridnates of the given contour points.
   d_randids: randomized indices that each thread uses for their own calculations.
   d_n: number of contour points.
*/
__global__ void ransac_circle(int* d_centerx, int* d_centery, float* d_centerrad, int* d_eval, int* d_vert_x, int* d_vert_y, int* d_randinds, int d_n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;	int i1 = d_randinds[index];
	int i2 = d_randinds[index + 1];
	int i3 = d_randinds[index + 2];
	int x1 = d_vert_x[i1];
	int x2 = d_vert_x[i2];
	int x3 = d_vert_x[i3];
	int y1 = d_vert_y[i1];
	int y2 = d_vert_y[i2];
	int y3 = d_vert_y[i3];

	/*a b c represent the variable independent coefficients in the circle equation (that only depend on circle center and radius).*/
	int a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2;
	int b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (x3 * x3 + y3 * y3) * (y2 - y1);
	int c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (x3 * x3 + y3 * y3) * (x1 - x2);

	/*these a b c variables are now used to get the original circle parameters back from the circle equation coefficients.*/
	float x = -b / (2 * a);
	float y = -c / (2 * a);
	float r = sqrt((float)((x - x1) * (x - x1) + (y - y1) * (y - y1)));
	d_centerx[index] = (int)x;
	d_centery[index] = (int)y;
	d_centerrad[index] = r;

	/*calculate the fourth coefficient that was unused thus far, and count how many points aline.*/
	int d = (int)((x * x) + (y * y) - ((x - x1) * (x - x1) + (y - y1) * (y - y1)));
	int count = 0;
	for (int i = 0; i < d_n; i++) {
		int x_i = d_vert_x[i];
		int y_i = d_vert_y[i];
		if (a * (x_i * x_i + y_i * y_i) + b * x + c * y + d == 0) count++;
	}
	d_eval[index] = count;

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	/* Parse the input file and fill up our host-side coordinate arrays. */
	char const* const inputFile = argv[1];
	FILE* file = fopen(inputFile, "r");
	int vert_x[num_vert];
	int vert_y[num_vert];

	char line[256];
	int i = 0;

	while (fgets(line, sizeof(line), file) != NULL) {
		sscanf(line, "%d %d", &vert_x[i], &vert_y[i]);
		++i;
	}

	fclose(file);

	/* Perform a sanity-check for the input parsing as a form of quick testing. */
	srand(time(NULL));
	int rand_i = rand() % num_vert;
	printf("Coordinates on random index %d are (%d,%d)\n", rand_i, vert_x[rand_i], vert_y[rand_i]);
	printf("Coordinates on first index are (%d,%d)\n", vert_x[0], vert_y[0]);
	printf("Coordinates on last index are (%d,%d)\n", vert_x[num_vert - 1], vert_y[num_vert - 1]);

	/* Get available compatible GPU device information. */
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;



		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: %s \n", dev, deviceProp.name);
		printf("\nMaxThreadsPerBlock: %d \n", deviceProp.maxThreadsPerBlock);
		printf("\nMaxThreadDim (%d,%d,%d)  \n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("\nMaxGridSize (%d,%d,%d)  \n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

	}

	/* Initialize and allocate the host side variables */

	int ts = threads * sizeof(int);
	int fs = threads * sizeof(float);

	int* h_centerx = (int*)malloc(ts);
	int* h_centery = (int*)malloc(ts);
	float* h_centerrad = (float*)malloc(fs);
	int* h_eval = (int*)malloc(ts);
	int* h_randinds = (int*)malloc(ts * 3);

	/* Initialize and allocate the device side variables. */

	int* d_centerx;
	int* d_centery;
	float* d_centerrad;
	int* d_eval;
	int* d_vert_x;
	int* d_vert_y;
	int* d_randinds;

	cudaMalloc((void**)& d_centerx, ts);
	cudaMalloc((void**)& d_centery, ts);
	cudaMalloc((void**)& d_centerrad, fs);
	cudaMalloc((void**)& d_eval, ts);
	cudaMalloc((void**)& d_vert_x, num_vert);
	cudaMalloc((void**)& d_vert_y, num_vert);
	cudaMalloc((void**)& d_randinds, ts * 3);

	/* Generate the random indices that each thread will work with. Device code has no access to the rand() function so this is a workaround. */

	for (int i = 0; i < threads; ++i) {
		int i1 = rand() % num_vert;
		int i2 = rand() % num_vert;
		int i3 = rand() % num_vert;
		while (i2 == i1) {
			i2 = rand() % num_vert;
		}
		while ((i3 == i1) || (i3 == i2)) {
			i3 = rand() % num_vert;
		}
		h_randinds[i * 3] = i1;
		h_randinds[i * 3 + 1] = i2;
		h_randinds[i * 3 + 2] = i3;
	}

	/* Pass the input parameters from host to device. */
	cudaMemcpy(d_vert_x, vert_x, num_vert, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vert_y, vert_y, num_vert, cudaMemcpyHostToDevice);
	cudaMemcpy(d_randinds, h_randinds, (ts * 3), cudaMemcpyHostToDevice);

	/* Run RANSAC. */

	ransac_circle << <threads / threads_per_block, threads_per_block >> > (d_centerx, d_centery, d_centerrad, d_eval, d_vert_x, d_vert_y, d_randinds, num_vert);

	/* Extract the values from the corresponding device arrays. */

	cudaMemcpy(h_centerx, d_centerx, ts, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_centery, d_centery, ts, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_centerrad, d_centerrad, fs, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_eval, d_eval, ts, cudaMemcpyDeviceToHost);

	/* Look for the best thread result. */

	int best_eval = h_eval[0];
	int best_eval_index = 0;
	for (int i = 1; i < threads; i++) {
		if (h_eval[i] > best_eval) {
			best_eval = h_eval[i];
			best_eval_index = i;
		}
	}

	printf("The best fitting circle via RANSAC has the center (%d,%d) with radius %0.3f found in thread #%d which goes through %d points", h_centerx[best_eval_index], h_centery[best_eval_index], h_centerrad[best_eval_index], best_eval_index, best_eval);

	cudaFree(d_vert_x);
	cudaFree(d_vert_y);
	cudaFree(d_randinds);
	cudaFree(d_centerx);
	cudaFree(d_centery);
	cudaFree(d_centerrad);
	cudaFree(d_eval);

	// finish
	// cudaDeviceReset causes the driver to clean up all state. While 
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	char ch;
	std::cin >> ch;

	exit(EXIT_SUCCESS);
}