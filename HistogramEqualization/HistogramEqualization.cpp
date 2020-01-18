// HistogramEquilization.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include "pch.h"
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define _CRT_SECURE_NO_WARNINGS

#include "FreeImage.h"
#include <CL/cl.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <time.h>


#define GRAYSCALE 256

#define MAX_SOURCE_SIZE	16384
#define WORKGROUP_DIM (16)

#define WORKGROUP_DIM_2 (256)

//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

void HistogramGPUlocal(unsigned char *imageIn, unsigned long *histogram, int width, int height);
void printHistogram(unsigned long *histogram);

void cdfCPU(unsigned long *histogram, unsigned long* cdf);
void cdfGPU(unsigned long *histogram, unsigned long* cdf);
void printHistogram(unsigned int *histogram);

int main(void)
{
	clock_t start, end;
	double timeGPUlocal;

	FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "bad.png", 0);
	FIBITMAP *imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);
	int pitch = FreeImage_GetPitch(imageBitmapGrey);

	unsigned char *imageIn = (unsigned char *)malloc(height*width * sizeof(unsigned char));
	unsigned long *histogram = (unsigned long *)malloc(GRAYSCALE * sizeof(unsigned long));
	unsigned long *cdf = (unsigned long *)malloc(GRAYSCALE * sizeof(unsigned long));



	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);

	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);

	printf("Histogram GPUlocal: \n");
	start = clock();
	HistogramGPUlocal(imageIn, histogram, width, height);
	end = clock();

	timeGPUlocal = (double)(end - start) / CLOCKS_PER_SEC;
	printHistogram(histogram);
	printf("\n");

	printf("HistogramLocal time: %f\n", timeGPUlocal);

	//cdfGPU(histogram, cdf, width, height);
	//cdfCPU(histogram, cdf2);

	//printHistogram(cdf2);

	cdfGPU(histogram, cdf);
	printHistogram(cdf);

	free(imageIn);
	free(histogram);

	return 0;
}

void HistogramGPUlocal(unsigned char *imageIn, unsigned long *histogram, int width, int height) {
	memset(histogram, 0, GRAYSCALE * sizeof(unsigned long));

	cl_int ret;

	// Branje datoteke
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("kernel.cl", "r");
	if (!fp)
	{
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose(fp);

	// Podatki o platformi
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;

	// Delali bomo s platform_id[0] na GPU
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
		device_id, &ret_num_devices);

	// Kontekst
	cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

	// Ukazna vrsta
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);


	// Priprava programa
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
		NULL, &ret);

	// Prevajanje
	ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

	// Log
	size_t build_log_len;
	char *build_log;
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
		0, NULL, &build_log_len);
	build_log = (char *)malloc(sizeof(char)*(build_log_len + 1));
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
		build_log_len, build_log, NULL);
	printf("%s\n", build_log);
	free(build_log);

	// Priprava kernela
	cl_kernel kernel = clCreateKernel(program, "localKernel", &ret);

	// DELITEV DELA
	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);
	//printf("veckratnik niti = %d\n", buf_size_t);

	size_t global_size[2] = { width, height };
	size_t local_size[2] = { WORKGROUP_DIM , WORKGROUP_DIM };

	// Alokacija pomnilnika na napravi
	cl_mem imgIn_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * sizeof(unsigned char), imageIn, &ret);

	cl_mem resultOut_mem_object = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, GRAYSCALE * sizeof(unsigned long), histogram, &ret);


	// Argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&imgIn_mem_object);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&resultOut_mem_object);



	// Zagon kernela
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		global_size, local_size, 0, NULL, NULL);

	// Kopiranje rezultatov (branje v pomnilnik iz naparave, 0 = offset(
	ret = clEnqueueReadBuffer(command_queue, resultOut_mem_object, CL_FALSE, 0,
		GRAYSCALE * sizeof(unsigned long), histogram, 0, NULL, NULL);

	// Ciscenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(imgIn_mem_object);
	ret = clReleaseMemObject(resultOut_mem_object);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}

void cdfCPU(unsigned long *histogram, unsigned long* cdf) {
	for (int i = 0; i < GRAYSCALE; i++) {
		cdf[i] = 0;
	}

	cdf[0] = histogram[0];
	for (int i = 1; i < GRAYSCALE; i++) {
		cdf[i] = cdf[i - 1] + histogram[i];
	}
}


void cdfGPU(unsigned long *histogram, unsigned long *cdf) {
	memset(cdf, 0L, GRAYSCALE * sizeof(unsigned long));

	cl_int ret;

	// Branje datoteke
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("kernel.cl", "r");
	if (!fp)
	{
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose(fp);

	// Podatki o platformi
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;

	// Delali bomo s platform_id[0] na GPU
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
		device_id, &ret_num_devices);

	// Kontekst
	cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

	// Ukazna vrsta
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);


	// Priprava programa
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
		NULL, &ret);

	// Prevajanje
	ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

	// Log
	size_t build_log_len;
	char *build_log;
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
		0, NULL, &build_log_len);
	build_log = (char *)malloc(sizeof(char)*(build_log_len + 1));
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
		build_log_len, build_log, NULL);
	printf("%s\n", build_log);
	free(build_log);

	// Priprava kernela
	cl_kernel kernel = clCreateKernel(program, "cdf", &ret);

	// DELITEV DELA
	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);

	size_t global_size[1] = { GRAYSCALE };
	size_t local_size[1] = { GRAYSCALE };

	//unsigned int *test = (unsigned int *)malloc(GRAYSCALE * sizeof(unsigned int));

	// Alokacija pomnilnika na napravi
	cl_mem histogram_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, GRAYSCALE * sizeof(unsigned long), histogram, &ret);
	//cl_mem current_mem_object = clCreateBuffer(context, CL_MEM_READ_WRITE, GRAYSCALE * sizeof(unsigned long), NULL, &ret);
	//cl_mem last_mem_object = clCreateBuffer(context, CL_MEM_READ_WRITE, GRAYSCALE * sizeof(unsigned long), NULL, &ret);
	cl_mem result_mem_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, GRAYSCALE * sizeof(unsigned long), cdf, &ret);

	// Argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&histogram_mem_object);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&result_mem_object);
	//ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&current_mem_object);
	//	ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&last_mem_object);

	// Zagon kernela
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		global_size, local_size, 0, NULL, NULL);

	// Kopiranje rezultatov (branje v pomnilnik iz naparave, 0 = offset(
	ret = clEnqueueReadBuffer(command_queue, result_mem_object, CL_FALSE, 0,
		GRAYSCALE * sizeof(unsigned long), cdf, 0, NULL, NULL);

	// Ciscenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(histogram_mem_object);
	ret = clReleaseMemObject(result_mem_object);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	//printHistogram(test);
}

void printHistogram(unsigned long *histogram) {
	printf("Barva\tPojavitve\n");
	for (int i = 0; i < GRAYSCALE; i++) {
		printf("%d\t%lu\n", i, histogram[i]);
	}
}

void printHistogram(unsigned int *histogram) {
	printf("Barva\tPojavitve\n");
	for (int i = 0; i < GRAYSCALE; i++) {
		printf("%d\t%d\n", i, histogram[i]);
	}
}