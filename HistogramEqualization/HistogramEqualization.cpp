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


#define GRAYLEVELS 256

#define MAX_SOURCE_SIZE	16384
#define WORKGROUP_DIM (16)

//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

void HistogramGPUlocal(unsigned char *imageIn, unsigned long *histogram, int width, int height);
void cdfGPU(unsigned long *histogram, unsigned long* cdf);
void minGPU(unsigned long *cdf, unsigned long *min);
void equalizeGPU(unsigned char *image, unsigned char *imageOut,unsigned long *cdf, int width, int height, long minCdf);

void cdfCPU(unsigned long *histogram, unsigned long* cdf);
void HistogramCPU(unsigned char *imageIn, unsigned long *histogram, int width, int height);
unsigned long minCPU(unsigned long * cdf);
void equalizeCPU(unsigned char *imageIn, unsigned char *imageOut, unsigned long *cdf, int width, int height, long minCdf);

void printHistogram(unsigned int *histogram);
void printHistogramL(unsigned long *histogram);

int main(void)
{
	FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "bad2.png", 0);
	FIBITMAP *imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);

	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);
	int pitch = FreeImage_GetPitch(imageBitmapGrey);

	unsigned char *imageIn = (unsigned char *)malloc(height*width * sizeof(unsigned char));
	unsigned char *imageOut = (unsigned char *)malloc(height*width * sizeof(unsigned char));

	unsigned long *histogram = (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));
	unsigned long *cdf = (unsigned long *)malloc(GRAYLEVELS * sizeof(unsigned long));

	memset(histogram, 0, GRAYLEVELS * sizeof(unsigned long));
	memset(cdf, 0, GRAYLEVELS * sizeof(unsigned long));

	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);

	unsigned long min;

	//CPU Equalization
	
	HistogramCPU(imageIn, histogram, width, height);
	cdfCPU(histogram, cdf);
	minCPU(cdf);
	min = minCPU(cdf);
	equalizeCPU(imageIn, imageOut, cdf, width, height, min);

	FIBITMAP *cpuBitmap = FreeImage_ConvertFromRawBits(imageOut, width, height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Save(FIF_PNG, cpuBitmap, "cpu.png", 0);
	FreeImage_Unload(cpuBitmap);

	
	memset(histogram, 0, GRAYLEVELS * sizeof(unsigned long));
	memset(cdf, 0, GRAYLEVELS * sizeof(unsigned long));
	memset(imageOut, 0, width * height * sizeof(unsigned char));

	//GPU Equalization
	HistogramGPUlocal(imageIn, histogram, width, height);
	cdfGPU(histogram, cdf);
	minGPU(cdf, &min);
	equalizeGPU(imageIn,imageOut,cdf,width,height,min);

	FIBITMAP *gpuBitmap = FreeImage_ConvertFromRawBits(imageOut, width, height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Save(FIF_PNG, gpuBitmap, "gpu.png", 0);
	FreeImage_Unload(gpuBitmap);


	free(histogram);
	free(cdf);
	free(imageIn);
	free(imageOut);

	return 0;
}

void HistogramGPUlocal(unsigned char *imageIn, unsigned long *histogram, int width, int height) {
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

	int we = ceil(width / (float)WORKGROUP_DIM)*WORKGROUP_DIM;
	int he = ceil(height / (float)WORKGROUP_DIM)*WORKGROUP_DIM;

	size_t global_size[2] = { we,he };
	size_t local_size[2] = { WORKGROUP_DIM,WORKGROUP_DIM };

	// Alokacija pomnilnika na napravi
	cl_mem imgIn_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * sizeof(unsigned char), imageIn, &ret);

	cl_mem resultOut_mem_object = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, GRAYLEVELS * sizeof(unsigned long), histogram, &ret);


	// Argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&imgIn_mem_object);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&resultOut_mem_object);
	ret |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &width);
	ret |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &height);

	// Zagon kernela
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		global_size, local_size, 0, NULL, NULL);

	// Kopiranje rezultatov (branje v pomnilnik iz naparave, 0 = offset(
	ret = clEnqueueReadBuffer(command_queue, resultOut_mem_object, CL_FALSE, 0,
		GRAYLEVELS * sizeof(unsigned long), histogram, 0, NULL, NULL);

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

void cdfGPU(unsigned long *histogram, unsigned long *cdf) {
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

	size_t global_size[1] = { GRAYLEVELS };
	size_t local_size[1] = { GRAYLEVELS };


	// Alokacija pomnilnika na napravi
	cl_mem histogram_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, GRAYLEVELS * sizeof(unsigned long), histogram, &ret);
	cl_mem result_mem_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, GRAYLEVELS * sizeof(unsigned long), cdf, &ret);

	// Argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&histogram_mem_object);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (unsigned long *)&result_mem_object);

	// Zagon kernela
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		global_size, local_size, 0, NULL, NULL);

	// Kopiranje rezultatov (branje v pomnilnik iz naparave, 0 = offset(
	ret = clEnqueueReadBuffer(command_queue, result_mem_object, CL_FALSE, 0,
		GRAYLEVELS * sizeof(unsigned long), cdf, 0, NULL, NULL);

	// Ciscenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(histogram_mem_object);
	ret = clReleaseMemObject(result_mem_object);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}

void minGPU(unsigned long *cdf, unsigned long *min) {
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
	cl_kernel kernel = clCreateKernel(program, "minCdf", &ret);

	// DELITEV DELA
	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);

	size_t global_size[1] = { GRAYLEVELS/2 };
	size_t local_size[1] = { GRAYLEVELS/2 };


	// Alokacija pomnilnika na napravi
	cl_mem histogram_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, GRAYLEVELS * sizeof(unsigned long), cdf, &ret);
	cl_mem result_mem_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned long), NULL, &ret);

	// Argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&histogram_mem_object);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&result_mem_object);

	// Zagon kernela
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		global_size, local_size, 0, NULL, NULL);

	// Kopiranje rezultatov (branje v pomnilnik iz naparave, 0 = offset(
	ret = clEnqueueReadBuffer(command_queue, result_mem_object, CL_FALSE, 0, sizeof(unsigned long), min, 0, NULL, NULL);

	// Ciscenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(histogram_mem_object);
	ret = clReleaseMemObject(result_mem_object);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}

void equalizeGPU(unsigned char *imageIn, unsigned char *imageOut ,unsigned long *cdf, int width, int height, long minCdf) {
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
	cl_kernel kernel = clCreateKernel(program, "equalize", &ret);

	// DELITEV DELA
	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);

	int we = ceil(width / (float)WORKGROUP_DIM)*WORKGROUP_DIM;
	int he = ceil(height / (float)WORKGROUP_DIM)*WORKGROUP_DIM;

	size_t global_size[2] = {we,he};
	size_t local_size[2] = {WORKGROUP_DIM,WORKGROUP_DIM};

	// Alokacija pomnilnika na napravi
	cl_mem imageIn_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, width * height * sizeof(unsigned char), imageIn, &ret);
	cl_mem imageOut_mem_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, &ret);

	cl_mem cdf_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, GRAYLEVELS * sizeof(unsigned long), cdf, &ret);

	// Argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&imageIn_mem_object);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&imageOut_mem_object);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cdf_mem_object);
	ret |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &width);
	ret |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &height);
	ret |= clSetKernelArg(kernel, 5, sizeof(unsigned long), &minCdf);


	// Zagon kernela
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
		global_size, local_size, 0, NULL, NULL);

	// Kopiranje rezultatov (branje v pomnilnik iz naparave, 0 = offset(
	ret = clEnqueueReadBuffer(command_queue, imageOut_mem_object, CL_FALSE, 0, width * height * sizeof(unsigned char), imageOut, 0, NULL, NULL);

	// Ciscenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(imageIn_mem_object);
	ret = clReleaseMemObject(imageOut_mem_object);
	ret = clReleaseMemObject(cdf_mem_object);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}

void printHistogramL(unsigned long *histogram) {
	printf("Barva\tPojavitve\n");
	for (int i = 0; i < GRAYLEVELS; i++) {
		printf("%d\t%lu\n", i, histogram[i]);
	}
}

void printHistogram(unsigned int *histogram) {
	printf("Barva\tPojavitve\n");
	for (int i = 0; i < GRAYLEVELS; i++) {
		printf("%d\t%d\n", i, histogram[i]);
	}
}

void HistogramCPU(unsigned char *imageIn, unsigned long *histogram, int width, int height)
{
	//za vsak piksel v sliki
	for (int i = 0; i < (height); i++)
		for (int j = 0; j < (width); j++)
		{
			histogram[imageIn[i*width + j]]++;
		}
}

void cdfCPU(unsigned long *histogram, unsigned long* cdf) {
	for (int i = 0; i < GRAYLEVELS; i++) {
		cdf[i] = 0;
	}

	cdf[0] = histogram[0];
	for (int i = 1; i < GRAYLEVELS; i++) {
		cdf[i] = cdf[i - 1] + histogram[i];
	}
}

unsigned long minCPU(unsigned long * cdf) {

	unsigned long min = 0;
	for (int i = 0; min == 0 && i < GRAYLEVELS; i++) {
		min = cdf[i];
	}

	return min;
}

void equalizeCPU(unsigned char *imageIn, unsigned char *imageOut, unsigned long *cdf, int width, int height, long minCdf) {
	unsigned long imageSize = width * height;
	
	float scale;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			scale = (float)(cdf[imageIn[i*width + j]] - minCdf) / (float)(imageSize - minCdf);
			scale = round(scale *(float)(GRAYLEVELS - 1));

			imageOut[i*width + j] = scale;
		}
	}
}
