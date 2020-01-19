__kernel void localKernel(
	__global unsigned char *globalIMAGEin,
	__global unsigned int *globalRESULT,
	const unsigned int width,
	const unsigned int height
)
{
	__local unsigned int localRESULT[256];

	// Nastavi lokalni pomnilnik na 0
	int lid = 	get_local_id(1) * get_local_size(0) + get_local_id(0);

	if(lid < 256){
		localRESULT[lid] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// Doda svojo vrednost v lokalni pomnilnik
	int globalX = get_global_id(0);
	int globalY = get_global_id(1);

	int gid = globalY*width + globalX;

	if(globalX < width & globalY < height){
		unsigned int color = globalIMAGEin[gid];
		atomic_inc(&localRESULT[color]);
	}

	if(globalX < width & globalY < height){
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//Kopiranje v globalni pomnilnik
		if(lid < 256){
			atomic_add(&globalRESULT[lid], localRESULT[lid]);
		}
	}
}

__kernel void cdf(__global unsigned int *histogram, __global unsigned int *cdf)
{
	int n = 256;
	__local unsigned int temp[256];// allocated on invocation
	int thid = get_global_id(0);
	int offset = 1;
	temp[2*thid] = histogram[2*thid]; // load input into shared memory
	temp[2*thid+1] = histogram[2*thid+1];
	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (thid < d){
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		 }
		 offset *= 2;
	}
	 if (thid == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		 if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
 barrier(CLK_LOCAL_MEM_FENCE);
 cdf[2*thid] = temp[2*thid]; // write results to device memory
 cdf[2*thid+1] = temp[2*thid+1];
} 

__kernel void minCdf(__global unsigned int *input, __global unsigned int *n)
{
	__local unsigned int temp[256];// allocated on invocation
	int gid = get_global_id(0);

	if(gid < 256){
		temp[gid] = input[gid];
	}

	for(int i = 128; i > 0; i=i/2)
	{
		if(gid < i){
			if(temp[gid] == 0){
				temp[gid] = temp[gid+i];
			} else if (temp[gid+i] != 0 & temp[gid+i] < temp[gid]){
				temp[gid] = temp[gid+i];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(gid == 0){
		*n = temp[0];
	}
}

__kernel void equalize(__global const unsigned char *imageIn,
					    __global unsigned char *imageOut,
						__global const unsigned int *cdf,
					 	const unsigned int width,
						const unsigned int height,
						const unsigned int cdfMin)
{
	int gX = get_global_id(0);
	int gY = get_global_id(1);

	int gid = gY * width + gX;

	if(gX < width & gY < height){
		unsigned long imageSize = width * height;

		float scale;
		scale = (float)(cdf[imageIn[gid]] - cdfMin) / (float) (imageSize - cdfMin);
		scale = round(scale * (float)(256-1));
		
		
		imageOut[gid] = (int) scale;

		//imageOut[gid] = 155;
	}
}
