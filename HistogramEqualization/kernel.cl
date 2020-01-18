__kernel void localKernel(
	__global unsigned char *globalIMAGEin,
	__global unsigned int *globalRESULT
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
	int width = get_global_size(0);

	int gid = globalY*width + globalX;
	unsigned int color = globalIMAGEin[gid];

	atomic_inc(&localRESULT[color]);

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//Kopiranje v globalni pomnilnik
	if(lid < 256){
		atomic_add(&globalRESULT[lid], localRESULT[lid]);
	}
}

#define SWAP(a,b) {__local unsigned long *tmp=a;a=b;b=tmp;}

__kernel void cdf(__global unsigned long *histogram,
                   __global unsigned int *output
)
{
	int gid = get_global_id(0);
	if(gid<256){
		output[gid] = gid;
	}
}
