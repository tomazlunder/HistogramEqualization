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

__kernel void cdf(__global unsigned int *histogram,
                   __global unsigned int *output
)
{
	__local unsigned int last[256];
	__local unsigned int current[256];



	int gid = get_global_id(0);
	if(gid<256){
		output[gid] = histogram[gid];
	}

}


__kernel void cdf2(__global unsigned int *histogram,
                   __global unsigned int *output)
{
	__local unsigned int last[256];
	__local unsigned int current[256];

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint size = get_local_size(0);

	if(gid > 255) return;


	//Nastavi lokalni pomnilnik na zacetno stanje
	last[lid] = histogram[gid];
	current[lid] = histogram[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

	uint temp;

    for(uint s = 1; s < size; s =s*2) {
        if(lid < (s-1) & lid != 0) {
            current[lid] = last[lid]+last[lid-s];
        } else {
            current[lid] = last[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

		//Zamenja last
		temp = last[lid];
        last[lid] = current[lid];
		current[lid] = temp;
    }

	//Copy result to global
	output[gid] = last[lid];
}

__kernel void prescan(__global unsigned int *histogram, __global unsigned int *cdf)
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
