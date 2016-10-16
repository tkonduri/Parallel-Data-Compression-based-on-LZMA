#include<stdio.h>

__global__ void probGPU(int *t){
	int bid=threadIdx.x;
	*t=bid;
}

int main(){
	int *test;
	int test2=5;
	clock_t begin2, end2;
	double time_spent2=0;
	begin2 = clock();
	
	cudaMalloc((void **)&test,sizeof(int));
	cudaMemcpy(test,&test2,sizeof(int),cudaMemcpyHostToDevice);
	probGPU<<<1,1>>>(test);
	cudaMemcpy(&test2,test,sizeof(int),cudaMemcpyDeviceToHost);
	end2 = clock();
	time_spent2 += (double)(end2 - begin2) / CLOCKS_PER_SEC;
	printf("%d.. Time= %lf",test2,time_spent2);
	return 0;
}