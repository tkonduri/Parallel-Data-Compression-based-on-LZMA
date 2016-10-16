#include<stdio.h>

__global__ void add(int *a, int *b, int *c){
	*c=*a+*b;
}

int main(){
	int i=5,j=10,res;
	int *dev_i,*dev_j,*dev_res;
	
	cudaMalloc((void **)&dev_i,sizeof(int));
	cudaMalloc((void **)&dev_j,sizeof(int));
	cudaMalloc((void **)&dev_res,sizeof(int));
	
	cudaMemcpy(dev_i,&i,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_j,&j,sizeof(int),cudaMemcpyHostToDevice);
	
	add<<<1,1>>>(dev_i,dev_j,dev_res);
	
	cudaMemcpy(&res,dev_res,sizeof(int),cudaMemcpyDeviceToHost);
	
	printf("The result from CPU is %d\n\tFrom GPU is %d",(i+j),res);

	return 0;
}