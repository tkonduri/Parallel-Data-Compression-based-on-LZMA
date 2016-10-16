#include <stdio.h>
#include<time.h>

__global__ void device_mulmatrix(int **dev_matrixA, int **dev_matrixB, int **dev_matrixRES)
{
	int x,y;
	x= blockIdx.x;
	y= blockIdx.y;
	
	int i;
	dev_matrixRES[x][y]=0;
	for(i=0;i<100;i++){
		dev_matrixRES[x][y]+=dev_matrixA[x][i]*dev_matrixB[i][x];
	}
}

int main(void)
{
	clock_t begin, end;
	double time_spent;
	begin = clock();		//End of begin time measurement routine here...
	
	int *dev_matrixA, *dev_matrixB,*dev_matrixRES;
	int **host_matA, **host_matB;
	int i,j;
	host_matA= (int **) malloc(sizeof(int)*100*100);
	host_matB= (int **) malloc(sizeof(int)*100*100);
	
	for(i=0;i<100;i++){
		for(j=0;j<100;j++){
			host_matA[100][100]= 2*i+j;
			host_matB[100][100]=2*j+i*i;
		}
	}
	cudaMalloc((void**)&dev_matrixA,sizeof(int)*100*100);
	cudaMalloc((void**)&dev_matrixB,sizeof(int)*100*100);
	cudaMalloc((void**)&dev_matrixRES,sizeof(int)*100*100);
	dim3 grid(100,100);
	cudaMemcpy((int *)host_matA ,dev_matrixA,sizeof(int)*100*100,cudaMemcpyHostToDevice);
	cudaMemcpy((int *)host_matA,dev_matrixA,sizeof(int)*100*100,cudaMemcpyHosttoDevice);
	
	device_mulmatrix<<<grid,1>>>(dev_matrixA,dev_matrixB, dev_matrixRES);

	//Measure end time and hence time req for app execution
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Execution time: %lf",time_spent);
  
  return 0;
}