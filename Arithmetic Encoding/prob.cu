#include<stdio.h>
#include<time.h>
#include<conio.h>

#define MAX 500000

__global__ void probGPU(char *text, int *prob){
	int tid=threadIdx.x;
	int bid=blockIdx.x;
	//if(text[bid]==tid) prob[tid]++;			//doesn't work due to race
	for(int i=0;i<MAX;i++){
		if(text[i]==bid) prob[bid]++;
	}
	__syncthreads();
}

void probCPU(char *str,int *tab){
	for(long int i=0;i<MAX;i++){
		tab[str[i]]++;
	}
}

int main(){
	char str[MAX];
	int tableGPU[129],tableCPU[129]={0};
	char *dev_c;
	int *dev_table;
	
	//*************************************GPU CODE BEGINS******************************************************
	clock_t begin1, end1;
	double time_spent1;
	begin1 = clock();													//Begin time measurement
	
	cudaMalloc((void **)&dev_c,MAX*sizeof(char));						
	cudaMalloc((void **)&dev_table, 129*sizeof(int));					
	
	cudaMemcpy(dev_c,str,MAX*sizeof(char),cudaMemcpyHostToDevice);
	
	begin1 = clock();
	probGPU<<<128,1>>>(dev_c,dev_table);
	//probGPU<<<128,1>>>(dev_c,dev_table);
	//probGPU<<<128,1>>>(dev_c,dev_table);
	//probGPU<<<128,1>>>(dev_c,dev_table);
	//probGPU<<<128,1>>>(dev_c,dev_table);
	end1 = clock();
	
	cudaMemcpy(tableGPU,dev_table,129*sizeof(int),cudaMemcpyDeviceToHost);	
	
	//end1 = clock();
	time_spent1 = (double)(end1 - begin1) / CLOCKS_PER_SEC;				//Measure time required
	//*************************************GPU CODE END*******************************************************
	
	//*************************************CPU CODE BEGINS******************************************************
	clock_t begin2, end2;
	double time_spent2;
	begin2 = clock();													//Begin time measurement
	
	probCPU(str,tableCPU);
	//probCPU(str,tableCPU);
	//probCPU(str,tableCPU);
	//probCPU(str,tableCPU);
	//probCPU(str,tableCPU);
	
	end2 = clock();
	time_spent2 = (double)(end2 - begin2) / CLOCKS_PER_SEC;				//Measure time required
	//*************************************CPU CODE END******************************************************
	
	/* debugging
	int count=0;
	for(int i=0;i<128;i++){
		if(tableGPU[i]!=0||tableCPU[i]!=0){
			if(count%2==0)printf("\n");
			else printf("\t");
			
			printf("%c %d %d",i,tableGPU[i],tableCPU[i]);
			count++;
		}
		
		if(count%30==0 && count!=0)getch();
	} */
	
	//*************************************Verify both the results**********************************************
	for(int i=0;i<128;i++){
		if(tableGPU[i]!=tableCPU[i]){
			printf("\n\n\t\tInconsistent Output.%lf %lf",time_spent1,time_spent2);
			exit(0);
		}
	}
	printf("Results verified....\nGPU=%lf CPU=%lf",time_spent1,time_spent2);
	
	
	/***Printing all probablity values(including the ones that are zero)*******
	for(int i=0;i<128;i++){
		if(i%2==0)printf("\n");
		else printf("\t");
		
		printf("%c %d",i,tableCPU[i]);
		//printf("\n%d %c",i);
		
		if(i%20==0&&i!=0) getch();
	}
	/**************************************************************************/
	
	/********Printing only non zero probability values**************************
	int count=0;
	for(int i=0;i<128;i++){
		if(table[i]!=0){
			if(count%2==0)printf("\n");
			else printf("\t");
			
			printf("%c %d",i,table[i]);
			count++;
		}
		
		if(count%30==0 && count!=0)getch();
	}
	/**************************************************************************/

	cudaFree(dev_c);
	cudaFree(dev_table);
	
}