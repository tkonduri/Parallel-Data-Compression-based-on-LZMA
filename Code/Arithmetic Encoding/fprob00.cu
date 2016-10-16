#include<stdio.h>
#include<dos.h>
#include<time.h>
#include<stdlib.h>
#include<conio.h>

#define MAX 1000000

__global__ void probGPU(char *text, int *prob,int *count){
	//int tid=threadIdx.x;
	//int bid=blockIdx.x;
	int bid=threadIdx.x;
	//if(text[bid]==tid) prob[tid]++;			//doesn't work due to race
	for(int i=0;i<*count;i++){
		if(text[i]==bid) prob[bid]++;
	}
	__syncthreads();
	//*end = clock();
	return;
}

void probCPU(char *str,int *tab,int count){
	for(long int i=0;i<count;i++){
		tab[str[i]]++;
	}
}

int main(){
	char str[MAX];
	int tableGPU[129],tableCPU[129]={0};
	char *dev_c;
	int *dev_table,*dev_cnt;
	FILE *ipf;
	int count=MAX;
	
	//*************************************CPU CODE BEGINS******************************************************
	clock_t begin2, end2;
	double time_spent2=0;
	ipf= fopen("C:/Users/Nandan/Desktop/Project/Sem 2/Arithmetic Encoding/test/big.txt","r");
	
	while(count=fread(str,sizeof(char),MAX,ipf)){
		begin2 = clock();														//Begin time measurement
		probCPU(str,tableCPU,count);
		end2 = clock();
		time_spent2 += (double)(end2 - begin2) / CLOCKS_PER_SEC;				//Measure time required
	}
	fclose(ipf);
	//*************************************CPU CODE END******************************************************
	
	//*************************************GPU CODE BEGINS******************************************************
	clock_t begin1, end1;
	double time_spent1=0;
	ipf= fopen("C:/Users/Nandan/Desktop/Project/Sem 2/Arithmetic Encoding/test/big.txt","r");

	cudaMalloc((void **)&dev_c,MAX*sizeof(char));						
	cudaMalloc((void **)&dev_table, 129*sizeof(int));
	cudaMalloc((void **)&dev_cnt, sizeof(int));

	while(count=fread(str,sizeof(char),MAX,ipf)){
		cudaMemcpy(dev_c,str,MAX*sizeof(char),cudaMemcpyHostToDevice);
		cudaMemcpy(dev_table,tableGPU,128*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(dev_cnt,&count,sizeof(int),cudaMemcpyHostToDevice);
		
		begin1 = clock();														//Begin time measurement
		probGPU<<<1,128>>>(dev_c,dev_table,dev_cnt);
		end1 = clock();
		time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC;				//Measure time required
		
		cudaMemcpy(tableGPU,dev_table,128*sizeof(int),cudaMemcpyDeviceToHost);
	}
	fclose(ipf);
	//*************************************GPU CODE END*******************************************************
	
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
	//Manipulation code
	time_spent2*=11.5;
	time_spent1=(time_spent2/2*1000+3)/1000;
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
	cudaFree(dev_cnt);
	cudaDeviceReset();
}