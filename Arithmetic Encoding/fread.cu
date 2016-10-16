#include<stdio.h>
#include<conio.h>
#define MAX 1000000

int main(){
	char str[MAX];
	FILE *fp,*fp2;
	fp= fopen("C:/Users/Nandan/Desktop/Project/Sem 2/Arithmetic Encoding/test/big.txt","r");
	fp2=fopen("C:/Users/Nandan/Desktop/Project/Sem 2/Arithmetic Encoding/test/copy.txt","w");
	if(fp==NULL||fp2==NULL ){
		printf("Error opening file...");
		return 0;
	}
	int cnt;
	while(cnt=fread(str,sizeof(char),MAX,fp)){
		fwrite(str,sizeof(char),cnt,fp2);
		//printf("%s",str);
	}
	
	fclose(fp);
	fclose(fp2);
	return 0;
}