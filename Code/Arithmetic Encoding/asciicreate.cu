#include<stdio.h>
#include<conio.h>
#include<math.h>

int main(){
	FILE *opf=fopen("C:/Users/Nandan/Desktop/Project/Sem 2/Arithmetic Encoding/test/test00.txt","wb");
	char arr[256];
	for(int i=0;i<256;i++){
		arr[i]=i;
	}
	fwrite (arr , sizeof(char), 256 , opf );
	fclose(opf);
	return 0;
}