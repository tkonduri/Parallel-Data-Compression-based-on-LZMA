#include<stdio.h>

int main(){
	unsigned char c;
	printf("Size of unsigned char= %d\nSize of unsigned int= %d\n",sizeof(c),sizeof(unsigned int));
	int temp=128*1+64*0+32*0+16*0+8*1+4*1+2*1+1*1;		//code=10001111
	c=temp;
	for(int i=0;i<8;i++){
		if(c&0x80)
			printf("1");
		else printf("0");
		c=c<<1;
	}
	return 0;
}