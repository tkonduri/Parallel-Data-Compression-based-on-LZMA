#include<stdio.h>
#include<conio.h>

void main(){
	char c;
	int i=0;
	for(c=0; c>=0 && c<128;c++){
		i=c;
		if(c%5!=0)printf("\t\t");
		else printf("\n");
		printf("%d= %c",i,c);
	}
}
