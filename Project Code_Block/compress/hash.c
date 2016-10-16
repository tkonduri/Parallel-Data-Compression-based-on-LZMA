#include<stdio.h>
#include<stdlib.h>

int hash(unsigned char* c){
/*******************************************************************************
	Function name: hash
	Used for hashing of 2 signed characters(having ASCII from 0 to 255
	to values 0 to 65535 without conflict
	Input: pointer to first character
	Output: hashed value
	Requirement: Both the characters should be in consecutive memory locations
	(array) and pointer to first char should be passed.
*******************************************************************************/
	int a,b;
	a=*c;
	c++;
	b=*c;
	return (a*256+b);
}


int shash( char* c){\
/*******************************************************************************
	Function name: shash
	Used for hashing of 2 signed characters(having ASCII from -128 to +127
	 to values 0 to 65535 without conflict
	Input: pointer to first character
	Output: hashed value
	Requirement: Both the characters should be in consecutive memory locations
	(array)and pointer to first char should be passed.
*******************************************************************************/
	int a,b;
	a=*c;
	c++;
	b=*c;
	a+=128;
	b+=128;
	return (a*256+b);
}

void main(){
	/*Check for unsigned char only*/
	unsigned char c[2];
	int i,j,r;
	int a[65536]={0};
	for(i=0;i<256;i++){
		for(j=0;j<256;j++){
			c[0]=i;
			c[1]=j;
			r=hash(c);
			if(a[r]==1){
				printf("Conflict at index %d",r);
				exit(0);
			}
		}
	}
	printf("Your hash function rocks!!! No conflict!");
}
