#include<stdio.h>


int calc_hash_index(unsigned char* c){
/*******************************************************************************
	Function name: hash calc_hash_index
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
int main()
{
    ch[]="?\n";
    printf("\n Hello world \n");
}
