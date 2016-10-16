#include<stdio.h>
#include<string.h>

union comp_pair
{
    int dist;
    int len;
}pair;


char *ch1;
 int main(int argc, char* argv[])
 {
     char *ch;
     ch1=strdup("sample.txt");
     printf("%s",ch1);
/*        char ch1[2]="r ",ch2[2]="s ";
        printf("\n\nr_ = %d",crc16(&ch1,2,0));
        printf("\n\ns_ = %d",crc16(&ch2,2,0));
*/
/*
        FILE *fp_in,*fp_out;
        pair.len=12;
        pair.dist=23;
        printf("%d\n",sizeof(pair));
        fp_in=fopen("temp.txt","w");
        //fprintf(fp_in,"%c",'a');
        fwrite(&pair,sizeof(pair),1,fp_in);
        fwrite(&pair,sizeof(pair),1,fp_in);
        fclose(fp_in);

        fp_in = fopen("TEMP.txt","r");
        if(!fp_in){
            printf("\nError");
            return 1;
        }

        fread(&pair,sizeof(pair),1,fp_in);
        printf("%d \t %d \n",pair.dist,pair.len);

        fread(&pair,sizeof(pair),1,fp_in);
        printf("%d \t %d \n",pair.dist,pair.len);
        fclose(fp_in);
*/

        printf("\nargc = %d",argc);
        printf("\nargv[0] = %s",argv[0]);
      //  printf("\nargv[1] = %s",*argv[1]);
       // printf("\nargv[2] = %s",*argv[2]);
        return 0;
 }

/*
fread()
Syntax:
#include <stdio.h>
int fread( void *buffer, size_t size, size_t num, FILE *stream );

Description:
The function fread() reads num number of objects (where each object is size bytes) and places them into
the array pointed to by buffer. The data comes from the given input stream.
The return value of the function is the number of things read...use |feof()| or |ferror()| to figure out
if an error occurs.



fwrite()
Syntax:
#include <stdio.h>
int fwrite( const void *buffer, size_t size, size_t count, FILE *stream );

Description:
The fwrite() function writes, from the array buffer, count objects of size size to stream.
The return value is the number of objects written.

*/
