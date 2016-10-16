#include<stdio.h>
#include<stdlib.h>
#include<string.h>

//Macro for INPUT file
char *input;

//Macro for OUPUT file
char *output;

FILE *fp_out;

unsigned int eof_flag;
//for encoding compressed data
struct Compress_pair
{
    unsigned int dist;
    unsigned int len;
}pair;

void init()
{
    fp_out = fopen(output,"w");
    fclose(fp_out);
}
void expand_write()
{
    FILE *fp_out_read, *fp_out_write;
    char ch;

        fp_out_read = fopen(output,"r+");
        if(!fp_out_read){
            printf("Error in opening the file %s",output);
            exit(1);
        }
        fseek(fp_out_read, pair.dist, SEEK_CUR);


        while(pair.len){

            ch = fgetc(fp_out_read);
            fp_out_write = fopen(output,"a+");
            fprintf(fp_out_write, "%c", ch);
            fclose(fp_out_write);

            pair.len--;
        }
        fclose(fp_out_read);


}
/*
void expand_write()
{
    //FILE *temp_fp_out;
    FILE *fp_in;
    char ch;
    while(pair.len){
        fp_in = fopen(input,"r+");
        if(!fp_in){
            printf("Error  in opening the input file %s",input);
            exit(0);
        }

        fseek(fp_in, pair.dist, SEEK_CUR);

        ch = fgetc(fp_out);
        fp_out = fopen(output, "a+");
        fprintf(fp_out,"%c", ch);
        fclose(fp_out);
        fclose(fp_in);
        pair.len--;
        pair.dist++;
    }

}
*/
int read_rec_expand(FILE *fp_in)
{
    char ch;
    while(1){
        ch = fgetc(fp_in);
        if(ch == EOF)
            return 1;

        if(ch == ',')
            return 0;


        else{
            fseek(fp_in,-1,SEEK_CUR);
            fread(&pair,sizeof(pair),1,fp_in);
            expand_write();
            printf("%d \t %d \n",pair.dist, pair.len);
        }
    }
}
int main(int argc, char* argv[])
{
    FILE *fp_in;
    char ch;

    eof_flag = 0;

    printf(" \n Hello world \n");

    if(argc == 1){
        input = strdup("output.txt");
        fp_in=fopen("output.txt","r");
        if(!fp_in){
            printf("\nError in opening the file, program is needed to close \n");
            return 1;
        }
        output = strdup("de_compress.txt");
        init();
    }
    else if(argc == 3){
        input = strdup(argv[1]);
        fp_in = fopen(argv[1],"r");
        if(!fp_in){
            printf("\nError in opening the file need to close \n");
            return 1;
        }
        output = strdup(argv[2]);
        init();
    }
    else{
        printf("\n Error in syntax of the command \n Syntax is as follows:- \n");
        printf("\n Decompress.exe <inputfile.txt> <outputfile.txt> ");

    }


    while(1){
        ch = fgetc(fp_in);
            if(ch == EOF)
                break;
            if(ch == ','){
                if(read_rec_expand(fp_in)){
                    break;
                }
            }
            else{
                fp_out = fopen(output,"a");
                fprintf(fp_out,"%c",ch);
                fclose(fp_out);
            }

    }
    printf("\n Decompression is done\n\n output file is:- %s",output);
    return 0;
}
