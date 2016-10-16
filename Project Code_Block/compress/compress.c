#include<stdio.h>
#include<string.h>
#include<malloc.h>

//Global file pointer for input file
FILE *fp_in;

//flags used
int eof_flag,                //for identifying end of file
    count = 0,             //cout of number of characters read
    rec_start_flag = 1;   //for identifying the records are being written
                            // rec_start_flag = 1 indicating that put ',' and start writing from first record toggle flag
                            // rec_start_flag = 0 indicating that put ',' and start writing from first bytes toggle flag

//input file
char *input;

//output file
char *output;

//opening the output file in append mode
#define write() fp_out=fopen(output,"a");fprintf(fp_out,"%c",ch[0]);fclose(fp_out)

//count of number of distances store in hash_arry index
#define MAX_hash_count 24

#define MAX_hash_size 65535

//for storing the distances, eventually get stored in hash_array
struct node
{
    int dist;
    struct node *next;
};

//for storing the distances in the from of hash list
struct hash_array
{
    int count;
    struct node *next;
};
struct hash_array hash_arr[MAX_hash_size];


//for encoding compressed data
struct Compress_pair
{
    unsigned int dist;
    unsigned int len;
}pair;

void init()
{
    FILE *fp_out=fopen(output,"w");
    fclose(fp_out);
}


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

//writing the record from struct compress_pair
void write_record()
{
    FILE *fp_out;
    fp_out = fopen(output,"a");      //in Append mode
    //fprintf(fp_out,"[%d,%d]",pair.dist,pair.len);
    if(rec_start_flag){
        fprintf(fp_out,"%c",',');
        rec_start_flag = 0;
    }
    fwrite(&pair,sizeof(pair),1,fp_out);
    fclose(fp_out);
}
void find_longest_match(char ch[2], unsigned int key_idx, int cur_count)
{
    FILE *fp_in2;
    struct node* temp = hash_arr[key_idx].next;
    unsigned int temp_len = 0;
    char cur_ch;
    pair.len = 0;

    while(temp){

        //using global input file pointer "fp_in"
        fseek(fp_in,-2,SEEK_CUR); //setting current file pointer

        //opening INPUT file for setting file pointer at current distance
        fp_in2 = fopen(input,"r");
        if(!fp_in2){
            printf("Error in opening the input file in find_logest_match()");
            exit(1);
        }
        fseek(fp_in2,temp->dist,SEEK_CUR);   //setting file pointer at current distance

        //finding the length of characters to be matched
        temp_len = 0;
        while(1){
            cur_ch = fgetc(fp_in);

            if(cur_ch==fgetc(fp_in2))
                temp_len++;
            else if(cur_ch==EOF){
                eof_flag = 1;
                break;
            }
            else
                break;
        }

        if(temp_len > pair.len){
            pair.len = temp_len;        //"temp_len" variable
            pair.dist = temp->dist;     //this "temp" is a node
        }

        fclose(fp_in2);
        temp = temp->next;              //Increment is very important in any link list
                                        //ALWAYS REMEMBER INCREMENT
    }
    count+=pair.len;
    write_record();

}
void insert_hash_array(unsigned char ch[2], int key_dist)
{
    FILE *fp_out;
    unsigned int key_idx = calc_hash_index(ch);//(int)(ch[0] & ch[1]);
    struct node *temp = (struct node*)malloc(sizeof(struct node));
    struct node *t;
    temp->dist = key_dist;
    if(hash_arr[key_idx].count == 0){
        hash_arr[key_idx].next=temp;
        temp->next=NULL;
        hash_arr[key_idx].count++;
        //write();            //macro for writing in output file

        //WRITING A BYTE IN OUPUT FILE
        fp_out = fopen(output,"a");      //in Append mode
        if(!rec_start_flag){
            fprintf(fp_out,"%c",',');
            rec_start_flag = 1;
        }
            fprintf(fp_out,"%c",ch[0]);
        fclose(fp_out);
    }
    else{

        //finding best match distance
        find_longest_match(ch, key_idx, key_dist);
        printf("\n\n %c \n\n",getc(fp_in));
        temp->next = hash_arr[key_idx].next;
        hash_arr[key_idx].next=temp;
        hash_arr[key_idx].count++;

        //Parallelized
        if(hash_arr[key_idx].count > MAX_hash_count){
            t = hash_arr[key_idx].next;
            while(t->next->next != NULL)
                t=t->next;
            free(t->next->next);
            t->next = NULL;
        }
    }
}

 int main(int argc, char* argv[])
 {
//    FILE *fp_in;
    unsigned char ch[2];

    printf(" \n Hello world \n");

    eof_flag = 0;
    if(argc == 1){
        input = strdup("sample.txt");
        fp_in=fopen("sample.txt","r");
        if(!fp_in){
            printf("\nError in opening the file,program is needed to close \n");
            return 1;
        }
        output = strdup("output.txt");
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
        printf("\n compress.exe <inputfile.txt> <outputfile.txt> ");

    }
    //fseek(fp,6,SEEK_CUR);
    //printf("\n\nChar is: %c \n\n",fgetc(fp_in));
    //printf("\n\n\n%d",sizeof(char));
    ch[0] = fgetc(fp_in);
    while(1){
        ch[1] = fgetc(fp_in);
        if(ch[1]== EOF || eof_flag)
            break;
        printf("%c",ch[0]);
//        hash_key = ch[0] & ch[1];
        insert_hash_array(ch,count);
        count++;
        ch[0]=ch[1];
    }
    fclose(fp_in);

    return 0;
 }
