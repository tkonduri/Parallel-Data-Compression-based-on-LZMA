#include<stdio.h>
#include<conio.h>
#include<math.h>

//#define BLKSZ 1000
#define MAX 10000
#define IPF "C:/Users/Nandan/Desktop/Project/Sem 2/Arithmetic Encoding/test/output/06/output.txt"
#define OPF "C:/Users/Nandan/Desktop/Project/Sem 2/Arithmetic Encoding/test/output/06/output02.txt"
#define g_FirstQuarter 0x20000000
#define g_ThirdQuarter 0x60000000
#define g_Half 0x40000000

void probCPU(char *str,double *tab,int count);
int arithmetic(char *str,double *low,double *high,unsigned char *op,int count);

int main(){
	FILE *ipf,*opf;
	//char buffer[1000][BLKSZ];
	char str[MAX];
	unsigned char op[MAX];
	double prob[257],low[257],high[257];
	int count,total=0;
	
	//Compute Probibilities
	ipf= fopen(IPF,"r");
	if(ipf==NULL)printf("Error opening input file!!!");
	//printf("Probability computation launch...");
	while(count=fread(str,sizeof(char),MAX,ipf)){
		probCPU(str,prob,count);
	}
	//printf("Probability computation return...");
	fclose(ipf);
	
	//Ensure total< 2^29
	prob[256]=1;
	for(count=0;count<257;count++){
		total+=prob[count];
	}
	
	if(total>pow((double)2,29)){
		int div=total/pow((double)2,29);
		for(count=0;count<257;count++){
			if(prob[count]!=0){
				prob[count]/=div;
				prob[count]=ceil(prob[count]);
			}
		}
	}
	
	//Calculate scaled low and high counts from freq.
	low[0]=0;
	high[0]=prob[0];
	for(count=1;count<257;count++){
		low[count]=high[count-1];
		high[count]=low[count]+prob[count];
	}
	///////////test code
	for(count=0;count<257;count++){
		{
			//printf("\n%d(%c)--->%lf ---- %lf to %lf",count,count,prob[count],low[count],high[count]);
		}
	}
	//getch();
	
	//Encode
	ipf= fopen(IPF,"r");
	if(ipf==NULL)printf("Error opening input file!!!");
	else printf("Opening input file successfull!!!");
	opf= fopen(OPF,"wb");
	if(opf==NULL)printf("Error opening output file!!!");
	else printf("Opening output file successfull!!!");
	while(count=fread(str,sizeof(char),MAX,ipf)){
		//printf("Launching encoder...");
		count= arithmetic(str,low,high,op,count);
		//printf("Successfull return from encoder...");
		fwrite (op , sizeof(char), count , opf );
		//printf("\nData write TRUE");
	}
	fclose(opf);
	fclose(ipf);
	//Encode complete
	
	return 0;
}

void probCPU(char *str,double *tab,int count){
	for(long int i=0;i<count;i++){
		tab[str[i]+128]++;
	}
}

/********************************************
Arithmatic Encoding function.
Precondition: String to be encoded given in str
Postcondition:
	1) op points to encoded string
	2) returns number of bytes of actual output
*********************************************/
int arithmetic(char *str,double *low,double *high,unsigned char *op,int count){
	unsigned int mLow= 0;
	unsigned int mHigh= 0x7FFFFFFF;
	unsigned int mStep= 0;
	unsigned int mScale= 0;
	int tempval=0,tempmul=128,ocntr=0;
	
	unsigned int low_count,high_count,total=high[256];
	//printf("Init total success...");
	for(int i=0;i<count;i++){
		low_count=low[str[i]+128];
		high_count=high[str[i]+128];
		
		//printf("Low and High got success...");
		mStep = ( mHigh - mLow + 1 ) / total;
		mHigh = mLow + mStep * high_count - 1;
		mLow = mLow + mStep * low_count;
		//char c=169;
		//printf("Range calculate success...%c %d %c %d %d",str[i],str[i],c,low_count,high_count);
		//////////////////////////////////////////////CODE REMAINING FROM HERE!!!
		// e1/e2 Scaling
		while( ( mHigh < g_Half ) || ( mLow >= g_Half ) )
			{
				if( mHigh < g_Half )
				{
					{
						tempval+=tempmul*0;
						if(tempmul==1){
							tempmul=128;
							op[ocntr++]=tempval;
							tempval=0;
						}
						else{
							tempmul/=2;
						}
					}
					mLow = mLow * 2;
					mHigh = mHigh * 2 + 1;

					// e3
					for(; mScale > 0; mScale-- ){
						tempval+=tempmul*1;
						if(tempmul==1){
							tempmul=128;
							op[ocntr++]=tempval;
							tempval=0;
						}
						else{
							tempmul/=2;
						}
					}
				}
				else if( mLow >= g_Half )
				{
					{
						tempval+=tempmul*1;
						if(tempmul==1){
							tempmul=128;
							op[ocntr++]=tempval;
							tempval=0;
						}
						else{
							tempmul/=2;
						}
					}
					mLow = 2 * ( mLow - g_Half );
					mHigh = 2 * ( mHigh - g_Half ) + 1;

					// e3
					for(; mScale > 0; mScale-- )
						{
						tempval+=tempmul*0;
						if(tempmul==1){
							tempmul=128;
							op[ocntr++]=tempval;
							tempval=0;
						}
						else{
							tempmul/=2;
						}
					}
				}
			}

		// e3
		while( ( g_FirstQuarter <= mLow ) && ( mHigh < g_ThirdQuarter ) )
		{
			mScale++;
			mLow = 2 * ( mLow - g_FirstQuarter );
			mHigh = 2 * ( mHigh - g_FirstQuarter ) + 1;
		}
		//printf("Scaling success...");
	}
	
	///eNCODE FINISH ROUTINE HERE...
	if( mLow < g_FirstQuarter ) // mLow < FirstQuarter < Half <= mHigh
	{
		{
			tempval+=tempmul*0;
			if(tempmul==1){
				tempmul=128;
				op[ocntr++]=tempval;
				tempval=0;
			}
			else{
				tempmul/=2;
			}
		}

		for( int i=0; i<mScale+1; i++ ) {
			tempval+=tempmul*1;
			if(tempmul==1){
				tempmul=128;
				op[ocntr++]=tempval;
				tempval=0;
			}
			else{
				tempmul/=2;
			}
		}
	}
	else // mLow < Half < ThirdQuarter <= mHigh
	{
		{
			tempval+=tempmul*1;
			if(tempmul==1){
				tempmul=128;
				op[ocntr++]=tempval;
				tempval=0;
			}
			else{
				tempmul/=2;
			}
		}
	}
	op[ocntr++]=tempval;
	//printf("Returning...");
	return ocntr;
}

/*
void Encode( const unsigned int low_count, 
								const unsigned int high_count, 
								const unsigned int total )
// total < 2^29
{
	// Bereich in Schritte unterteilen
	mStep = ( mHigh - mLow + 1 ) / total; // oben offenes intervall => +1

	// obere Grenze aktualisieren
	mHigh = mLow + mStep * high_count - 1; // oben offenes intervall => -1
	
	// untere Grenze aktualisieren
	mLow = mLow + mStep * low_count;

	// e1/e2 Mapping durchführen
	while( ( mHigh < g_Half ) || ( mLow >= g_Half ) )
		{
			if( mHigh < g_Half )
			{
				SetBit( 0 );
				mLow = mLow * 2;
				mHigh = mHigh * 2 + 1;

				// e3
				for(; mScale > 0; mScale-- )
					SetBit( 1 );
			}
			else if( mLow >= g_Half )
			{
				SetBit( 1 );
				mLow = 2 * ( mLow - g_Half );
				mHigh = 2 * ( mHigh - g_Half ) + 1;

				// e3
				for(; mScale > 0; mScale-- )
					SetBit( 0 );
			}
		}

	// e3
	while( ( g_FirstQuarter <= mLow ) && ( mHigh < g_ThirdQuarter ) )
	{
		mScale++;
		mLow = 2 * ( mLow - g_FirstQuarter );
		mHigh = 2 * ( mHigh - g_FirstQuarter ) + 1;
	}
}

void ArithmeticCoderC::EncodeFinish()
{
	// Es gibt zwei Möglichkeiten, wie mLow und mHigh liegen, d.h.
	// zwei Bits reichen zur Entscheidung aus.
	
	if( mLow < g_FirstQuarter ) // mLow < FirstQuarter < Half <= mHigh
	{
		SetBit( 0 );

		for( int i=0; i<mScale+1; i++ ) // 1 + e3-Skalierung abbauen
			SetBit(1);
	}
	else // mLow < Half < ThirdQuarter <= mHigh
	{
		SetBit( 1 ); // der Decoder fügt die Nullen automatisch an
	}

	// Ausgabepuffer leeren
	SetBitFlush();
}

*/


	/*total scale test code
	printf("Before division....");
	for(count=0;count<257;count++){
		if(prob[count]!=0){
			printf("\n%d(%c)--->%lf",count,count,prob[count]);
		}
	}
	int div=2;
	for(count=0;count<257;count++){
		if(prob[count]!=0){
			prob[count]/=div;
			prob[count]=ceil(prob[count]);
		}
	}
	printf("\n\nAfter division...");
	for(count=0;count<257;count++){
		if(prob[count]!=0){
			printf("\n%d(%c)--->%lf",count,count,prob[count]);
		}
	}
	*/
	
	/* Low_count and hig_count from prob test code
	for(count=0;count<257;count++){
		if(prob[count]!=0){
			printf("\n%d(%c)--->%lf ---- %lf to %lf",count,count,prob[count],low[count],high[count]);
		}
	}	
	*/