#ifndef __ARITHMETICCODERC_H__
#define __ARITHMETICCODERC_H__

#include <fstream>
using namespace std;

class ArithmeticCoderC  
{
public:
	ArithmeticCoderC();

	void SetFile( fstream *file );

	void Encode( const unsigned int low_count, 
	             const unsigned int high_count, 
	             const unsigned int total );
	void EncodeFinish();

	void DecodeStart();
	unsigned int DecodeTarget( const unsigned int total );
	void Decode( const unsigned int low_count, 
	             const unsigned int high_count );

protected:
	// bit operations
	void SetBit( const unsigned char bit );
	void SetBitFlush();
	unsigned char GetBit();

	unsigned char mBitBuffer;
	unsigned char mBitCount;

	// in-/output stream
	fstream *mFile;

	// encoder & decoder
	unsigned int mLow;
	unsigned int mHigh;
	unsigned int mStep;
	unsigned int mScale;

	// decoder
	unsigned int mBuffer;
};

#endif