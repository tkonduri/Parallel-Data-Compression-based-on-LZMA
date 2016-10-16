#include "ArithmeticCoderC.h"
#include "tools.h"

// Konstanten zur Bereichsunterteilung bei 32-Bit-Integern 
// oberstes Bit wird zur Vermeidung von Überläufen freigehalten
const unsigned int g_FirstQuarter = 0x20000000;
const unsigned int g_ThirdQuarter = 0x60000000;
const unsigned int g_Half         = 0x40000000;

ArithmeticCoderC::ArithmeticCoderC()
{
	mBitCount = 0;
	mBitBuffer = 0;

	mLow = 0;
	mHigh = 0x7FFFFFFF; // arbeite nur mit den unteren 31 bit
	mScale = 0;

	mBuffer = 0;
	mStep = 0;
}

void ArithmeticCoderC::SetFile( fstream *file )
{
	mFile = file;
}

void ArithmeticCoderC::SetBit( const unsigned char bit )
{
	// Bit dem Puffer hinzufügen
	mBitBuffer = (mBitBuffer << 1) | bit;
	mBitCount++;

	if(mBitCount == 8) // Puffer voll
	{
		// schreiben
		mFile->write(reinterpret_cast<char*>(&mBitBuffer),sizeof(mBitBuffer));
		mBitCount = 0;
	}
}

void ArithmeticCoderC::SetBitFlush()
{
	// Puffer bis zur nächsten Byte-Grenze mit Nullen auffüllen
	while( mBitCount != 0 )
		SetBit( 0 );
}

unsigned char ArithmeticCoderC::GetBit()
{
	if(mBitCount == 0) // Puffer leer
	{
		if( !( mFile->eof() ) ) // Datei komplett eingelesen?
			mFile->read(reinterpret_cast<char*>(&mBitBuffer),sizeof(mBitBuffer));
		else
			mBitBuffer = 0; // Nullen anhängen

		mBitCount = 8;
	}

	// Bit aus Puffer extrahieren
	unsigned char bit = mBitBuffer >> 7;
	mBitBuffer <<= 1;
	mBitCount--;

	return bit;
}

void ArithmeticCoderC::Encode( const unsigned int low_count, 
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

void ArithmeticCoderC::DecodeStart()
{
	// Puffer mit Bits aus dem Eingabe-Code füllen
	for( int i=0; i<31; i++ ) // benutze nur die unteren 31 bit
		mBuffer = ( mBuffer << 1 ) | GetBit();
}

unsigned int ArithmeticCoderC::DecodeTarget( const unsigned int total )
// total < 2^29
{
	// Bereich in Schritte unterteilen
	mStep = ( mHigh - mLow + 1 ) / total; // oben offenes intervall => +1

	// aktuellen Wert zurückgeben
	return ( mBuffer - mLow ) / mStep;	
}

void ArithmeticCoderC::Decode( const unsigned int low_count, 
															 const unsigned int high_count )
{	
	// obere Grenze aktualisieren
	mHigh = mLow + mStep * high_count - 1; // oben offenes intervall => -1

	// untere Grenze aktualisieren
	mLow = mLow + mStep * low_count;

	// e1/e2
	while( ( mHigh < g_Half ) || ( mLow >= g_Half ) )
		{
			if( mHigh < g_Half )
			{
				mLow = mLow * 2;
				mHigh = mHigh * 2 + 1;
				mBuffer = 2 * mBuffer + GetBit();
			}
			else if( mLow >= g_Half )
			{
				mLow = 2 * ( mLow - g_Half );
				mHigh = 2 * ( mHigh - g_Half ) + 1;
				mBuffer = 2 * ( mBuffer - g_Half ) + GetBit();
			}
			mScale = 0;
		}

	// e3
	while( ( g_FirstQuarter <= mLow ) && ( mHigh < g_ThirdQuarter ) )
	{
		mScale++;
		mLow = 2 * ( mLow - g_FirstQuarter );
		mHigh = 2 * ( mHigh - g_FirstQuarter ) + 1;
		mBuffer = 2 * ( mBuffer - g_FirstQuarter ) + GetBit();
	}
}
