#include "ModelOrder0C.h"

ModelOrder0C::ModelOrder0C()
{
	// Häufigkeiten mit 1 initialisieren
	mTotal = 257; // 256 + Endsymbol
	for( unsigned int i=0; i<257; i++ )
		mCumCount[i] = 1;	
}

void ModelOrder0C::Encode()
{
	while( !mSource->eof() )
	{
		unsigned char symbol;

		// Symbol lesen
		mSource->read( reinterpret_cast<char*>(&symbol), sizeof( symbol ) );

		if( !mSource->eof() )
		{
			// Häufigkeiten kumulieren
			unsigned int low_count = 0;
			unsigned char j;
			for( j=0; j<symbol; j++ )
				low_count += mCumCount[j];

			// Symbol kodieren
			mAC.Encode( low_count, low_count + mCumCount[j], mTotal );

			// update model
			mCumCount[ symbol ]++;
			mTotal++;
		}
	}

	// End-Symbol schreiben
	mAC.Encode( mTotal-1, mTotal, mTotal );
}

void ModelOrder0C::Decode()
{
	unsigned int symbol;

	do
	{
		unsigned int value;

		// Wert lesen
		value = mAC.DecodeTarget( mTotal );

		unsigned int low_count = 0;

		// Symbol bestimmen
		for( symbol=0; low_count + mCumCount[symbol] <= value; symbol++ )
			low_count += mCumCount[symbol];

		// Symbol schreiben
		if( symbol < 256 )
			mTarget->write( reinterpret_cast<char*>(&symbol), sizeof( char ) );

		// Dekoder anpassen
		mAC.Decode( low_count, low_count + mCumCount[ symbol ] );

		// update model
		mCumCount[ symbol ]++;
		mTotal++;
	}
	while( symbol != 256 );
}