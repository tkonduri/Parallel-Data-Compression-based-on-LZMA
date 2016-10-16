#include "ModelI.h"

void ModelI::Process( fstream *source, fstream *target, ModeE mode )
{
	mSource = source;
	mTarget = target;

	if( mode == MODE_ENCODE )
	{
		mAC.SetFile( mTarget );

		// kodieren
		Encode();

		mAC.EncodeFinish();
	}
	else // MODE_DECODE
	{
		mAC.SetFile( mSource );

		mAC.DecodeStart();

		// dekodieren
		Decode();
	}
};