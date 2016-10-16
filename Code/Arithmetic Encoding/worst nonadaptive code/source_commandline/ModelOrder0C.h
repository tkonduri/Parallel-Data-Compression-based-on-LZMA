#ifndef __MODELORDER0C_H__
#define __MODELORDER0C_H__

#include "ModelI.h"

class ModelOrder0C : public ModelI  
{
public:
	ModelOrder0C();

protected:
	void Encode();
	void Decode();

	unsigned int mCumCount[ 257 ];
	unsigned int mTotal;
};

#endif 
