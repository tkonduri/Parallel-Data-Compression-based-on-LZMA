#include <iostream>
#include <fstream>
using namespace std;

#include "ModelOrder0C.h"

// signature: "ACMC" (0x434D4341, intel byte order)
const int g_Signature = 0x434D4341;

int main(int argc, char *argv[])
{
	cout << "Arithmetische Codierung" << endl;

	if( argc != 3 )
	{
		cout << "Syntax: AC source target" << endl;
		return 1;
	}

	fstream source, target;
	ModelI* model;
	
	// Modell auswählen, hier nur Order0
	model = new ModelOrder0C;

	source.open( argv[1], ios::in | ios::binary );
	target.open( argv[2], ios::out | ios::binary );

	if( !source.is_open() )
	{
		cout << "Kann Eingabestrom nicht öffnen";
		return 2;
	}
	if( !target.is_open() )
	{
		cout << "Kann Ausgabestrom nicht öffnen";
		return 3;
	}

	unsigned int signature;
	source.read(reinterpret_cast<char*>(&signature),sizeof(signature));
	if( signature == g_Signature )
	{
		cout << "Decodiere " << argv[1] << " zu " << argv[2] << endl;
		model->Process( &source, &target, MODE_DECODE );
	}
	else
	{
		cout << "Codiere " << argv[1] << " zu " << argv[2] << endl;
		source.seekg( 0, ios::beg );
		target.write( reinterpret_cast<const char*>(&g_Signature),
									sizeof(g_Signature) );
		model->Process( &source, &target, MODE_ENCODE );
	}

	source.close();
	target.close();

	return 0;
}