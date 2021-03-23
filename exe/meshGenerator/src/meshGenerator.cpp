//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

// Experiment: Exports the passage grids to SU2 format.
// Warning: Never used for "production".

#include <iostream>
#include <string>
#include <omp.h>

#include "geometryParameters.h"
#include "meshParameters.h"
#include "bladeGeometry.h"
#include "passageMesh.h"

using namespace fileManagement;
using namespace geometryGeneration;
using namespace mesh;

int main(int argc, char* argv[])
{
    if(argc!=4) {
        std::cout << "Usage: " << argv[0] << " input1 input2 output" << std::endl
                  << "  input1: path to the geometry settings file"  << std::endl
                  << "  input2: path to the   mesh   settings file"  << std::endl
                  << "  output: path to the   mesh    output  file"  << std::endl
                  << std::endl;
        return 0;
    }

    std::string geometryFile(argv[1]),
                meshInputFile(argv[2]),
                meshOutFile(argv[3]);

    GeometryParamManager geomParams;
    MeshParamManager     meshParams;
    BladeGeometry        bladeGeom;
    PassageMesh          mesher;

    assert(geomParams.readFile(geometryFile) ==0 && "error reading geometry file");
    assert(meshParams.readFile(meshInputFile)==0 && "error reading mesh settings");

    assert(bladeGeom.buildGeometry(geomParams,meshParams)==0 && "error buiding geometry");

    assert(mesher.mesh(bladeGeom,meshParams)==0 && "error meshing geometry");
    mesher.scale(0.001,0);

    PassageMesh::meshInfo info = mesher.getMeshInfo();
    std::cout << std::endl << "Nmeri:  " << info.nMeri
              << std::endl << "Npitch: " << info.nPitch
              << std::endl << "Nspan:  " << info.nSpan
              << std::endl << "jLE:    " << info.jLE
              << std::endl << "jTE:    " << info.jTE
              << std::endl << "SIZE:   " << (info.nMeri-1)*(info.nPitch-1)*(info.nSpan-1)
              << std::endl;

    assert(mesher.saveSU2file(meshOutFile)==0 && "error saving output file");

    return 0;
}
