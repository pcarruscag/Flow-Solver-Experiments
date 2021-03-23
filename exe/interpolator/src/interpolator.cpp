//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

// Experiment: Map one grid to another in a nearest neighbor sense.
// This uses connectivity information and so has O(N) complexity.

#include <iostream>
#include <fstream>
#include <omp.h>

#include "geometryParameters.h"
#include "meshParameters.h"
#include "bladeGeometry.h"
#include "passageMesh.h"
#include "unstructuredMesh.h"

using namespace fileManagement;
using namespace geometryGeneration;
using namespace mesh;

int main(int argc, char* argv[])
{
    if(argc!=3){ std::cout
        << "Usage: " << argv[0] << "  dir  np" << std::endl
        << "  dir  : directory with the input files"        << std::endl
        << "  np   : number of threads (int)"               << std::endl
        << std::endl;
        return 1;
    }
    std::string basePath(argv[1]);
    int numParts = atoi(argv[2]);

    omp_set_num_threads(numParts);
    Eigen::initParallel();

    // Input files
    std::string geometryFile(basePath+"geomParams.txt");
    std::string meshInputFile(basePath+"meshSettings.txt");
    std::string donorInputFile(basePath+"donorSettings.txt");

    // File managers
    GeometryParamManager geometryParams;
    MeshParamManager meshParams, donorParams;

    // Read input files
    assert(geometryParams.readFile(geometryFile)==0);
    assert(meshParams.readFile(meshInputFile)==0);
    assert(donorParams.readFile(donorInputFile)==0);


    // Create donor and target meshes
    BladeGeometry geometry;
    PassageMesh meshGenerator(true);
    UnstructuredMesh target, donor;
    VectorXi permutation;

    assert(geometry.buildGeometry(geometryParams,meshParams)==0);
    assert(meshGenerator.mesh(geometry,meshParams)==0);
    assert(meshGenerator.convertToUnstructured(target)==0);
    assert(target.rcmOrdering(permutation,true,false)==0);
    assert(target.renumberCells(permutation)==0);
    assert(target.computeGeoProps()==0);

    assert(geometry.buildGeometry(geometryParams,donorParams)==0);
    assert(meshGenerator.mesh(geometry,donorParams)==0);
    assert(meshGenerator.convertToUnstructured(donor)==0);
    assert(donor.rcmOrdering(permutation,true,false)==0);
    assert(donor.renumberCells(permutation)==0);
    assert(donor.computeGeoProps()==0);


    // Map donor onto target
    double time = -omp_get_wtime();
    VectorXi map;
    std::cout << std::endl;
    assert(target.nearNeighbourMap(donor,map)==0);
    std::cout << time+omp_get_wtime() << std::endl;


    std::ofstream file;
    file.open("map.txt");
    file << map << std::endl;
    file.close();

    return 0;
}
