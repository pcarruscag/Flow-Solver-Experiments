//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include <iostream>
#include <omp.h>

#include "geometryParameters.h"
#include "meshParameters.h"
#include "flowParameters.h"
#include "bladeGeometry.h"
#include "passageMesh.h"
#include "unstructuredMesh.h"

#ifdef GPU
    #include "flow_gpu.h"
#else
    #include "flow_simple.h"
#endif

using namespace fileManagement;
using namespace geometryGeneration;
using namespace mesh;
using namespace flow;

int main(int argc, char* argv[])
{
    if(argc!=5){ std::cout
        << "Usage: " << argv[0] << "  task  dir  np  morph" << std::endl
        << "  task : -msh  run only the mesher"             << std::endl
        << "         -flw  include the flow solver"         << std::endl
        << "  dir  : directory with the input files"        << std::endl
        << "  np   : number of threads (int)"               << std::endl
        << "  morph: morph mesh (0/1)"                      << std::endl
        << std::endl;
        return 1;
    }
    std::string task(argv[1]), basePath(argv[2]);
    int numParts = atoi(argv[3]);
    bool mrphFlag = bool(atoi(argv[4])),
         meshFlag = !strcmp(task.c_str(),"-msh"),
         flowFlag = !strcmp(task.c_str(),"-flw");
    // validate input
    if(numParts<1 || !(meshFlag || flowFlag)){
        std::cout << "Invalid options" << std::endl;
        return 1;
    }
    double timer;
    std::ofstream fileOut;

    omp_set_num_threads(numParts);
    Eigen::initParallel();

    // Input files
    std::string baseGeomFile( basePath+"baseParams.txt");
    std::string geometryFile( basePath+"geomParams.txt");
    std::string meshInputFile(basePath+"meshSettings.txt");
    std::string flowInputFile(basePath+"flowSettings.txt");
    std::string initFile(basePath+"initialConditions.dat");

    // Output files
    std::string meshFile(basePath+"mesh.txt");
    std::string flowFile(basePath+"flow.txt");

    // File managers
    GeometryParamManager geometryParams;
    MeshParamManager meshParams;
    FlowParamManager flowParams;

    // Read input files
    if(mrphFlag)
        assert(geometryParams.readFile(baseGeomFile)==0);
    else
        assert(geometryParams.readFile(geometryFile)==0);
    assert(meshParams.readFile(meshInputFile)==0);
    assert(flowParams.readFile(flowInputFile)==0);

    BladeGeometry geometry;
    PassageMesh meshGenerator(true);
    UnstructuredMesh master;
    PressureBasedSegregatedSolver flowSolver;
    {
        // mesh generation scope
        // Build and mesh geometry
        timer = -omp_get_wtime();

        assert(geometry.buildGeometry(geometryParams,meshParams)==0);
        assert(meshGenerator.mesh(geometry,meshParams)==0);
        if(mrphFlag) {
            assert(geometryParams.readFile(geometryFile)==0);
            assert(geometry.buildGeometry(geometryParams,meshParams,true)==0);
            assert(meshGenerator.morph(geometry)==0);
        }
        GeometryParamManager::geometryInfo geoInfo = geometryParams.getGeometryInfo();
        assert(meshGenerator.scale(0.001f,2*(geoInfo.nCPhub+geoInfo.nCPshroud))==0);

        // Create and initialize partitions
        if(flowFlag){
        assert(meshGenerator.convertToUnstructured(master)==0);
        VectorXi permutation;
        master.rcmOrdering(permutation);
        master.renumberCells(permutation);
        master.computeGeoProps();
        }
        PassageMesh::meshInfo info = meshGenerator.getMeshInfo();
        std::cout << std::endl << "Nmeri:  " << info.nMeri
                  << std::endl << "Npitch: " << info.nPitch
                  << std::endl << "Nspan:  " << info.nSpan
                  << std::endl << "jLE:    " << info.jLE
                  << std::endl << "jTE:    " << info.jTE
                  << std::endl << "SIZE:   " << (info.nMeri-1)*(info.nPitch-1)*(info.nSpan-1)
                  << std::endl << "Time:   " << timer+omp_get_wtime()
                  << std::endl << std::endl;

        // save the grid for post-processing
        fileOut.open(meshFile.c_str());
        const MatrixXf *grid_p;
        meshGenerator.getGrid(grid_p);
        fileOut << *grid_p << std::endl;
        fileOut.close();

        if(meshFlag) return 0;
    }   // end mesh generation scope

    // Run the solver and get results
    timer = -omp_get_wtime();

    flowSolver.setControl(flowParams);
    assert(flowSolver.setMesh(&master)==0);
    assert(flowSolver.applyBndConds(flowParams)==0);
//    if(!flowSolver.initialize(initFile))
//        std::cout << std::endl << "Flow solver initialized" << std::endl;
    assert(flowSolver.run()==0);
//    if(!flowSolver.saveState(initFile))
//        std::cout << "Flow solver state saved" << std::endl;
    fileOut.open(flowFile.c_str());
    MatrixXf solution;
    flowSolver.getSolution(&solution);
    fileOut << solution << std::endl;
    fileOut.close();

    std::cout << "Flow solver time: " << timer+omp_get_wtime() << std::endl;

    return 0;
}
