//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include <iostream>
#include <omp.h>
#include <fenv.h>

#include "adjoint.h"
#include "flow.h"
#include "geometryParameters.h"
#include "meshParameters.h"
#include "flowParameters.h"
#include "adjointParameters.h"
#include "bladeGeometry.h"
#include "passageMesh.h"
#include "unstructuredMesh.h"

using namespace fileManagement;
using namespace geometryGeneration;
using namespace mesh;
using namespace flow;
using namespace adjoint;

int main(int argc, char* argv[])
{
//    feenableexcept(FE_INVALID | FE_OVERFLOW);

    if(argc!=5){ std::cout
        << "Usage: " << argv[0] << "  task  dir  np  morph" << std::endl
        << "  task : -msh  run only the mesher"             << std::endl
        << "         -flw  include the flow solver"         << std::endl
        << "         -adj  include the adjoint solver"      << std::endl
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
         flowFlag = !strcmp(task.c_str(),"-flw"),
         adjFlag  = !strcmp(task.c_str(),"-adj");
    // validate input
    if(numParts<1 || !(meshFlag || flowFlag || adjFlag)){
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
    std::string adjInputFile(basePath+"adjointSettings.txt");
    std::string initFile(basePath+"initialConditions.dat");
    std::string initFileB(basePath+"initialConditionsB.dat");

    // Output files
    std::string meshFile(basePath+"mesh.txt");
    std::string flowFile(basePath+"flow.txt");
    std::string outputFile(basePath+"objectiveGradient.txt");

    // File managers
    GeometryParamManager geometryParams;
    MeshParamManager meshParams;
    FlowParamManager flowParams;
    AdjointParamManager adjointParams;

    // Read input files
    if(mrphFlag)
        assert(geometryParams.readFile(baseGeomFile)==0);
    else
        assert(geometryParams.readFile(geometryFile)==0);
    assert(meshParams.readFile(meshInputFile)==0);
    assert(flowParams.readFile(flowInputFile)==0);
    assert(adjointParams.readFile(adjInputFile)==0);

    std::string dxdparmFile;
    adjointParams.getScratchDir(dxdparmFile);
    dxdparmFile += "dxdparm.dat";

    // ajoint solver scope
    BladeGeometry geometry;
    PassageMesh meshGenerator(true);
    AdjointInputData adjData;
    PressureBasedCoupledSolverAdjoint adjointSolver;
    {
        // flow solver scope
        UnstructuredMesh master;
        std::vector<UnstructuredMesh> meshPartitions;
        PressureBasedCoupledSolver flowSolver;
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
            if(flowFlag || adjFlag){
            assert(meshGenerator.convertToUnstructured(master)==0);
            assert(master.partitioner(numParts,meshPartitions)==0);

            #pragma omp parallel num_threads(numParts)
            assert(meshPartitions[omp_get_thread_num()].computeGeoProps()==0);
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
        assert(flowSolver.setMesh(&meshPartitions)==0);
        assert(flowSolver.applyBndConds(flowParams)==0);
        if(!flowSolver.initialize(initFile))
            std::cout << std::endl << "Flow solver initialized" << std::endl;
        assert(flowSolver.run()==0);
        if(adjFlag) assert(flowSolver.getAdjointInput(master,&adjData)==0);

        std::cout << "Flow solver time: " << timer+omp_get_wtime() << std::endl;

        if(!flowSolver.saveState(initFile))
            std::cout << "Flow solver state saved" << std::endl;
        fileOut.open(flowFile.c_str());
        MatrixXf solution;
        flowSolver.getSolution(master,&solution);
        fileOut << solution << std::endl;
        fileOut.close();

        if(flowFlag) return 0;
    }   // end flow solver scope

    timer = -omp_get_wtime();

    assert(adjointSolver.setData(&adjData,adjointParams)==0);
    if(!adjointSolver.initialize(initFileB))
        std::cout << std::endl << "Adjoint solver initialized" << std::endl;
    assert(adjointSolver.run()==0);

    VectorXf objectives;
    assert(adjointSolver.getObjectiveVals(&objectives)==0);
    MatrixXf gradient;
    assert(adjointSolver.getDerivatives(&gradient)==0);

    std::cout << "Adjoint solver time: " << timer+omp_get_wtime() << std::endl;

    if(!adjointSolver.saveState(initFileB))
        std::cout << "Adjoint solver state saved" << std::endl;

    // chain rule with the grid jacobians
    const MatrixXf *xCoordJacobian_p, *yCoordJacobian_p, *zCoordJacobian_p;
    assert(meshGenerator.getJacobians(xCoordJacobian_p,yCoordJacobian_p,zCoordJacobian_p)==0);

    MatrixXf dxdparm(3*xCoordJacobian_p->rows(),xCoordJacobian_p->cols());
    for(int i=0; i<xCoordJacobian_p->rows(); ++i) {
      dxdparm.row(3*i  ) = xCoordJacobian_p->row(i);
      dxdparm.row(3*i+1) = yCoordJacobian_p->row(i);
      dxdparm.row(3*i+2) = zCoordJacobian_p->row(i);
    }
    gradient = dxdparm.transpose()*gradient;

    fileOut.open(outputFile.c_str());
    fileOut << objectives.transpose() << std::endl << std::endl << gradient << std::endl;
    fileOut.close();

    return 0;
}
