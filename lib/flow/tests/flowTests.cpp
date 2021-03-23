//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "flowTests.h"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <omp.h>
#include <Eigen/Dense>
#include <math.h>
#include <stdlib.h>

#define TEST_TOL 0.1
#define TEST_DIR "../../lib/flow/tests/case"

using namespace fileManagement;
using namespace geometryGeneration;
using namespace mesh;
using namespace flow;
using namespace Eigen;

namespace flow
{
void testSuite(const bool expensiveTests)
{
    std::cout << std::endl;
    std::cout << "### FLOW SOLVER TESTS ###" << std::endl;
    _adjointInputTest();
    _flowTest("2",true);
    _flowTest("3",true);
    _flowTest("4",true);
    _flowTest("5",true);
    if(expensiveTests)
    {
        _flowTest("1",true);
        _flowTest("6",true);
        _flowTest("7",true);
    }
}

void _adjointInputTest()
{
    std::cout << std::endl;
    std::cout << "****** ADJOINT INPUT ******" << std::endl;
    int numParts = omp_get_max_threads();

    // Input files
    std::string basePath("../../lib/flow/tests/adjointInput/");
    std::string geometryFile( basePath+"geomParams.txt");
    std::string meshInputFile(basePath+"meshSettings.txt");
    std::string flowInputFile(basePath+"flowSettings.txt");

    // File managers
    GeometryParamManager geometryParams;
    MeshParamManager meshParams;
    FlowParamManager flowParams;

    // Objects
    BladeGeometry geometry;
    PassageMesh meshGenerator;
    PassageMeshInspector meshInspector(&meshGenerator);
    UnstructuredMesh master;
    std::vector<UnstructuredMesh> meshPartitions;
    PressureBasedCoupledSolver flowSolver;
    AdjointInputData adjData, adjData2;

    // Read input files
    assert(geometryParams.readFile(geometryFile)==0);
    assert(meshParams.readFile(meshInputFile)==0);
    assert(flowParams.readFile(flowInputFile)==0);

    // Build and mesh geometry
    assert(geometry.buildGeometry(geometryParams,meshParams)==0);
    assert(meshGenerator.mesh(geometry,meshParams)==0);

    // Create and initialize partitions
    std::vector<double> partitionSizes(numParts,1.0/double(numParts));
    assert(meshGenerator.convertToUnstructured(master)==0);
    assert(master.scale(0.001)==0);
    assert(master.partitioner(numParts,meshPartitions)==0);

    #pragma omp parallel num_threads(numParts)
    assert(meshPartitions[omp_get_thread_num()].computeGeoProps()==0);

    // Run the solver and get results
    flowSolver.setControl(flowParams);
    assert(flowSolver.setMesh(&meshPartitions)==0);
    assert(flowSolver.applyBndConds(flowParams)==0);
    assert(flowSolver.run()==0);
    assert(flowSolver.getAdjointInput(master,&adjData)==0);

    // Checksum the results
    float domainVolume = 0.0f, sumU = 0.0f, sumV = 0.0f, sumW = 0.0f, sumP = 0.0f;
    #pragma omp parallel reduction(+:domainVolume,sumU,sumV,sumW,sumP)
    {
        int partNum = omp_get_thread_num();
        domainVolume += meshPartitions[partNum].cells.volume.sum();
        sumU += flowSolver.m_flwFld_C[partNum].u.sum();
        sumV += flowSolver.m_flwFld_C[partNum].v.sum();
        sumW += flowSolver.m_flwFld_C[partNum].w.sum();
        sumP += flowSolver.m_flwFld_C[partNum].p.sum();
    }
    assert(std::abs(adjData.u.sum()/sumU-1.0f)<1.0e-5f);
    assert(std::abs(adjData.v.sum()/sumV-1.0f)<1.0e-5f);
    assert(std::abs(adjData.w.sum()/sumW-1.0f)<1.0e-5f);
    assert(std::abs(adjData.p.sum()/sumP-1.0f)<1.0e-5f);

    // setup an unstructured mesh with the adjointData
    UnstructuredMesh testDomain;
    testDomain.vertices.number = adjData.vertexNumber;
    testDomain.vertices.coords = adjData.verticesCoord;
    testDomain.faces.number = adjData.faceNumber;
    testDomain.faces.connectivity = adjData.connectivity;
    testDomain.faces.verticesStart = adjData.verticesStart;
    testDomain.faces.verticesIndex = adjData.verticesIndex;
    testDomain.faces.groups.resize(7);
    for(int i=0; i<7; i++)
        testDomain.faces.groups[i].first = i;
    for(int i=0; i<adjData.boundaryNumber; ++i)
        switch(adjData.boundaryType(i))
        {
          case 1:
            testDomain.faces.groups[2].second.push_back(i);
            break;
          case 3:
            testDomain.faces.groups[3].second.push_back(i);
            break;
          case 5:
            testDomain.faces.groups[6].second.push_back(i);
            break;
          case 7:
            if(adjData.boundaryConditions(i,1)==1)
                testDomain.faces.groups[4].second.push_back(i);
            else
                testDomain.faces.groups[5].second.push_back(i);
            break;
          default:
            assert(false);
        }
    testDomain.cells.number = adjData.cellNumber;
    testDomain.m_meshIsDefined = true;

    assert(testDomain.computeGeoProps()==0);
    assert(std::abs(testDomain.cells.volume.sum()/domainVolume-1.0f)<1.0e-5f);

    // Verify periodic mapping
    for(int i=0; i<adjData.boundaryNumber; ++i)
        if(adjData.boundaryType(i)==7)
        {
            int j = adjData.boundaryConditions(i,4);
            assert(adjData.boundaryType(j)==7);

            float tol = 0.001f*std::sqrt(testDomain.faces.area.row(i).norm());

            float zi = testDomain.faces.centroid(i,2);
            float zj = testDomain.faces.centroid(j,2);
            float ri = testDomain.faces.centroid.block(i,0,1,2).norm();
            float rj = testDomain.faces.centroid.block(j,0,1,2).norm();

            assert(std::abs(zi-zj)<tol);
            assert(std::abs(ri-rj)<tol);
        }

    // Run the solver with the mesh from the adjoint data
    assert(testDomain.partitioner(numParts,meshPartitions,true)==0);
    #pragma omp parallel num_threads(numParts)
    assert(meshPartitions[omp_get_thread_num()].computeGeoProps()==0);

    PressureBasedCoupledSolver flowSolver2;
    flowSolver2.setControl(flowParams);
    assert(flowSolver2.setMesh(&meshPartitions)==0);
    assert(flowSolver2.applyBndConds(flowParams)==0);
    assert(flowSolver2.run()==0);
    assert(flowSolver2.getAdjointInput(testDomain,&adjData2)==0);

    assert((adjData.u-adjData2.u).norm()/adjData.u.norm()<0.1f);
    assert((adjData.v-adjData2.v).norm()/adjData.v.norm()<0.1f);
    assert((adjData.w-adjData2.w).norm()/adjData.w.norm()<0.1f);
    assert((adjData.p-adjData2.p).norm()/adjData.p.norm()<0.1f);
}

void _flowTest(const std::string caseNum, const bool extraCalculations)
{
    std::cout << std::endl;
    std::cout << "****** CASE " << caseNum << " ******" << std::endl;
    int numParts = omp_get_max_threads();

    std::string externalComp(TEST_DIR+caseNum+"/compare.sh");
    std::string errorFile(TEST_DIR+caseNum+"/error.txt");
    std::string testEnvironVar("TESTDIR=");
    testEnvironVar += TEST_DIR+caseNum;
    putenv(const_cast<char*>(testEnvironVar.c_str()));

    // Input files
    std::string geometryFile( TEST_DIR+caseNum+"/geomParams.txt");
    std::string meshInputFile(TEST_DIR+caseNum+"/meshSettings.txt");
    std::string flowInputFile(TEST_DIR+caseNum+"/flowSettings.txt");

    // Output files
    std::string meshOutFile(TEST_DIR+caseNum+"/mesh.txt");
    std::string flowOutFile(TEST_DIR+caseNum+"/results.txt");

    // File managers
    GeometryParamManager geometryParams;
    MeshParamManager meshParams;
    FlowParamManager flowParams;

    // Objects
    BladeGeometry geometry;
    PassageMesh meshGenerator;
    PassageMeshInspector meshInspector(&meshGenerator);
    UnstructuredMesh master;
    std::vector<UnstructuredMesh> meshPartitions;
    PressureBasedCoupledSolver flowSolver;

    // Read input files
    assert(geometryParams.readFile(geometryFile)==0);
    assert(meshParams.readFile(meshInputFile)==0);
    assert(flowParams.readFile(flowInputFile)==0);

    // Build and mesh geometry
    assert(geometry.buildGeometry(geometryParams,meshParams)==0);
    assert(meshGenerator.mesh(geometry,meshParams)==0);
    meshInspector.save3Dgrid(meshOutFile);

    PassageMesh::meshInfo info = meshGenerator.getMeshInfo();
    std::cout << std::endl << "Nmeri:  " << info.nMeri
              << std::endl << "Npitch: " << info.nPitch
              << std::endl << "Nspan:  " << info.nSpan
              << std::endl << "jLE:    " << info.jLE
              << std::endl << "jTE:    " << info.jTE
              << std::endl << "SIZE:   " << (info.nMeri-1)*(info.nPitch-1)*(info.nSpan-1)
              << std::endl;

    // Create and initialize partitions
    assert(meshGenerator.convertToUnstructured(master)==0);
    assert(master.partitioner(numParts,meshPartitions)==0);

    #pragma omp parallel
    {
        int partNum = omp_get_thread_num();
        meshPartitions[partNum].scale(0.001);
        assert(meshPartitions[partNum].computeGeoProps()==0);
    }

    // Run the solver
    flowSolver.setControl(flowParams);
    assert(flowSolver.setMesh(&meshPartitions)==0);
    assert(flowSolver.applyBndConds(flowParams)==0);

    double time = -omp_get_wtime();
    assert(flowSolver.run()==0);
    time += omp_get_wtime();

    // Get results
    MatrixXf flowField;
    flowSolver.getSolution(master,&flowField);
    std::ofstream file;
    file.open(flowOutFile.c_str());
    file << flowField << std::endl;
    file.close();

    // Compare results with reference
    ArrayXd errNorms(6);
    assert(system("cp $TESTDIR/../compare.sh $TESTDIR/compare.sh")==0);
    assert(system(externalComp.c_str())==0);
    {
        std::ifstream errFile;
        errFile.open(errorFile);
        double errU, errV, errW, errP, errT1, errT2;
        errFile >> errU >> errV >> errW >> errP >> errT1 >> errT2;
        errFile.close();
        assert(system(("rm "+errorFile).c_str())==0);
        errNorms << errU, errV, errW, errP, errT1, errT2;
    }
    assert(system("rm $TESTDIR/compare.sh")==0);
    std::cout << "Comparison with reference results:" << std::endl;
    std::cout << errNorms << std::endl;
    assert((errNorms.head(4) < TEST_TOL).all());
    assert((errNorms.tail(2) < TEST_TOL*2.0).all());

    // Conservation checks
    float massIn = 0.0, massOut = 0.0, massPeriodics = 0.0;
    #pragma omp parallel reduction(+:massIn,massOut,massPeriodics)
    {
        int partNum = omp_get_thread_num();

        // Mass flow in equal to mass flow out
        for(int i=0; i<flowSolver.m_boundaries[partNum].number; ++i)
        {
            float flux = flowSolver.m_flwFld_F[partNum].u(i)*meshPartitions[partNum].faces.area(i,0)+
                         flowSolver.m_flwFld_F[partNum].v(i)*meshPartitions[partNum].faces.area(i,1)+
                         flowSolver.m_flwFld_F[partNum].w(i)*meshPartitions[partNum].faces.area(i,2);
            if(flowSolver.m_boundaries[partNum].type(i)==1) massIn        += flux;
            if(flowSolver.m_boundaries[partNum].type(i)==3) massOut       += flux;
            if(flowSolver.m_boundaries[partNum].type(i)==7) massPeriodics += flux;
        }
    }

    std::cout << "Mass In\\Out: " << std::abs(massIn/massOut+1.0f) << std::endl;
    std::cout << "Mass Periodics: " << std::abs(massPeriodics/massOut) << std::endl;
    assert(std::abs(massIn/massOut+1.0f) < 0.0025f);
    assert(std::abs(massPeriodics/massOut) < 0.0025f);

    if(extraCalculations)
    {
    float areaWall = 0.0, areaIntYplus = 0.0, areaIn = 0.0, areaIntP = 0.0;
    #pragma omp parallel reduction(+:areaWall,areaIntYplus,areaIn,areaIntP)
    {
        int prt = omp_get_thread_num();
        for(int i=0; i<flowSolver.m_boundaries[prt].number; ++i)
        {
            if(flowSolver.m_boundaries[prt].type(i)==5)
            {
                areaWall += meshPartitions[prt].faces.area.row(i).norm();
                areaIntYplus += meshPartitions[prt].faces.area.row(i).norm()*
                                flowSolver.m_boundaries[prt].conditions(i,5);
            }
            if(flowSolver.m_boundaries[prt].type(i)<=2)
            {
                areaIn   += meshPartitions[prt].faces.area.row(i).norm();
                areaIntP += meshPartitions[prt].faces.area.row(i).norm()*flowSolver.m_flwFld_F[prt].p(i);
            }
        }
    }

    float mdot = 0.0, delta_rVt = 0.0, delta_p0 = 0.0;
    #pragma omp parallel reduction(+:mdot,delta_rVt,delta_p0)
    {
        int prt = omp_get_thread_num();
        for(int i=0; i<flowSolver.m_boundaries[prt].number; ++i)
        {
            int bndType = flowSolver.m_boundaries[prt].type(i);

            if(bndType<=4) // in out
            {
                float u = flowSolver.m_flwFld_F[prt].u(i),
                      v = flowSolver.m_flwFld_F[prt].v(i),
                      w = flowSolver.m_flwFld_F[prt].w(i),
                      p = flowSolver.m_flwFld_F[prt].p(i);
                float flux = u*meshPartitions[prt].faces.area(i,0)+
                             v*meshPartitions[prt].faces.area(i,1)+
                             w*meshPartitions[prt].faces.area(i,2);

                u -= flowSolver.m_rotationalSpeed*meshPartitions[prt].faces.centroid(i,1);
                v += flowSolver.m_rotationalSpeed*meshPartitions[prt].faces.centroid(i,0);
                float rVt = v*meshPartitions[prt].faces.centroid(i,0)-u*meshPartitions[prt].faces.centroid(i,1);

                float p0 = p+0.5f*flowSolver.m_rho*(std::pow(u,2.0f)+std::pow(v,2.0f)+std::pow(w,2.0f));

                delta_rVt += flux*rVt;
                delta_p0  += flux*p0;
                if(bndType==3) mdot += flux;
            }
        }
    }
    delta_rVt /= mdot; delta_p0 /= mdot;
    float eff = delta_p0/(flowSolver.m_rho*flowSolver.m_rotationalSpeed*delta_rVt);

    std::cout << "Inlet Pressure: " << areaIntP/areaIn << std::endl;
    std::cout << "Pressure Rise: " << delta_p0 << std::endl;
    std::cout << "Efficiency: " << eff << std::endl;
    std::cout << "Area ave y+: " << areaIntYplus/areaWall << std::endl;
    }
    std::cout << "Solution time: " << time << std::endl;
}
}
