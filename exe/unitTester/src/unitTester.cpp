//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include <iostream>

#include "fileManagementTests.h"
#include "mathUtilsTests.h"
#include "bladeGeometryTests.h"
#include "meshTests.h"
#include "flowTests.h"
#include "adjointTests.h"
#include <omp.h>
#include <Eigen/Core>

int main(int argc, char* argv[])
{
    if(argc!=4) {
        std::cout << "Usage: " << argv[0] << " task  mode  np"  << std::endl
                  << "  task: -all   run all tests"             << std::endl
                  << "        -file  file management"           << std::endl
                  << "        -math  math utilities"            << std::endl
                  << "        -geom  geometry generator"        << std::endl
                  << "        -mesh  mesh generation"           << std::endl
                  << "        -flow  flow solver"               << std::endl
                  << "        -adj   adjoint solver"            << std::endl
                  << "  mode: -full  run expensive tests"       << std::endl
                  << "        -fast  don't run expensive tests" << std::endl
                  << "  np:   \"int\"  number of threads"       << std::endl
                  << std::endl;
        return 0;
    }

    std::string task(argv[1]), mode(argv[2]);
    int numThreads = atoi(argv[3]);

    bool expensiveTests = !strcmp(mode.c_str(),"-full");

    omp_set_num_threads(numThreads);
    Eigen::initParallel();

    if(!strcmp(task.c_str(),"-file") || !strcmp(task.c_str(),"-all"))
    {
        fileManagement::testSuite();
    }
    if(!strcmp(task.c_str(),"-math") || !strcmp(task.c_str(),"-all"))
    {
        mathUtils::testSuite();
    }
    if(!strcmp(task.c_str(),"-geom") || !strcmp(task.c_str(),"-all"))
    {
        geometryGeneration::testSuite();
    }
    if(!strcmp(task.c_str(),"-mesh") || !strcmp(task.c_str(),"-all"))
    {
        mesh::testSuite(expensiveTests);
    }
    if(!strcmp(task.c_str(),"-flow") || !strcmp(task.c_str(),"-all"))
    {
        flow::testSuite(expensiveTests);
    }
    if(!strcmp(task.c_str(),"-adj" ) || !strcmp(task.c_str(),"-all"))
    {
        adjoint::testSuite();
    }

    return 0;
}
