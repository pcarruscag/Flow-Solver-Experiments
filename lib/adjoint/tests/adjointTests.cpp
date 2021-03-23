//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "adjointTests.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sstream>

using namespace fileManagement;
using namespace geometryGeneration;
using namespace mesh;
using namespace flow;

namespace adjoint
{
void testSuite()
{
    std::cout << std::endl;
    std::cout << "### ADJOINT SOLVER TESTS ###" << std::endl;
    _adjointTest();
}

void _adjointTest()
{
    int nCases = 2;
    std::string basePath("../../lib/adjoint/tests/");

    for(int i=1; i<=nCases; ++i)
    {
        std::cout << std::endl;
        std::cout << "****** CASE " << i << " ******" << std::endl;

        std::stringstream casePath; casePath << basePath << "case" << i;

        std::string testEnvironVar("TESTDIR="+casePath.str());
        putenv(const_cast<char*>(testEnvironVar.c_str()));

        std::string command(casePath.str()+"/test.py");
        std::string result(casePath.str()+"/pass.inf");

        assert(system(command.c_str())==0);
        std::ifstream file;
        file.open(result);
        assert(file.good());
    }
}
}
