//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef FLOWTESTS_H
#define FLOWTESTS_H

#include "../src/flow.h"
#include "../../fileManagement/src/geometryParameters.h"
#include "../../fileManagement/src/meshParameters.h"
#include "../../fileManagement/src/flowParameters.h"
#include "../../geometry/src/bladeGeometry.h"
#include "../../mesh/src/passageMesh.h"
#include "../../mesh/src/unstructuredMesh.h"
#include "../../mesh/tests/meshTests.h"

#include <string>

namespace flow
{
    void testSuite(const bool expensiveTests=false);
    void _flowTest(const std::string caseNum, const bool extraCalculations);
    void _adjointInputTest();
}

#endif // FLOWTESTS_H
