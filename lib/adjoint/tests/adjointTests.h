//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef ADJOINTTESTS_H
#define ADJOINTTESTS_H

#include "../src/adjoint.h"
#include "../../flow/src/flow.h"
#include "../../fileManagement/src/geometryParameters.h"
#include "../../fileManagement/src/meshParameters.h"
#include "../../fileManagement/src/flowParameters.h"
#include "../../fileManagement/src/adjointParameters.h"
#include "../../geometry/src/bladeGeometry.h"
#include "../../mesh/src/passageMesh.h"
#include "../../mesh/src/unstructuredMesh.h"

namespace adjoint
{
    void testSuite();
    void _adjointTest();
}

#endif // ADJOINTTESTS_H
