//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef MESHTESTS_H
#define MESHTESTS_H

#include "../src/passageMesh.h"
#include "../src/unstructuredMesh.h"
#include <string>

namespace mesh
{
    void testSuite(const bool expensiveTests=false);
    void _meshTests(const bool expensiveTests);

    class PassageMeshInspector
    {
      public:
        PassageMeshInspector(PassageMesh* obj) : m_obj(obj) {}
        ~PassageMeshInspector() {}

        int finDiffVerification(fileManagement::GeometryParamManager& geoParams,
                                const fileManagement::MeshParamManager& mshParams,
                                const double dx);
        int finDiffVerifBladeGeo(fileManagement::GeometryParamManager& geoParams,
                                const fileManagement::MeshParamManager& mshParams,
                                const double dx1, const double dx2);

        int save3Dgrid(const std::string& filePath) const;
        int save3DgridJacobian(const std::string& filePath) const;
        int comp3DgridJacobian(const std::string& filePath, const double tol) const;
        int save2Dlayers(const std::string& filePath) const;
        int save2DlayerJacobian(const std::string& filePath) const;
        int comp2DlayerJacobian(const std::string& filePath, const double tol) const;
      private:
        PassageMeshInspector();
        PassageMesh* m_obj;
    };
}

#endif // MESHTESTS_H
