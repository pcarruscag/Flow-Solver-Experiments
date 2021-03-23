//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef BLADEGEOMETRYGENERATOR_H
#define BLADEGEOMETRYGENERATOR_H

#include <vector>
#include <string>

#include "../../fileManagement/src/geometryParameters.h"
#include "../../fileManagement/src/meshParameters.h"
#include "../../mathUtils/src/splines.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

using namespace Eigen;

namespace mesh
{
    class PassageMesh;
    class PassageMeshInspector;
}

namespace geometryGeneration
{
    class BladeGeometry
    {
        friend class mesh::PassageMesh;
        friend class mesh::PassageMeshInspector;

        typedef AutoDiffScalar<VectorXd> adtype;
      public:
        BladeGeometry();
        int buildGeometry(const fileManagement::GeometryParamManager &geoParams,
                          const fileManagement::MeshParamManager &meshParams,
                          const bool fixSize = false);
        int saveGeometry(const std::string &filePath) const;
        int saveJacobian(const std::string &filePath) const;
      private:
        int m_errorCode, m_nIndepVars, m_nPtMeri, m_leIndex, m_teIndex;
        mathUtils::interpolation::Cspline<adtype> m_zHubSpline, m_rHubSpline, m_zShrSpline, m_rShrSpline;
        std::vector<mathUtils::interpolation::Cspline<adtype> > m_meri2tSplines;
        std::vector<Matrix<double,Dynamic,1> > m_meriCoord, m_pitchCoord;
        std::vector<Matrix<double,Dynamic,Dynamic> > m_meriCoordDeriv, m_pitchCoordDeriv;

        // Returns the meridional length of each spanwise section
        void m_buildMeri2tSplines(const fileManagement::GeometryParamManager &geoParams,
                                  const mathUtils::interpolation::Cspline<adtype> &zHubSpline,
                                  const mathUtils::interpolation::Cspline<adtype> &rHubSpline,
                                  const mathUtils::interpolation::Cspline<adtype> &zShrSpline,
                                  const mathUtils::interpolation::Cspline<adtype> &rShrSpline,
                                  std::vector<mathUtils::interpolation::Cspline<adtype> > &meri2tSplines,
                                  std::vector<adtype> &meriLen);
    };
}

#endif // BLADEGEOMETRYGENERATOR_H
