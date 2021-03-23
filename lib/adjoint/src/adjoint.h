//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef ADJOINT_H
#define ADJOINT_H

#include "../../flow/src/flow.h"
#include "../../fileManagement/src/adjointParameters.h"

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace flow;
using namespace fileManagement;

namespace adjoint
{
    class PressureBasedCoupledSolverAdjoint
    {
        // numerical precision used during solution of adjoint system
        using numPrec = float;

        // a block sparse storage format to store the flow jacobian
        struct flowJacobianMatrixType {
            ArrayXi outerIdxPtr;
            ArrayXi cellIdx;
            Matrix<numPrec,Dynamic,4> coefficients;
            // cell coloring data for vector-matrix multiplication
            static constexpr int color_groupSize = 4;
            std::vector<int> color_outerIdxPtr;
            std::vector<int> color_cellIdx;
            // action of the jacobian on a vector/matrix
            Matrix<numPrec,Dynamic,Dynamic> product;
            double timer;
            void gemm(const int nCols, const numPrec* mat);
            void free();
            void writeToFile(const std::string &filePath);
        };
      public:
        enum ObjectiveFcn {MDOT_IN,   MDOT_OUT,
                           BLOCKAGE,  THRUST,    TORQUE,
                           DELTAP_TT, DELTAP_TS, DELTAP_SS,
                           ETA_TT,    ETA_TS,    ETA_SS};
        enum VariableType {FLOW, GEOMETRY};

        PressureBasedCoupledSolverAdjoint();
        int setData(const AdjointInputData* inData_p,
                    const AdjointParamManager control);
        int initialize(const std::string &filePath);
        int run();
        int getObjectiveVals(Matrix<numPrec,Dynamic,1>* values) const;
        int getDerivatives(Matrix<numPrec,Dynamic,Dynamic>* gradient) const;
        int getAdjointVars(Matrix<numPrec,Dynamic,Dynamic>* adjVars ) const;
        int getObjectiveJacobian(const VariableType varType,
                           Matrix<numPrec,Dynamic,Dynamic>* jacobian) const;
        int saveState(const std::string &filePath) const;
      private:
        const AdjointInputData *m_inData_p;
        AdjointParamManager m_control;
        // calculation data
        flowJacobianMatrixType          m_flowJacobian;
        Matrix<numPrec,Dynamic,Dynamic> m_objGeoJacobian;
        Matrix<numPrec,Dynamic,Dynamic> m_objFlowJacobian;
        // results
        Matrix<numPrec,Dynamic,1>       m_objValues;
        Matrix<numPrec,Dynamic,Dynamic> m_adjointVars;
        Matrix<numPrec,Dynamic,Dynamic> m_objGradient;
        // parameters
        std::vector<int> m_objectiveList;
        bool m_firstOrder;
        bool m_dataIsSet;
        bool m_isInit;
        bool m_finished;
        int  m_partNum;
        int  m_objNum;
        // timers
        double m_tFlwJac, m_tGeoJac, m_tObjJac, m_tSolve;

        // cell based connectivity used to search for neighbours of a cell
        void m_buildConnectivity();
        std::vector<std::vector<int> > m_connectivity_F;

        // functions involved in computing jacobians
        int m_computeObjective();
        int m_computeResidualJacobianFlw();
        int m_computeResidualJacobianGeo();
        template<VariableType T> // differentiate w.r.t variable type T
        int m_computeResidualForCell(const int cell,
                                     Vector4f& residual,
                                     Vector4f& residualNorm,
                                     std::vector<int>& cells,
                                     std::vector<int>& cellDegEndIdx,
                                     std::vector<int>& vertices,
                                     Matrix<numPrec,Dynamic,4>& deriv) const;
        template<bool cellsOnly = false>
        int m_gatherNeighbourhood(const int cell,
                                  const int maxDegree,
                                  std::vector<int>& vertices,
                                  std::vector<int>& faces,
                                  std::vector<int>& faceDegEndIdx,
                                  std::vector<int>& cells,
                                  std::vector<int>& cellDegEndIdx) const;
        int m_allocateFlowJacobian();
        template<typename RHS_t> int m_solveAdjointSystem();
    };
}

#endif // ADJOINT_H
