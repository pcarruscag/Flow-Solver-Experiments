//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef FLOW_SIMPLE_H
#define FLOW_SIMPLE_H

// Experiment: Write a solver in a shared-memory vectorized way, at least the main loop.
// It will be an incompressible pressure-based segregated solver, SIMPLE algorithm.
// Being an experiment it will be a very barebones implementation.

#include "../../mesh/src/unstructuredMesh.h"
#include "../../fileManagement/src/flowParameters.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace fileManagement;
using namespace mesh;
using namespace Eigen;

namespace flow
{
    typedef int int_t;
    typedef float float_t;

    typedef Matrix<float_t,Dynamic,1> vectorXf_t;
    typedef Matrix<float_t,Dynamic,3> matrixX3f_t;
    typedef SparseMatrix<float_t,RowMajor> matrixCSRf_t;
    typedef Map<matrixCSRf_t> mappedCSRf_t;

    typedef Matrix<int_t,Dynamic,1> vectorXi_t;
    typedef SparseMatrix<int_t,RowMajor> matrixCSRi_t;

    class PressureBasedSegregatedSolver
    {
      public:
        struct flowField
        {
            vectorXf_t u, v, w, p;
        };
        struct gradients
        {
            matrixX3f_t u, v, w, p;
        };
      private:
        struct boundaries
        {
            int number;
            VectorXi type;
            MatrixXf conditions;
        };
        struct adjacency
        {
            std::vector<int_t> outerIndex;
            std::vector<int_t> cells;
            std::vector<int_t> faces;
        };
        struct gradCalcData
        {
            vectorXf_t ax;
            vectorXf_t ay;
            vectorXf_t az;
            vectorXf_t dir;
        };
        struct matAssembler
        {
            std::vector<int> diagonal;
            std::vector<int> nz0pos;
            std::vector<int> nz0face;
            std::vector<int> nz1pos;
            std::vector<int> nz1face;
            int Nnz0, Nnz1;
            float_t* nzPtr;
            vectorXf_t momDiagCoeff;
        };
      public:
        PressureBasedSegregatedSolver();
        void setControl(const FlowParamManager& flowParams);
        int setMesh(UnstructuredMesh* domain);
        int applyBndConds(const FlowParamManager& flowParams);
        int run();
        int saveState(const std::string &filePath) const;
        int getSolution(MatrixXf* solution_p) const;
      private:
        bool m_ctrlIsSet;
        bool m_meshIsSet;
        bool m_bndrsSet;
        bool m_isInit;
        bool m_finished;

        int m_allocate(); // variables that use significant amounts of memory are allocated here
        void m_boundaryValues();
        void m_computeLimiters(const float_t* phiC_ptr, const float_t* phiF_ptr,
                               const matrixX3f_t* nablaPhi_ptr, float_t* limit_ptr) const;
        void m_compute2ndOrderFluxes();

        // preprocessing methods
        int m_buildAdjacency();
        int m_buildFaceIterp();
        int m_buildGradCalc();
        int m_buildMatAssembler();
        int m_defaultInit();

        fileManagement::FlowParamManager::controlParameters m_control;

        float_t m_rho;
        float_t m_mu;

        UnstructuredMesh* m_mesh;
        flowField m_flwFld_C; // at the cell centroid
        flowField m_flwFld_F; // at the face centroid
        gradients m_nabla;
        flowField m_limit; // flux limiters
        boundaries m_boundaries;

        adjacency m_adjacency;

        // linear system variables
        matrixCSRf_t m_systemMat;
        matrixX3f_t m_momentumSrc, m_momentumSol;
        vectorXf_t m_pcorrectSrc, m_pcorrectSol;

        matrixCSRf_t m_faceInterpMat; // interpolate from cells to faces
        flowField    m_faceGivenVals; // given face values, e.g. bc's or grid velocities

        // compute gradients
        gradCalcData m_gradCalcData;
        mappedCSRf_t m_gradCalcX;
        mappedCSRf_t m_gradCalcY;
        mappedCSRf_t m_gradCalcZ;

        // momentum fluxes variables
        vectorXf_t m_mdot;
        vectorXf_t m_diffCoeff;
        matrixX3f_t m_2ndOrderFluxes;
        matAssembler m_matAssembler;

        mappedCSRf_t m_sumFluxes;
    };
}
#endif // FLOW_SIMPLE_H
