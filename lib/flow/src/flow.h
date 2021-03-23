//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef FLOW_H
#define FLOW_H

#include "../../mesh/src/unstructuredMesh.h"
#include "../../fileManagement/src/flowParameters.h"
#include "../../mathUtils/src/sparseMatrix.h"
#include "turbulenceModels.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/StdVector>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace fileManagement;
using namespace mesh;
using namespace mathUtils::sparse;
using namespace Eigen;
using std::vector;
using std::min;
using std::max;
using std::cout;

namespace flow
{
    // dummy struct to create a scoped enum with conversion to int
    struct BoundaryType
    {
        enum types {VELOCITY = 1, TOTALPRESSURE = 2,
                    PRESSURE = 3, MASSFLOW = 4,
                    WALL     = 5, SYMMETRY = 6,
                    PERIODIC = 7, GHOST = 8, INTERNAL = 9};
    };

    struct AdjointInputData;

    class PressureBasedCoupledSolver
    {
        friend void _flowTest(const std::string caseNum, const bool extraCalculations);
        friend void _adjointInputTest();

        typedef vector<Matrix<float,4,4>,aligned_allocator<Matrix<float,4,4> > > vectorOfMatrix4f;
        typedef vector<Matrix<float,1,1>,aligned_allocator<Matrix<float,1,1> > > vectorOfMatrix1f;

      public:
        struct flowField
        {
            VectorXf u, v, w, p, turb1, turb2;
        };
        struct gradients
        {
            MatrixX3f u, v, w, p, turb1, turb2;
        };
        struct communication
        {
            int partI,  partJ;  // Partitions involved in the communication
            int startI, startJ; // Starting ghost cell index of each partition
            int length;         // Number of ghost cells
        };
      private:
        struct boundaries
        {
            int number;
            VectorXi type;
            MatrixXf conditions;
        };
        struct ghostCells
        {
            int number;
            VectorXi nghbrPart;
            VectorXi nghbrCell;
            flowField flwFld;
            gradients nabla;
            flowField limit;
            // Vectors to store the coefficients that fall outside of the partition's matrix
            // MatrixXf used for generality coefMat_t is a vector of 1x1 matrices
            vectorOfMatrix4f coefMat;
            vectorOfMatrix1f coefMat_t;
            // Vectors to store data from the neighbour cell
            // needed for pressure equation coefficients
            VectorXf vol_nb, diag_nb, mut;
        };
        struct residuals
        {
            ArrayXf values, norms;
            void init() {values.setZero(6); norms.setOnes(6);}
            inline void updateNorms(const float threshold = 1e-7f) {
                for(int i=0; i<6; ++i)
                    if((norms(i)==1.0f && values(i)>threshold) || values(i)>norms(i)*5.0f) norms(i) = values(i);
            }
            inline bool isConverged(const float tol)
                {return (values/norms < tol).all();}
            inline bool hasDiverged(const float threshold = 1e5f)
                {return (!values.isFinite().all() || (values/norms > threshold).any());}
            void print(const int iteration);
        };
        enum UpdateType {FLOW_FIELD, GRADS_LIMS, DIAGONAL};
      public:
        PressureBasedCoupledSolver();
        void setControl(const FlowParamManager& flowParams);
        int setMesh(vector<UnstructuredMesh>* partitions);
        int applyBndConds(const FlowParamManager& flowParams);
        int initialize(const std::string &filePath);
        int run();
        int getSolution(const UnstructuredMesh& coalescedMesh,
                        MatrixXf* solution_p) const;
        int saveState(const std::string &filePath) const;
        int getAdjointInput(const UnstructuredMesh& coalescedMesh,
                            struct AdjointInputData* data);
      private:
        bool m_ctrlIsSet;
        bool m_meshIsSet;
        bool m_bndrsSet;
        bool m_isInit;
        bool m_finished;

        int m_allocate(); // variables that use significant amounts of memory are allocated here
        int m_deallocate(); // free memory at the end of the solution
        int m_mapPeriodics();
        int m_mapGhostCells();
        void m_computeWallDistance();
        void m_boundaryValues(const int prt, const turbulenceModels::ModelBase* turbModel);
        // Vectors passed as pointers to avoid creating temporary objects
        void m_computeLimiters(const int prt, const VectorXf& phi, const VectorXf& phiGC,
                               const VectorXf& phiF, const MatrixX3f& nablaPhi, VectorXf& limit) const;
        int m_matrixIndexes();
        int m_computeMassFlow(const int prt);
        // Coeffnum is passed to assembleMatrixP from assembleMatrix because the matrix is the same
        // assembleMatrixP applies under-relaxation to the uvwp system, factor computed in this->run()
        int m_assembleMatrix( const int prt, const int Nc, const int Nf, const int Nb, const float cfl);
        int m_assembleMatrixP(const int prt, const int Nf, const int Nb);
        template<class ModelType>
        int m_assembleMatrixT(const int prt, const int Nf, const int Nb, const ModelType* turbModel,
                              const turbulenceModels::ModelBase::ModelVariables turbVar);
        // These operation are in a separate function that can be used for periodic boundaries
        void m_momentumCoefficients(const int prt, const int i);
        void m_pressureCoefficients(const int prt, const int i);

        int m_updateGhostCells(const UpdateType type, const int part);

        float m_areaOfGroup(const int part, const int group) const;
        void  m_bndrsOfType(const int part, const int type, vector<int>& faces,
                            const bool append = true) const;
        int m_defaultInit();

        int m_partNum;
        fileManagement::FlowParamManager::controlParameters m_control;
        residuals m_residuals;

        float m_rotationalSpeed;
        float m_rho;
        float m_mu;

        // One index of the vector for each partition

        vector<communication> m_communications;

        vector<UnstructuredMesh>* m_part;
        vector<flowField>  m_flwFld_C; // at the cell centroid
        vector<flowField>  m_flwFld_F; // at the face centroid
        vector<gradients>  m_nabla;
        vector<flowField>  m_limit; // flux limiters
        vector<VectorXf>   m_mut, m_wallDist, m_mdot;
        vector<boundaries> m_boundaries;
        vector<ghostCells> m_ghostCells;

        vector<SBCSRmatrix<float> > m_coefMat; // Matrix of coefficients of the discretized equation
        vector<VectorXf> m_solution, m_source; // Solution and RHS of the uvwp linear system

        // Same variables for turbulence
        vector<SparseMatrix<float,RowMajor> > m_coefMat_t;
        vector<VectorXf> m_solution_t, m_source_t;

        vector<VectorXi> m_matDiagMap; // Lookup array for diagonal entries of the matrix
        vector<MatrixX2i> m_matOffMap; // Same for off-diagonal, one entry per face
    };

    struct AdjointInputData
    {
        using solvePrec = float;

        int vertexNumber, faceNumber, cellNumber, boundaryNumber;
        float rotationalSpeed, rho, mu, venkatK;
        // cell centroid values
        VectorXf u, v, w, p, mut, mdiag, residuals, ulim, vlim, wlim;
        // face definition and connectivity
        vector<int> verticesStart;
        vector<int> verticesIndex;
        MatrixX3f verticesCoord;
        vector<std::pair<int,int> > connectivity;
        // boundary definition
        ArrayXi  boundaryType;
        MatrixXf boundaryConditions;

        // transposed solver matrix
        vector<vector<int> > rowPtr;
        vector<vector<int> > colIdx;
        vector<SBCSRmatrix<solvePrec> > Aii;
        vector<vector<Matrix<solvePrec,4,4>,aligned_allocator<Matrix<solvePrec,4,4> > > > Aij;
        vector<VectorXi> Aij_ptr;
        vector<PressureBasedCoupledSolver::communication> communications;
    };
}
#endif // FLOW_H
