//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef SOLVERS_H
#define SOLVERS_H

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <limits>

#include "preconditioners.h"
#include "matrix.h"
#include "sparseMatrix.h"

using namespace Eigen;
using namespace mathUtils::preconditioners;
using namespace mathUtils::matrix;
using namespace mathUtils::sparse;

namespace mathUtils
{
    namespace solvers
    {
        template <typename type> bool tdma(const std::vector<type> &a,
                                           const std::vector<type> &b,
                                           std::vector<type> &c,
                                           std::vector<type> &d,
                                           std::vector<type> &x);

        /*** Single Partition Iterative Solvers ***/

        template <class Matrix_t, class RHS_t>
        class SolverBase
        {
          protected:
            using Scalar_t = typename RHS_t::Scalar;

            // Input data
            const Matrix_t* m_A_p = nullptr;
            const RHS_t* m_b_p = nullptr;
            RHS_t* m_x_p = nullptr;
            const PreconditionerBase<Matrix_t,RHS_t>* m_M = nullptr;

            // Data used during solution
            RHS_t m_y;
            // Number of variables per index and right hand sides
            int m_numVarsPerIdx, m_numRHS;

            // Solution information
            int m_numIters, m_exitCode;
            Scalar_t m_res0, m_normRes0, m_res1, m_normRes1;
            Matrix<Scalar_t,Dynamic,1> m_varRes0; // vector with initial residual for each variable

            // Provide a place for derived classes to implement initialization operations when data is set
            virtual bool m_init() {return true;}

            #define USING_BASE_MEMBERS(BASE)  \
              using BASE::m_A_p;              \
              using BASE::m_b_p;              \
              using BASE::m_x_p;              \
              using BASE::m_y;                \
              using BASE::m_M;                \
              using BASE::m_numRHS;           \
              using BASE::m_numVarsPerIdx;    \
              using BASE::m_numIters;         \
              using BASE::m_normRes0;         \
              using BASE::m_normRes1;         \
              using BASE::m_res0;             \
              using BASE::m_res1;             \
              using BASE::m_varRes0;          \
              using BASE::m_exitCode

          public:
            void setData(const Matrix_t& A,
                         const RHS_t& b,
                         RHS_t& x,
                         const int defaultNumVarsPerIdx=1)
            {
                m_A_p = &A; m_b_p = &b; m_x_p = &x;
                m_y.resize(b.rows(), b.cols());

                m_numVarsPerIdx = defaultNumVarsPerIdx;
                m_numRHS = b.cols();

                assert(m_numRHS > 0 && "the rhs is empty");
                assert(m_init() && "failed to initialize solver");
            }

            void setPreconditioner(const PreconditionerBase<Matrix_t,RHS_t>* M) {m_M = M;}

            void getWorkVector(RHS_t& vec) {vec = m_y;}

            virtual int solve(const int maxIters, const Scalar_t tol, const bool verbose) = 0;

            inline int exitCode() const {return m_exitCode;}
            inline int numIters() const {return m_numIters;}
            inline Scalar_t initialResidual(bool normalized=true) const {return normalized? m_normRes0:m_res0;}
            inline Scalar_t finalResidual(  bool normalized=true) const {return normalized? m_normRes1:m_res1;}
            inline Matrix<Scalar_t,Dynamic,1> initialVariableResiduals() const {return m_varRes0;}

            virtual ~SolverBase() {}
        };


        template <class Matrix_t, class RHS_t>
        class BiCGSTAB final : public SolverBase<Matrix_t,RHS_t>
        {
          private:
            using Base = SolverBase<Matrix_t,RHS_t>;
            using Scalar_t = typename Base::Scalar_t;
            USING_BASE_MEMBERS(Base);

            Array<Scalar_t,Dynamic,1> varRes0;

            // The intermideate vectors used by the method
            RHS_t r, r_star, v, p, t;

          public:
            int solve(const int maxIters = 100,
                      const Scalar_t tol = 1e-6,
                      const bool verbose = false) override
            {
                int numEl = m_x_p->rows(), iter, imaxRes;
                Scalar_t eps;

                Array<Scalar_t,1,RHS_t::ColsAtCompileTime> rho_old(m_numRHS), alpha(m_numRHS),
                                                           resNorm(m_numRHS), omega(m_numRHS),
                                                           beta(m_numRHS),    rho(m_numRHS),
                                                           res(m_numRHS);
                rho.setOnes(); alpha.setOnes(); omega.setOnes();

                // Set default preconditioner if none was provided
                preconditioners::Identity<Matrix_t,RHS_t> defaultPrecond;
                if(m_M == nullptr) m_M = &defaultPrecond;

                // Allocate/initialize working vectors
                r.resize(numEl,m_numRHS);  v.setZero(numEl,m_numRHS);
                t.resize(numEl,m_numRHS);  p.setZero(numEl,m_numRHS);
                r_star.resize(numEl,m_numRHS);

                // compute residual
                r = *m_b_p - (*m_A_p)*(*m_x_p);
                r_star = r;

                resNorm = m_b_p->colwise().norm();
                res = r.colwise().norm();

                if((resNorm < Scalar_t(10)*std::numeric_limits<Scalar_t>::min()).all()) return 0;

                for(int j=0; j<m_numRHS; ++j)
                  if(resNorm(j) < Scalar_t(10)*std::numeric_limits<Scalar_t>::min())
                    resNorm(j) = 1.0;

                eps = (res/resNorm).maxCoeff(&imaxRes);
                m_normRes0 = eps;
                m_res0 = res(imaxRes);

                m_varRes0.setZero(m_numVarsPerIdx);
                for(int i=0; i<numEl/m_numVarsPerIdx; ++i)
                  for(int j=0; j<m_numVarsPerIdx; ++j)
                    m_varRes0(j) += r.row(m_numVarsPerIdx*i+j).squaredNorm();
                m_varRes0 = m_varRes0.array().sqrt();

                for(iter=0; iter<maxIters && eps>tol; ++iter)
                {
                    rho_old = rho;
                    rho = r_star.cwiseProduct(r).colwise().sum();
                    beta = (rho/rho_old)*(alpha/omega);

                    for(int j=0; j<m_numRHS; ++j)
                        p.col(j) = r.col(j)+beta(j)*(p.col(j)-omega(j)*v.col(j));

                    m_M->solve(p,m_y);
                    v = (*m_A_p) * m_y;

                    alpha = r_star.cwiseProduct(v).colwise().sum();
                    for(int j=0; j<m_numRHS; ++j)
                      if(std::abs(alpha(j)) < Scalar_t(10)*std::numeric_limits<Scalar_t>::min())
                        alpha(j) = 1.0;
                    alpha = rho/alpha;

                    for(int j=0; j<m_numRHS; ++j) {
                        m_x_p->col(j) += alpha(j)*m_y.col(j);
                        r.col(j) -= alpha(j)*v.col(j);
                    }

                    m_M->solve(r,m_y);
                    t = (*m_A_p) * m_y;

                    omega = t.colwise().squaredNorm();
                    for(int j=0; j<m_numRHS; ++j)
                      if(std::abs(omega(j)) < Scalar_t(10)*std::numeric_limits<Scalar_t>::min())
                        omega(j) = 1.0;

                    omega = t.cwiseProduct(r).colwise().sum().array() / omega;

                    for(int j=0; j<m_numRHS; ++j) {
                        m_x_p->col(j) += omega(j)*m_y.col(j);
                        r.col(j) -= omega(j)*t.col(j);
                    }

                    // Check convergence
                    res = r.colwise().norm();
                    eps = (res/resNorm).maxCoeff(&imaxRes);
                }
                m_y = r_star; // keep initial residual in working vector

                m_numIters = iter;
                m_normRes1 = eps;
                m_res1 = res(imaxRes);
                m_exitCode = iter<maxIters? 0:1;
                if(verbose){
                    std::cout << "BiCGSTAB: Iteration " << iter << "  Residual " << eps;
                    if(m_exitCode!=0) std::cout << "  (Did not converge!)";
                    std::cout << std::endl;
                }

                return m_exitCode;
            }
        };


        template <class Matrix_t, class RHS_t>
        class ConjugateGradient final : public SolverBase<Matrix_t,RHS_t>
        {
          private:
            using Base = SolverBase<Matrix_t,RHS_t>;
            using Scalar_t = typename Base::Scalar_t;
            USING_BASE_MEMBERS(Base);

            Array<Scalar_t,Dynamic,1> varRes0;

            // The intermideate vectors used by the method
            RHS_t v, p;

          public:
            int solve(const int maxIters = 100,
                      const Scalar_t tol = 1e-6,
                      const bool verbose = false) override
            {
                int numEl = m_x_p->rows(), iter, imaxRes;
                Scalar_t eps;

                Array<Scalar_t,1,RHS_t::ColsAtCompileTime> alpha(m_numRHS), beta(m_numRHS),
                                                           gamma(m_numRHS), res(m_numRHS),
                                                           resNorm(m_numRHS);

                // Set default preconditioner if none was provided
                preconditioners::Identity<Matrix_t,RHS_t> defaultPrecond;
                if(m_M == nullptr) m_M = &defaultPrecond;

                // Allocate/initialize working vectors
                v.resize(numEl,m_numRHS); p.resize(numEl,m_numRHS);

                // compute residual
                m_y = *m_b_p - (*m_A_p)*(*m_x_p);

                resNorm = m_b_p->colwise().norm();
                res = m_y.colwise().norm();

                if((resNorm < Scalar_t(10)*std::numeric_limits<Scalar_t>::min()).all()) return 0;

                for(int j=0; j<m_numRHS; ++j)
                  if(resNorm(j) < Scalar_t(10)*std::numeric_limits<Scalar_t>::min())
                    resNorm(j) = 1.0;

                eps = (res/resNorm).maxCoeff(&imaxRes);
                m_normRes0 = eps;
                m_res0 = res(imaxRes);

                m_varRes0.setZero(m_numVarsPerIdx);
                for(int i=0; i<numEl/m_numVarsPerIdx; ++i)
                  for(int j=0; j<m_numVarsPerIdx; ++j)
                    m_varRes0(j) += m_y.row(m_numVarsPerIdx*i+j).squaredNorm();
                m_varRes0 = m_varRes0.array().sqrt();

                m_M->solve(m_y, v); p = v;

                gamma = m_y.cwiseProduct(v).colwise().sum();

                for(iter=0; iter<maxIters; ++iter)
                {
                    v = (*m_A_p) * p;

                    alpha = gamma / p.cwiseProduct(v).colwise().sum().array();

                    for(int j=0; j<m_numRHS; ++j) {
                        m_x_p->col(j) += alpha(j)*p.col(j);
                        m_y.col(j) -= alpha(j)*v.col(j);
                    }

                    // Check convergence
                    res = m_y.colwise().norm();
                    eps = (res/resNorm).maxCoeff(&imaxRes);
                    if(eps < tol) break;

                    m_M->solve(m_y, v);

                    beta = gamma;
                    gamma = m_y.cwiseProduct(v).colwise().sum();
                    beta = gamma/beta;

                    for(int j=0; j<m_numRHS; ++j)
                        p.col(j) = v.col(j)+beta(j)*p.col(j);
                }

                m_numIters = iter;
                m_normRes1 = eps;
                m_res1 = res(imaxRes);
                m_exitCode = iter<maxIters? 0:1;
                if(verbose){
                    std::cout << "CG: Iteration " << iter << "  Residual " << eps;
                    if(m_exitCode!=0) std::cout << "  (Did not converge!)";
                    std::cout << std::endl;
                }

                return m_exitCode;
            }
        };
        #undef USING_BASE_MEMBERS


        /*** Multiple Partition Iterative Solvers ***/

        // Helper functions to specialize matrix-vector product calls, avoids
        // overloading * for our types (which would require a lot of Eigen templating)
        template<class A_t, class x_t, class y_t>
        inline void matVecProd(const A_t& A, const x_t& x, y_t& y) { y = A*x; }

        template<typename T, class x_t, class y_t>
        inline void matVecProd(const SBCSRmatrix<T>& A, const x_t& x, y_t& y) { A.vectorProduct(x,y); }


        // Abstract base class implementing the basic methods to transfer data between
        // partitions and multiply the distributed matrix by a distributed vector
        template <class Matrix_t, class RHS_t, class Comm_t, class VectorOfMatrix_t>
        class DistributedSolverBase
        {
          protected:
            using Scalar_t = typename RHS_t::Scalar;

            // The outer vector is for the different partitions

            // Input data
            // Diagonal entries of the block matrix and distributed vectors (RHS and LHS)
            const std::vector<Matrix_t> * m_Aii_p;
            const std::vector<RHS_t> * m_bi_p;
            std::vector<RHS_t> * m_xi_p;
            // Off-diagonal entries
            // One square matrix of size numVarsPerIdx per connection with other partition
            // and local index the matrix is associated to
            std::vector<const VectorXi *> m_nghbrIdx_p;
            std::vector<const VectorOfMatrix_t *> m_Aij_p;
            // Vector of a structure with the mapping between off-diagonal entries
            const std::vector<Comm_t> * m_communications;
            // Preconditioner
            std::vector<const PreconditionerBase<Matrix_t,RHS_t> *> m_precond_p;

            // Data used during solution
            // Distributed vectors the matrix is multiplied by
            std::vector<RHS_t> m_yi, m_yij;
            // Partial and reduced results of dot product
            Array<Scalar_t,Dynamic,RHS_t::ColsAtCompileTime> m_partialDot;
            Array<Scalar_t,1,RHS_t::ColsAtCompileTime> m_reducedDot;
            // Number of partitions and variables per index
            int m_numParts, m_numVarsPerIdx, m_numRHS;

            // Solution information
            int m_numIters, m_exitCode;
            Scalar_t m_res0, m_normRes0, m_res1, m_normRes1;
            Matrix<Scalar_t,Dynamic,1> m_varRes0; // vector with initial residual for each variable

            // To be called for all partition, can be executed by all threads simultaneously
            inline void m_updateOffDiagEntries(const int partI)
            {
                #pragma omp barrier
                int partJ, startI, startJ, idxI, idxJ;
                for(auto & comm : (*m_communications))
                {
                    partJ = -1;
                    if(partI==comm.partI) {partJ=comm.partJ; startI=comm.startI; startJ=comm.startJ;}
                    if(partI==comm.partJ) {partJ=comm.partI; startI=comm.startJ; startJ=comm.startI;}
                    if(partJ>-1)
                      for(int i=0; i<comm.length; ++i) {
                        idxI = m_numVarsPerIdx*(startI++);
                        idxJ = m_numVarsPerIdx*(*m_nghbrIdx_p[partJ])(startJ++);
                        m_yij[partI].block(idxI,0,m_numVarsPerIdx,m_numRHS)
                        =m_yi[partJ].block(idxJ,0,m_numVarsPerIdx,m_numRHS);
                      }
                }
            }
            inline void m_multiplyMatrix(const int part, RHS_t& lhs)
            {
                m_updateOffDiagEntries(part);

                // Diagonal block
                matVecProd((*m_Aii_p)[part], m_yi[part], lhs);

                // Off-diagonal
                if(m_numParts>1)
                for(int i=0; i<(*m_nghbrIdx_p[part]).rows(); ++i) {
                    int C0 = (*m_nghbrIdx_p[part])(i);
                    lhs.block(m_numVarsPerIdx*C0,0,m_numVarsPerIdx,m_numRHS) += (*m_Aij_p[part])[i]*
                        m_yij[part].block(m_numVarsPerIdx*i,0,m_numVarsPerIdx,m_numRHS);
                }
            }

            template<class T>
            inline void m_dotProd(const int part, const RHS_t& v1, const RHS_t& v2, T& result)
            {
                const int N = v1.rows();

                for(int i=0; i<m_numRHS; ++i)
                  m_partialDot(part,i) = stableDotProd(N, v1.col(i).data(), v2.col(i).data());

                #pragma omp barrier
                #pragma omp master
                m_reducedDot = m_partialDot.colwise().sum();
                #pragma omp barrier
                result = m_reducedDot;
            }

            // Provide a place for derived classes to implement initialization operations when data is set
            virtual bool m_init() {return true;}

            #define USING_BASE_MEMBERS(BASE)      \
              using BASE::m_Aii_p;                \
              using BASE::m_bi_p;                 \
              using BASE::m_xi_p;                 \
              using BASE::m_nghbrIdx_p;           \
              using BASE::m_Aij_p;                \
              using BASE::m_communications;       \
              using BASE::m_precond_p;            \
              using BASE::m_yij;                  \
              using BASE::m_yi;                   \
              using BASE::m_numParts;             \
              using BASE::m_numVarsPerIdx;        \
              using BASE::m_numIters;             \
              using BASE::m_numRHS;               \
              using BASE::m_exitCode;             \
              using BASE::m_res0;                 \
              using BASE::m_normRes0;             \
              using BASE::m_res1;                 \
              using BASE::m_normRes1;             \
              using BASE::m_varRes0;              \
              using BASE::m_multiplyMatrix;       \
              using BASE::m_dotProd

          public:
            DistributedSolverBase() {
                m_precond_p = std::vector<const PreconditionerBase<Matrix_t,RHS_t>*>(1);
            }
            void setData(const std::vector<Matrix_t > * Aii_p,
                         const std::vector<RHS_t> * bi_p,
                         std::vector<RHS_t> * xi_p,
                         const std::vector<const VectorXi*>& nghbrIdx_p,
                         const std::vector<const VectorOfMatrix_t*>& Aij_p,
                         const std::vector<Comm_t> * communications,
                         const int defaultNumVarsPerIdx=1)
            {
                m_Aii_p = Aii_p;   m_bi_p = bi_p;   m_xi_p = xi_p;
                m_nghbrIdx_p = nghbrIdx_p;   m_Aij_p = Aij_p;
                m_communications = communications;

                m_numParts = m_Aii_p->size();
                m_numRHS = (*m_bi_p)[0].cols();
                assert(m_numRHS > 0 && "the rhs is empty");

                m_yi.resize(m_numParts);
                m_yij.resize(m_numParts);
                m_partialDot.resize(m_numParts,m_numRHS);
                m_reducedDot.resize(m_numRHS);

                // with one partition there are no submatrices to get value from
                if(m_numParts==1)
                    m_numVarsPerIdx = defaultNumVarsPerIdx;
                else
                    m_numVarsPerIdx = (*m_Aij_p[0])[0].rows();

                // initialization routine of derived solvers
                assert(m_init() && "failed to initialize solver");
            }

            void setPreconditioner(
                const std::vector<const PreconditionerBase<Matrix_t,RHS_t>*> precond_p)
            {
                m_precond_p = precond_p;
            }

            void getWorkVector(const int prt, RHS_t& vec) {vec = m_yi[prt];}

            virtual int solve(const int maxIters, const Scalar_t tol, const bool verbose) = 0;

            inline int exitCode() const {return m_exitCode;}
            inline int numIters() const {return m_numIters;}
            inline Scalar_t initialResidual(bool normalized=true) const {return normalized? m_normRes0:m_res0;}
            inline Scalar_t finalResidual(  bool normalized=true) const {return normalized? m_normRes1:m_res1;}
            inline Matrix<Scalar_t,Dynamic,1> initialVariableResiduals() const {return m_varRes0;}

            virtual ~DistributedSolverBase() {}
        };

        template <class Matrix_t, class RHS_t, class Comm_t,
                  class VectorOfMatrix_t = std::vector<Matrix<typename RHS_t::Scalar,Dynamic,Dynamic> > >
        class DistBiCGSTAB final : public DistributedSolverBase<Matrix_t,RHS_t,Comm_t,VectorOfMatrix_t>
        {
          private:
            using Base = DistributedSolverBase<Matrix_t,RHS_t,Comm_t,VectorOfMatrix_t>;
            using Scalar_t = typename Base::Scalar_t;
            USING_BASE_MEMBERS(Base);

            Array<Scalar_t,Dynamic,Dynamic> varRes0;

            // The intermideate vectors used by the method
            std::vector<RHS_t> r, r_star, v, p, t;

            // Resizing of containers for the number of threads
            bool m_init() override
            {
                varRes0.resize(m_numVarsPerIdx,m_numParts);

                r.resize(m_numParts);  v.resize(m_numParts);
                p.resize(m_numParts);  t.resize(m_numParts);
                r_star.resize(m_numParts);

                if(m_precond_p.size() != size_t(m_numParts)) m_precond_p.resize(m_numParts);
                return true;
            }
          public:
            // in parallel needs to be called by every thread, e.g.
            // #pragma omp parallel num_threads(numParts)
            int solve(const int maxIters = 100,
                      const Scalar_t tol = 1e-6,
                      const bool verbose = false) override
            {
                const int part = omp_get_thread_num();

                // references to simplify code downstream (i is for part)
                auto& xi = (*m_xi_p)[part];
                auto& bi = (*m_bi_p)[part];
                auto& yi = m_yi[part];
                auto& ri = r[part];
                auto& vi = v[part];
                auto& ti = t[part];
                auto& pi = p[part];
                auto& r0i = r_star[part];
                auto& Mi = m_precond_p[part];

                const int numEl = xi.rows();
                int imaxRes;
                Scalar_t eps;

                using Array_t = Array<Scalar_t,1,RHS_t::ColsAtCompileTime>;
                Array_t rho_old(m_numRHS), alpha = Array_t::Ones(m_numRHS),
                        resNorm(m_numRHS), omega = Array_t::Ones(m_numRHS),
                        beta(m_numRHS),    rho   = Array_t::Ones(m_numRHS),
                        aux(m_numRHS);

                // Set default preconditioner if none was provided
                preconditioners::Identity<Matrix_t,RHS_t> defaultPrecond;
                if(Mi == nullptr) Mi = &defaultPrecond;

                // Allocate/initialize working vectors
                if(m_numParts>1) m_yij[part].resize(m_nghbrIdx_p[part]->rows()*m_numVarsPerIdx,m_numRHS);
                ri.resize(numEl,m_numRHS);  vi.setZero(numEl,m_numRHS);
                ti.resize(numEl,m_numRHS);  pi.setZero(numEl,m_numRHS);
                r0i.resize(numEl,m_numRHS);
                varRes0.col(part).setZero();

                // compute residual, r is used as temp for Ax, x is copied into y
                yi = -xi;
                m_multiplyMatrix(part,ri);
                ri += bi;
                r0i = ri;

                // compute initial residuals and norms
                for(int i=0; i<numEl/m_numVarsPerIdx; ++i)
                  for(int j=0; j<m_numVarsPerIdx; ++j)
                    varRes0(j,part) += ri.row(m_numVarsPerIdx*i+j).squaredNorm();

                m_dotProd(part, ri, ri, aux);
                m_dotProd(part, bi, bi, resNorm);
                resNorm = resNorm.sqrt();
                eps = (aux.sqrt()/resNorm).maxCoeff(&imaxRes);

                #pragma omp master
                {
                    m_normRes0 = eps;
                    m_res0 = std::sqrt(aux(imaxRes));
                    m_varRes0 = varRes0.rowwise().sum().sqrt();
                }

                int iter = 0;
                while(iter++ < maxIters && eps > tol)
                {
                    rho_old = rho;
                    m_dotProd(part, r0i, ri, rho);

                    beta = (rho/rho_old)*(alpha/omega);
                    for(int j=0; j<m_numRHS; ++j)
                      pi.col(j) = ri.col(j)+beta(j)*(pi.col(j)-omega(j)*vi.col(j));

                    Mi->solve(pi, yi);
                    m_multiplyMatrix(part, vi);

                    m_dotProd(part, r0i, vi, alpha);
                    alpha = rho/alpha;
                    for(int j=0; j<m_numRHS; ++j) {
                      xi.col(j) += alpha(j)*yi.col(j);
                      ri.col(j) -= alpha(j)*vi.col(j);
                    }

                    Mi->solve(ri, yi);
                    m_multiplyMatrix(part, ti);

                    m_dotProd(part, ti, ri, omega);
                    m_dotProd(part, ti, ti, aux);
                    omega /= aux;
                    for(int j=0; j<m_numRHS; ++j) {
                      xi.col(j) += omega(j)*yi.col(j);
                      ri.col(j) -= omega(j)*ti.col(j);
                    }

                    // Check convergence
                    m_dotProd(part, ri, ri, aux);
                    eps = (aux.sqrt()/resNorm).maxCoeff(&imaxRes);
                }
                yi = r0i; // keep initial residual in working vector
                #pragma omp master
                {
                    m_numIters = iter;
                    m_normRes1 = eps;
                    m_res1 = std::sqrt(aux(imaxRes));
                    m_exitCode = iter<maxIters? 0:1;
                    if(verbose){
                        std::cout << "BiCGSTAB: Iteration " << iter << "  Residual " << eps;
                        if(m_exitCode!=0) std::cout << "  (Did not converge!)";
                        std::cout << std::endl;
                    }
                }
                #pragma omp barrier
                return m_exitCode;
            }
        };
        #undef USING_BASE_MEMBERS
    }
}

#endif // SOLVERS_H
