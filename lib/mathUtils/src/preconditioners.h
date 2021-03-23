//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef PRECONDITIONERS_H
#define PRECONDITIONERS_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>
#include <cassert>
#include <stdlib.h>

#include "matrix.h"
#include "sparseMatrix.h"

namespace mathUtils
{
// forward declaration of testing function (friend)
void _preconditionerTests();

namespace preconditioners
{
using namespace Eigen;
using namespace sparse;

// Interface that all preconditioners need to implement
template<class Matrix_t, class RHS_t>
class PreconditionerBase
{
  public:
    virtual int compute(const Matrix_t& mat) = 0;
    virtual int solve(const RHS_t& rhs, RHS_t& lhs) const = 0;
    virtual ~PreconditionerBase() {}
};

// Identity preconditioner approximates A by I
template<class Matrix_t, class RHS_t>
class Identity final: public PreconditionerBase<Matrix_t,RHS_t>
{
  public:
    int compute(const Matrix_t&) {return 0;}
    int solve(const RHS_t& rhs, RHS_t& lhs) const {lhs = rhs; return 0;}
};

// Diagonal preconditioner approximates A by diag(A)
template<class Matrix_t, class RHS_t>
class Diagonal final: public PreconditionerBase<Matrix_t,RHS_t>
{
  private:
    using Scalar_t = typename RHS_t::Scalar;

    Matrix<Scalar_t,Dynamic,1> m_diagonal;
    bool m_isReady = false;

  public:
    int compute(const Matrix_t& mat)
    {
        assert(mat.rows() == mat.cols());

        m_diagonal.resize(mat.rows());
        for(int i=0; i<mat.rows(); ++i)
            m_diagonal(i) = Scalar_t(1)/mat.coeff(i,i); // inversion done here only once!

        m_isReady = true;
        return 0;
    }

    int solve(const RHS_t& rhs, RHS_t& lhs) const
    {
        assert(m_isReady);
        lhs = m_diagonal.asDiagonal()*rhs;
        return 0;
    }
};

// Zero fill-in ILU shares the non-zero pattern of A, algorithm for compressed row storage
template<class RHS_t>
class ILU0 final: public PreconditionerBase<SparseMatrix<typename RHS_t::Scalar,RowMajor>,RHS_t>
{
    friend void mathUtils::_preconditionerTests();
  private:
    using Scalar_t = typename RHS_t::Scalar;

    int m_rows, m_nnz;
    // pointers to indexes of A
    const int *m_rowPtr, *m_colIdx;
    Matrix<Scalar_t,Dynamic,1> m_values;
    bool m_isReady = false;

  public:
    int compute(const SparseMatrix<Scalar_t,RowMajor>& mat)
    {
        assert(mat.isCompressed());
        assert(mat.rows() == mat.cols());

        m_rows = mat.rows();
        m_nnz  = mat.nonZeros();
        m_rowPtr = mat.outerIndexPtr();
        m_colIdx = mat.innerIndexPtr();
        m_values.resize(m_nnz);

        // copy non zeros of mat to m_values, factorization will be done in place
        #pragma omp simd
        for(int i=0; i<m_nnz; ++i) m_values(i) = mat.valuePtr()[i];

        // factorize
        for(int i=1; i<m_rows; ++i)
        {
            for(int k_p = m_rowPtr[i]; m_colIdx[k_p] < i; ++k_p)
            {
                // search for a_kk
                int k = m_colIdx[k_p];
                int l_p = m_rowPtr[k];
                while(m_colIdx[l_p] != k) ++l_p;
                m_values(k_p) /= m_values(l_p);

                for(int j_p = k_p+1; j_p < m_rowPtr[i+1]; ++j_p)
                {
                    // search for a_kj (may not exist)
                    int j = m_colIdx[j_p];
                    for(int l_p = m_rowPtr[k]; l_p < m_rowPtr[k+1]; ++l_p)
                    {
                        if(m_colIdx[l_p] == j)
                        {
                            m_values(j_p) -= m_values(k_p)*m_values(l_p);
                            break;
                        }
                    }
                }
            }
        }
        m_isReady = true;
        return 0;
    }

    int solve(const RHS_t& rhs, RHS_t& lhs) const
    {
        assert(m_isReady);
        assert(rhs.rows() == m_rows);
        assert(rhs.cols() == lhs.cols());
        assert(rhs.rows() == lhs.rows());

        // Forward substitution Ly=b
        lhs = rhs;
        for(int i=1; i<m_rows; ++i)
            for(int j_p = m_rowPtr[i]; m_colIdx[j_p] < i; ++j_p)
                lhs.row(i) -= m_values(j_p)*lhs.row(m_colIdx[j_p]);

        // Backward substitution Ux=y
        for(int i=m_rows-1; i>=0; --i)
        {
            int j_p;
            for(j_p = m_rowPtr[i+1]-1; m_colIdx[j_p] > i; --j_p)
                lhs.row(i) -= m_values(j_p)*lhs.row(m_colIdx[j_p]);
            lhs.row(i) /= m_values(j_p);
        }

        return 0;
    }
};

// Zero fill-in ILU shares the non-zero pattern of A, algorithm for compressed row storage
// Version for block matrix with special sparse block format
template<class RHS_t>
class SBILU0 final: public PreconditionerBase<SBCSRmatrix<typename RHS_t::Scalar>,RHS_t>
{
  private:
    using Scalar_t = typename RHS_t::Scalar;
    using BlockType = Matrix<Scalar_t,4,4,ColMajor>;

    // pointers to indexes of A and array of coefficients
    const int *m_rowPtr = nullptr;
    const int *m_colIdx = nullptr;
    std::vector<BlockType,aligned_allocator<BlockType> > m_invDiag;
    std::vector<BlockType,aligned_allocator<BlockType> > m_values;

    // sizes and flag to indicate if compute has been called
    int m_rows = 0, m_nnz = 0;
    bool m_isReady = false;

  public:
    int compute(const SBCSRmatrix<Scalar_t>& mat)
    {
        m_nnz = mat.nnz();
        m_rows = mat.rows();
        m_rowPtr = mat.rowPtr();
        m_colIdx = mat.colIdx();

        m_invDiag.resize(m_rows);
        m_values.resize(m_nnz);

        // copy non zeros of mat to m_values, factorization will be done in place
        // not a direct copy, the factorization has dense blocks, the matrice's are sparse
        for(int i=0; i<m_nnz; ++i)
        {
            auto blk = mat.getBlock(i);
            m_values[i].setZero();
            for(int k=0; k<4; ++k) m_values[i](k,k) = blk.diag(k);
            for(int k=0; k<3; ++k) m_values[i](k,3) = blk.rCol(k);
            for(int k=0; k<3; ++k) m_values[i](3,k) = blk.bRow(k);
        }

        // factorize
        for(int i=1; i<m_rows; ++i)
        {
            // invert previous diagonal
            int l_p = m_rowPtr[i-1];
            while(m_colIdx[l_p] != i-1) ++l_p;
            m_invDiag[i-1] = m_values[l_p].inverse();

            for(int k_p = m_rowPtr[i]; m_colIdx[k_p] < i; ++k_p)
            {
                int k = m_colIdx[k_p];
                m_values[k_p] *= m_invDiag[k];

                for(int j_p = k_p+1; j_p<m_rowPtr[i+1]; ++j_p)
                {
                    // search for a_kj (may not exist)
                    int j = m_colIdx[j_p];
                    for(int l_p = m_rowPtr[k]; l_p < m_rowPtr[k+1]; ++l_p)
                    {
                        if(m_colIdx[l_p] == j)
                        {
                            m_values[j_p] -= m_values[k_p]*m_values[l_p];
                            break;
                        }
                    }
                }
            }
        }
        // invert last diagonal
        m_invDiag.back() = m_values.back().inverse();

        m_isReady = true;
        return 0;
    }

    int solve(const RHS_t& rhs, RHS_t& lhs) const
    {
        assert(m_isReady);
        assert(rhs.rows() == 4*m_rows);
        assert(rhs.cols() == lhs.cols());
        assert(rhs.rows() == lhs.rows());

        int nrhs = rhs.cols();

        // Forward substitution Ly=b
        lhs = rhs;
        for(int i=1; i<m_rows; ++i)
        {
            for(int j_p = m_rowPtr[i]; m_colIdx[j_p] < i; ++j_p)
            {
                int j = m_colIdx[j_p];
                lhs.block(4*i,0,4,nrhs) -= m_values[j_p]*lhs.block(4*j,0,4,nrhs);
            }
        }

        // Backward substitution Ux=y
        for(int i=m_rows-1; i>=0; --i)
        {
            int j_p;
            for(j_p = m_rowPtr[i+1]-1; m_colIdx[j_p] > i; --j_p)
            {
                int j = m_colIdx[j_p];
                lhs.block(4*i,0,4,nrhs) -= m_values[j_p]*lhs.block(4*j,0,4,nrhs);
            }
            lhs.block(4*i,0,4,nrhs) = m_invDiag[i]*lhs.block(4*i,0,4,nrhs);
        }
        return 0;
    }
};

} // preconditioners
} // mathUtils

#endif // PRECONDITIONERS_H
