//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <cassert>
#include <stdlib.h>

namespace mathUtils
{
    namespace sparse
    {
        // A block-compressed-storage-row sparse matrix where the blocks
        // have a particular size (4x4) and sparse structure that results
        // from the discretization used in the coupled solver:
        // X 0 0 X - u
        // 0 X 0 X - v
        // 0 0 X X - w
        // X X X X - p
        // Operations are optimized for this block format.
        template<typename T>
        class SBCSRmatrix
        {
          private:
            enum : int {BlockSize = 10};

            const int* m_rowPtr = nullptr;
            const int* m_colIdx = nullptr;
            T* m_values = nullptr;

            int m_rows = 0;
            int m_size = 0;

            template<typename U>
            void deepCopy(const SBCSRmatrix<U>& other)
            {
                m_rows = other.m_rows;
                m_size = other.m_size;
                m_rowPtr = other.m_rowPtr;
                m_colIdx = other.m_colIdx;

                allocate();

                for(int i=0; i<m_size; ++i)
                  m_values[i] = other.m_values[i];
            }

          public:
            template<typename U> friend class SBCSRmatrix;

            // This facilitates accesses to each section of a block.
            struct BlockView
            {
                T* m_blk;

                BlockView(T* blk) : m_blk(blk) {}

                inline T& diag(int i) {return m_blk[i];}
                inline T& rCol(int i) {return m_blk[4+i];}
                inline T& bRow(int i) {return m_blk[7+i];}
            };

            struct ConstBlockView
            {
                const T* m_blk;

                ConstBlockView(const T* blk) : m_blk(blk) {}

                inline T diag(int i) const {return m_blk[i];}
                inline T rCol(int i) const {return m_blk[4+i];}
                inline T bRow(int i) const {return m_blk[7+i];}
            };

            ~SBCSRmatrix() {free(m_values);}

            SBCSRmatrix& operator= (const SBCSRmatrix& rhs)
            {
                deepCopy(rhs);
                return *this;
            }

            template<typename U>
            SBCSRmatrix<T>& operator= (const SBCSRmatrix<U>& rhs)
            {
                deepCopy(rhs);
                return *this;
            }

            void setStructure(int rows, int nnz,
                              const int* rowPtr, const int* colIdx)
            {
                m_rows = rows;
                m_size = nnz*BlockSize;
                m_rowPtr = rowPtr;
                m_colIdx = colIdx;
            }

            void allocate()
            {
                free(m_values);
                auto sz = ((m_size*sizeof(T)+63)/64)*64;
                m_values = static_cast<T*>(aligned_alloc(64,sz));
            }

            inline void setZero()
            {
                #pragma omp simd
                for(int i=0; i<m_size; ++i) m_values[i] = 0;
            }

            void transposeInPlace()
            {
                for(int i=0; i<m_rows; ++i)
                {
                    int j_p = m_rowPtr[i];
                    while(m_colIdx[j_p] != i)
                    {
                        int j = m_colIdx[j_p];
                        auto bij = getBlock(j_p);
                        auto bji = getBlock(j,i);

                        // swap diagonal of blocks
                        for(int k=0; k<4; ++k)
                        {
                            T tmp = bij.diag(k);
                            bij.diag(k) = bji.diag(k);
                            bji.diag(k) = tmp;
                        }

                        // transpose-swap off-diags of blocks
                        for(int k=0; k<3; ++k)
                        {
                            T tmp = bij.rCol(k);
                            bij.rCol(k) = bji.bRow(k);
                            bji.bRow(k) = tmp;

                            tmp = bij.bRow(k);
                            bij.bRow(k) = bji.rCol(k);
                            bji.rCol(k) = tmp;
                        }
                        ++j_p;
                    }

                    // transpose the diagonal block
                    auto bii = getBlock(j_p);

                    for(int k=0; k<3; ++k)
                    {
                        T tmp = bii.rCol(k);
                        bii.rCol(k) = bii.bRow(k);
                        bii.bRow(k) = tmp;
                    }
                }
            }

            inline int rows() const {return m_rows;}
            inline int cols() const {return m_rows;}
            inline int nnz() const {return m_size/BlockSize;}

            inline const int* rowPtr() const {return m_rowPtr;}
            inline const int* colIdx() const {return m_colIdx;}

            inline int getBlockIndex(int row, int col) const
            {
                for(int k=m_rowPtr[row]; k<m_rowPtr[row+1]; ++k)
                  if(m_colIdx[k]==col)
                    return k;
                return -1;
            }

            inline BlockView getBlock(int i)
            {
                return BlockView(&m_values[i*BlockSize]);
            }

            inline BlockView getBlock(int row, int col)
            {
                return getBlock(getBlockIndex(row,col));
            }

            inline ConstBlockView getBlock(int i) const
            {
                return ConstBlockView(&m_values[i*BlockSize]);
            }

            inline ConstBlockView getBlock(int row, int col) const
            {
                return getBlock(getBlockIndex(row,col));
            }

            template<class x_t, class y_t>
            void vectorProduct(const x_t& x, y_t& y) const
            {
                assert(x.rows() == 4*m_rows);
                assert(x.cols() == y.cols());
                assert(x.rows() == y.rows());

                for(int i=0; i<m_rows; ++i)
                {
                    for(int l=0; l<4; ++l)
                      y.row(4*i+l).setZero();

                    for(int k=m_rowPtr[i]; k<m_rowPtr[i+1]; ++k)
                    {
                        const int j = m_colIdx[k];
                        auto bij = ConstBlockView(&m_values[k*BlockSize]);

                        y.row(4*i+0) += bij.diag(0)*x.row(4*j+0) + bij.rCol(0)*x.row(4*j+3);
                        y.row(4*i+1) += bij.diag(1)*x.row(4*j+1) + bij.rCol(1)*x.row(4*j+3);
                        y.row(4*i+2) += bij.diag(2)*x.row(4*j+2) + bij.rCol(2)*x.row(4*j+3);

                        y.row(4*i+3) += bij.diag(3)*x.row(4*j+3) + bij.bRow(0)*x.row(4*j+0)+
                                        bij.bRow(1)*x.row(4*j+1) + bij.bRow(2)*x.row(4*j+2);
                    }
                }
            }
        };
    }
}

#endif // SPARSEMATRIX_H
