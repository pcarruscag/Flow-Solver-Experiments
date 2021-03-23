//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef MATRIX_H
#define MATRIX_H

#include <Eigen/Sparse>
#include <vector>
#include <numeric>

namespace mathUtils
{
    namespace matrix
    {
        using namespace Eigen;

        template <typename type> class Triplet
        {
          public:
            int m_row, m_column;
            type m_value;
            Triplet() : m_row(0), m_column(0), m_value(0) {}
            Triplet(const int row, const int column, const type& value=type(0)) :
                m_row(row), m_column(column), m_value(value) {}
            const int& row() const {return m_row;}
            const int& col() const {return m_column;}
            const type& value() const {return m_value;}
        };

        template <typename type> class NeumaierSummator
        {
          private:
            type m_sum, m_compensation;
          public:
            NeumaierSummator() : m_sum(0), m_compensation(0) {}
            void reset() {m_sum=0; m_compensation=0;}
            inline void add(const type& sample)
            {
                type temp = m_sum+sample;
                if(std::abs(m_sum) > std::abs(sample))
                    m_compensation += (m_sum-temp)+sample;
                else
                    m_compensation += (sample-temp)+m_sum;
                m_sum = temp;
            }
            type result() const {return m_sum+m_compensation;}
        };

        // recursive dot product computation to reduce numerical error
        template <typename type, int safeSize=1024>
        type stableDotProd(int N, const type* v1, const type* v2)
        {
            if(N<safeSize)
            {
                type res = 0;
                for(int i=0; i<N; ++i) res += v1[i]*v2[i];
                return res;
            }
            else
            {
                int N1 = N/2;
                int N2 = N-N1;
                return stableDotProd<type,safeSize>(N1,v1,v2)+
                       stableDotProd<type,safeSize>(N2,v1+N1,v2+N1);
            }
        }

        // default comparator of sortIndexes
        template <typename type>
        bool lessThan(const type& a, const type& b) {return a < b;}

        // default comparator of pruneByNumber
        template <typename type>
        bool greaterThan(const type& a, const type& b) {return a > b;}

        template <typename type>
        std::vector<int> sortIndexes(const std::vector<type > &v,
                                     bool(*comparator)(const type&,const type&) = &lessThan)
        {
          // initialize original index locations
          std::vector<int> idx(v.size());
          std::iota(idx.begin(), idx.end(), 0);

          // sort indexes based on comparing values in v
          std::sort(idx.begin(), idx.end(),
                    [&](size_t i1, size_t i2) {return (*comparator)(v[i1],v[i2]);});
          return idx;
        }

        template <typename Scalar,int Storage>
        void pruneByNumber(const int N,
                           SparseVector<Scalar,Storage> *vec,
                           bool(*comparator)(const Scalar&,const Scalar&) = &greaterThan)
        {
            // make a copy of vec with absolute values
            std::vector<Scalar> vecTemp;
            Scalar* vecVals = vec->valuePtr();
            vecTemp.reserve(vec->nonZeros());
            for(typename SparseVector<Scalar,Storage>::InnerIterator it(*vec); it; ++it)
                vecTemp.push_back(std::abs(it.value()));
            // indexes of sorting vector in desired order
            // descending to keep largest, ascending to keep smallest
            std::vector<int> idxToKeep = sortIndexes(vecTemp,comparator);
            // set undesired values to 0 and call prune method to remove them
            for(int i=N; i<vec->nonZeros(); ++i)
                vecVals[idxToKeep[i]] = Scalar(0);
            vec->prune(Scalar(0));
        }

        template <typename type1, typename type2>
        bool comparePair1(const std::pair<type1,type2>& P1,const std::pair<type1,type2>& P2)
        {
            if(P1.first < P2.first)
                return true;
            else if(P1.first == P2.first)
                return P1.second < P2.second;
            else
                return false;
        }

        template <typename type1, typename type2>
        bool comparePair2(const std::pair<type1,type2>& P1,const std::pair<type1,type2>& P2)
        {
            if(P1.second < P2.second)
                return true;
            else if(P1.second == P2.second)
                return P1.first < P2.first;
            else
                return false;
        }

        template <typename T>
        bool comparePair3(const std::pair<T,T>& P1,const std::pair<T,T>& P2)
        {
            // increasing order of minimum element in the pairs
            using std::min; using std::max;

            T min1 = min(P1.first,P1.second),
              min2 = min(P2.first,P2.second);

            if(min1 < min2)
                return true;
            else if(min1 == min2)
                return max(P1.first,P1.second) < max(P2.first,P2.second);
            else
                return false;
        }

        template <typename type>
        bool existsInVector(const std::vector<type>& vec, const type& val)
        {
            bool exist = false;
            for(type x : vec)
                exist |= (x==val);
            return exist;
        }

        template <typename type>
        int findInVector(const std::vector<type>& vec, const type& val)
        {
            for(int i=0, N=vec.size(); i<N; ++i)
                if(vec[i]==val) return i;
            return -1;
        }

        template<typename type> class JoinedIterator
        {
          private:
            JoinedIterator();
            std::vector<std::vector<type> const*> m_pointersToVectors;
            std::vector<int> m_innerSizes;
            int m_outerSize, m_outerIdx, m_innerIdx;
            type const* m_value_p;

          public:
            JoinedIterator(std::vector<std::vector<type> const*> const& pointersToVectors, const int iterType=0)
            {
                m_pointersToVectors = pointersToVectors;

                for(auto ptrToVec : m_pointersToVectors)
                    m_innerSizes.push_back(ptrToVec->size());
                m_outerSize = m_innerSizes.size();

                if(iterType == 0)
                {
                    m_outerIdx = 0;
                    m_innerIdx = 0;
                    m_value_p = &(m_pointersToVectors[m_outerIdx]->at(m_innerIdx));
                }
                else
                {
                    m_outerIdx = m_pointersToVectors.size()-1;
                    m_innerIdx = m_innerSizes[m_outerIdx];
                    m_value_p  = 0;
                }
            }

            bool operator== (JoinedIterator const& other) const
            {
                return (m_outerIdx == other.m_outerIdx && m_innerIdx == other.m_innerIdx)? true : false;
            }
            bool operator!= (JoinedIterator const& other) const
            {
                return !(*this == other);
            }

            type const& operator*()  const { return *m_value_p; }
            type const* operator->() const { return  m_value_p; }

            JoinedIterator& operator++()
            {
                if(++m_innerIdx == m_innerSizes[m_outerIdx])
                {
                    if(m_outerIdx < m_outerSize-1) {
                        ++m_outerIdx; m_innerIdx = 0;
                        m_value_p = &(m_pointersToVectors[m_outerIdx]->at(m_innerIdx));
                    } else
                        m_value_p = 0;
                } else
                    m_value_p = &(m_pointersToVectors[m_outerIdx]->at(m_innerIdx));

                return *this;
            }

            JoinedIterator operator++(int)
            {
                JoinedIterator rc(*this);
                this->operator++();
                return rc;
            }
        };

        // mechanism to extract the value of an active scalar that is compatible with passive
        template<typename T>
        inline float getVal(const T& x) {return x.value();}
        template<>
        inline float getVal<float>(const float& x) {return x;}

        // convert connectivity information from list of lists to compressed row storage
        // the latter providing more efficient access
        template<typename T>
        void makeCRSfromLIL(const std::vector<std::vector<T> >& LIL,
                            std::vector<int>& outerIdxPtr,
                            std::vector<T>& rowData)
        {
            int nRows = LIL.size();

            outerIdxPtr.reserve(nRows+1);
            outerIdxPtr.push_back(0);
            for(int i=0; i<nRows; ++i)
              outerIdxPtr.push_back(outerIdxPtr[i]+LIL[i].size());

            rowData.reserve(outerIdxPtr[nRows]);
            for(int i=0; i<nRows; ++i)
              for(int j=0; j<int(LIL[i].size()); ++j)
                rowData.push_back(LIL[i][j]);
        }
    }
}

#endif // MATRIX_H
