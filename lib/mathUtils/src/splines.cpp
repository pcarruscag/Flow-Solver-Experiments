//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "splines.h"
#include "solvers.h"
#include <cassert>
#include <math.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

using namespace Eigen;

namespace mathUtils{
namespace interpolation{

template <typename type> Cspline<type>::Cspline()
{
    m_dataIsReady = false;
}

template <typename type> Cspline<type>::Cspline(const std::vector<type> &x,
                                                const std::vector<type> &y,
                                                const double endCondition)
{
    setData(x,y,endCondition);
}

template <typename type> void Cspline<type>::setData(const std::vector<type> &x,
                                                     const std::vector<type> &y,
                                                     const double endCondition)
{
    std::vector<type> ldiag, mdiag, udiag, b;
    assert(x.size()==y.size());
    assert(x.size()>1);

    m_x = x;
    m_y = y;
    m_nPt = x.size();

    m_dx.resize(m_nPt-1);
    ldiag.resize(m_nPt);
    mdiag.resize(m_nPt);
    udiag.resize(m_nPt);
    b.resize(m_nPt);

    for(int i=0; i<m_nPt-1; i++)
    {
        m_dx[i] = x[i+1]-x[i];
    }

    mdiag[0] = 1.0;  udiag[0] = -endCondition;  b[0] = 0.0;

    for(int i=1; i<m_nPt-1; i++)
    {
        ldiag[i] = m_dx[i-1]/6.0;
        mdiag[i] = (m_dx[i]+m_dx[i-1])/3.0;
        udiag[i] = m_dx[i]/6.0;
        b[i] = (y[i+1]-y[i])/m_dx[i]-(y[i]-y[i-1])/m_dx[i-1];
    }
    ldiag[m_nPt-1] = -endCondition;  mdiag[m_nPt-1] = 1.0;  b[m_nPt-1] = 0.0;

    assert(mathUtils::solvers::tdma(ldiag,mdiag,udiag,b,m_d2y));
    m_dataIsReady = true;
}

template <typename type> int Cspline<type>::m_getIndex(const type &x) const
{
    assert(m_dataIsReady);
    int index = 0;

    if(x<m_x[0])
    {
        assert(m_x[0]-x<m_extrapTol*(m_x[1]-m_x[0]));
        return 0;
    }
    for(index=0; index<m_nPt-1; index++)
    {
        if(x>=m_x[index] && x<m_x[index+1])
            break;
    }
    if(index==m_nPt-1)
    {
        index = m_nPt-2;
        assert(x-m_x[m_nPt-1]<m_extrapTol*(m_x[m_nPt-1]-m_x[m_nPt-2]));
    }
    return index;
}

template <typename type> type Cspline<type>::m_evaluate(const type &x) const
{
    int index = m_getIndex(x);

    return (pow(m_x[index+1]-x,3.0)/m_dx[index]-m_dx[index]*(m_x[index+1]-x))*m_d2y[index]/6.0+
           (pow(x-m_x[index] , 3.0)/m_dx[index]-m_dx[index]*(x - m_x[index]))*m_d2y[index+1]/6.0+
           (m_y[index]*(m_x[index+1]-x)+m_y[index+1]*(x-m_x[index]))/m_dx[index];
}

template <typename type> type Cspline<type>::m_deriv(const type &x) const
{
    int index = m_getIndex(x);

    return (-3.0*pow(m_x[index+1]-x,2.0)/m_dx[index]+m_dx[index])*m_d2y[index] / 6.0+
           ( 3.0*pow(x-m_x[index] , 2.0)/m_dx[index]-m_dx[index])*m_d2y[index+1]/6.0+
           (m_y[index+1]-m_y[index])/m_dx[index];
}

template <typename type> type Cspline<type>::operator()(const type x) const
{
    return m_evaluate(x);
}

template <typename type> std::vector<type> Cspline<type>::operator()(const std::vector<type> &x) const
{
    int N = x.size();
    std::vector<type> y(N);
    for(int i=0; i<N; i++)
        y[i] = m_evaluate(x[i]);
    return y;
}

template <typename type> type Cspline<type>::deriv(const type x) const
{
    return m_deriv(x);
}

template <typename type> std::vector<type> Cspline<type>::deriv(const std::vector<type> &x) const
{
    int N = x.size();
    std::vector<type> y(N);
    for(int i=0; i<N; i++)
        y[i] = m_deriv(x[i]);
    return y;
}

template <typename type> PiecewiseLinear<type>::PiecewiseLinear()
{
    m_dataIsReady = false;
}

template <typename type> PiecewiseLinear<type>::PiecewiseLinear(const std::vector<type> &x,
                                                                const std::vector<type> &y)
{
    setData(x,y);
}

template <typename type> void PiecewiseLinear<type>::setData(const std::vector<type> &x,
                                                             const std::vector<type> &y)
{
    assert(x.size()==y.size());
    assert(x.size()>1);
    m_x = x;
    m_y = y;
    m_nPt = x.size();
    m_dataIsReady = true;
}

template <typename type> int PiecewiseLinear<type>::m_getIndex(const type &x) const
{
    assert(m_dataIsReady);
    int index = 0;

    if(x<m_x[0])
    {
        assert(m_x[0]-x<m_extrapTol*(m_x[1]-m_x[0]));
        return 0;
    }
    for(index=0; index<m_nPt-1; index++)
    {
        if(x>=m_x[index] && x<m_x[index+1])
            break;
    }
    if(index==m_nPt-1)
    {
        index = m_nPt-2;
        assert(x-m_x[m_nPt-1]<m_extrapTol*(m_x[m_nPt-1]-m_x[m_nPt-2]));
    }
    return index;
}

template <typename type> type PiecewiseLinear<type>::m_evaluate(const type &x) const
{
    int index = m_getIndex(x);
    return m_y[index]+(m_y[index+1]-m_y[index])/(m_x[index+1]-m_x[index])*(x-m_x[index]);
}

template <typename type> type PiecewiseLinear<type>::operator()(const type x) const
{
    return m_evaluate(x);
}

template <typename type> std::vector<type> PiecewiseLinear<type>::operator()(const std::vector<type> &x) const
{
    int N = x.size();
    std::vector<type> y(N);
    for(int i=0; i<N; i++)
        y[i] = m_evaluate(x[i]);
    return y;
}

template <typename type> std::vector<type> splineInterp(const std::vector<type> &x,
                                                        const std::vector<type> &y,
                                                        const std::vector<type> &xi,
                                                        const double endCondition)
{
    Cspline<type> spline(x,y,endCondition);
    return spline(xi);
}

template <typename type> std::vector<type> pwiseLinInterp(const std::vector<type> &x,
                                                        const std::vector<type> &y,
                                                        const std::vector<type> &xi)
{
    PiecewiseLinear<type> spline(x,y);
    return spline(xi);
}

// Explicit Instanciations
typedef AutoDiffScalar<VectorXd> adtype;

template class Cspline<double>;
template class Cspline<adtype>;

template std::vector<double> splineInterp(const std::vector<double> &x,
                                          const std::vector<double> &y,
                                          const std::vector<double> &xi,
                                          const double endCondition=0.0);
template std::vector<adtype> splineInterp(const std::vector<adtype> &x,
                                          const std::vector<adtype> &y,
                                          const std::vector<adtype> &xi,
                                          const double endCondition=0.0);
template std::vector<double> pwiseLinInterp(const std::vector<double> &x,
                                          const std::vector<double> &y,
                                          const std::vector<double> &xi);
}}
