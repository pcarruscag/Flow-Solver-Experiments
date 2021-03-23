//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "solvers.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

using namespace Eigen;

namespace mathUtils{
namespace solvers{

template <typename type> bool tdma(const std::vector<type> &a,
                                   const std::vector<type> &b,
                                   std::vector<type> &c,
                                   std::vector<type> &d,
                                   std::vector<type> &x)
{
    type denom;
    int N=a.size();
    if(N<2) return false;
    if((a.size()+b.size()-c.size()-d.size())!=0) return false;

    x.resize(N);

    c[0]/=b[0];
    d[0]/=b[0];
    for(int i=1; i<N-1; i++)
    {
        denom = 1.0/(b[i]-a[i]*c[i-1]);
        c[i] *= denom;
        d[i] = (d[i]-a[i]*d[i-1])*denom;
    }
    x[N-1] = d[N-1];
    for(int i=N-2; i>-1; i--)
    {
        x[i] = d[i]-c[i]*x[i+1];
    }
    return true;
}

// Explicit Instanciations
typedef AutoDiffScalar<VectorXd> adtype;

template bool tdma<double>(const std::vector<double>&,
                           const std::vector<double>&,
                           std::vector<double>&,
                           std::vector<double>&,
                           std::vector<double>&);
template bool tdma<adtype>(const std::vector<adtype>&,
                           const std::vector<adtype>&,
                           std::vector<adtype>&,
                           std::vector<adtype>&,
                           std::vector<adtype>&);
}}
