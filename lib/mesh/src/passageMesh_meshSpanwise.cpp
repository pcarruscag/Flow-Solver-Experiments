//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "passageMesh.h"

#include <unsupported/Eigen/AutoDiff>

namespace mesh
{
int PassageMesh::m_meshSpanwise(const int row, const int col, const int rows, const int cols,
                                const int nSpan, const bool computeJacobians)
{
    typedef AutoDiffScalar<VectorXd> adtype;
    int nVars = m_bladeGeo->m_nIndepVars;
    const adtype ADINI1 = adtype(0.0,VectorXd::Zero(nVars)),
                 ADINI2 = adtype(0.0,VectorXd::Zero(3*nSpan));

    // Map points to 3D
    std::vector<adtype> x(nSpan,ADINI1), y(nSpan,ADINI1), z(nSpan,ADINI1), theta(nSpan,ADINI1);
    std::vector<double> drdm(nSpan), dzdm(nSpan);

    for(int layer=0; layer<nSpan; layer++)
    {
        double span = double(layer)/double(nSpan-1),
               p = m_layerPitch[layer](row,col);
        adtype m = ADINI1; m = m_layerMeri[layer](row,col);
        adtype t = m_bladeGeo->m_meri2tSplines[layer](m),
               zHub = m_bladeGeo->m_zHubSpline(t),
               rHub = m_bladeGeo->m_rHubSpline(t),
               zShr = m_bladeGeo->m_zShrSpline(t),
               rShr = m_bladeGeo->m_rShrSpline(t);
        z[layer] = (1.0-span)*zHub+span*zShr;
        adtype r = (1.0-span)*rHub+span*rShr;
        theta[layer] = p/r;
        x[layer] = r*cos(theta[layer]);
        y[layer] = r*sin(theta[layer]);

        drdm[layer] = m_bladeGeo->m_meri2tSplines[layer].deriv(m).value();
        dzdm[layer] = drdm[layer];
        drdm[layer]*= (1.0-span)*m_bladeGeo->m_rHubSpline.deriv(t).value()+
                         span   *m_bladeGeo->m_rShrSpline.deriv(t).value();
        dzdm[layer]*= (1.0-span)*m_bladeGeo->m_zHubSpline.deriv(t).value()+
                         span   *m_bladeGeo->m_zShrSpline.deriv(t).value();
    }
    // Jacobian of 3D layer coordinates
    MatrixXd xCPjacobian(nSpan,nVars), yCPjacobian(nSpan,nVars), zCPjacobian(nSpan,nVars);

    for(int layer=0; layer<nSpan; layer++)
    {
        RowVectorXd dMeri, dPitch;
        dMeri  = m_layerJacobian[layer].row(row*cols+col)*m_bladeGeo->m_meriCoordDeriv[layer];
        dPitch = m_layerJacobian[layer].row(row*cols+col)*m_bladeGeo->m_pitchCoordDeriv[layer];
        double thetaVal = theta[layer].value();
        for(int var=0; var<nVars; var++)
        {
            xCPjacobian(layer,var) = x[layer].derivatives()(var)+drdm[layer]*(cos(thetaVal)+
                thetaVal*sin(thetaVal))*dMeri(var)-sin(thetaVal)*dPitch(var);
            yCPjacobian(layer,var) = y[layer].derivatives()(var)+drdm[layer]*(sin(thetaVal)-
                thetaVal*cos(thetaVal))*dMeri(var)+cos(thetaVal)*dPitch(var);
            zCPjacobian(layer,var) = z[layer].derivatives()(var)+dzdm[layer]*dMeri(var);
        }
    }
    // Initialize control points for differentiation
    std::vector<adtype> s(nSpan,ADINI2);
    for(int layer=0; layer<nSpan; layer++)
    {
        s[layer] = double(layer)/double(nSpan-1);
        x[layer] = adtype(x[layer].value(),3*nSpan,3*layer  );
        y[layer] = adtype(y[layer].value(),3*nSpan,3*layer+1);
        z[layer] = adtype(z[layer].value(),3*nSpan,3*layer+2);
    }
    // Calculate dimensional span
    adtype span = ADINI2;
    for(int i=1; i<nSpan; i++)
        span += sqrt(pow(x[i]-x[i-1],2.0)+pow(y[i]-y[i-1],2.0)+pow(z[i]-z[i-1],2.0));

    // Calculate required blending
    double dt= 1.0/double(m_nSpanGrdPts-1);
    adtype alfa = 1.0+(m_nearWallSize/span-dt)*2.0*M_PI/sin(2*M_PI*dt);

    // Calculate normalized sampling points
    std::vector<adtype> si(m_nSpanGrdPts,ADINI2);
    for(int i=0; i<m_nSpanGrdPts; i++)
        si[i] = i*dt-(1.0-alfa)*sin(2*M_PI*i*dt)/M_PI*0.5;

    // Interpolate spanwise and store volume mesh vertex coordinates
    std::vector<adtype> xi = mathUtils::interpolation::splineInterp(s,x,si),
                        yi = mathUtils::interpolation::splineInterp(s,y,si),
                        zi = mathUtils::interpolation::splineInterp(s,z,si);
    for(int k=0; k<m_nSpanGrdPts; k++)
    {
        m_vertexCoords(row*cols+col+k*rows*cols,0) = xi[k].value();
        m_vertexCoords(row*cols+col+k*rows*cols,1) = yi[k].value();
        m_vertexCoords(row*cols+col+k*rows*cols,2) = zi[k].value();
    }
    // Differentiate w.r.t 3D layer coordinates and compute final jacobian
    for(int k=0; k<m_nSpanGrdPts && computeJacobians; k++)
    {
        int vertex = row*cols+col+k*rows*cols;
        for(int var=0; var<nVars; var++)
        {
            #define SUM(src,dst) {                                      \
            double sum = 0.0;                                           \
            for(int l=0; l<nSpan; l++)                                  \
                sum += src[k].derivatives()(3*l  )*xCPjacobian(l,var)+  \
                       src[k].derivatives()(3*l+1)*yCPjacobian(l,var)+  \
                       src[k].derivatives()(3*l+2)*zCPjacobian(l,var);  \
            dst(vertex,var) = sum;}

            SUM(xi,m_xCoordJacobian)
            SUM(yi,m_yCoordJacobian)
            SUM(zi,m_zCoordJacobian)

            #undef SUM
        }
    }
    return 0;
}
}
