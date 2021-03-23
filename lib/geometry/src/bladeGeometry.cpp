//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "bladeGeometry.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>

using namespace mathUtils::interpolation;

namespace geometryGeneration {

BladeGeometry::BladeGeometry()
{
    m_errorCode = 1;
}

int BladeGeometry::buildGeometry(const fileManagement::GeometryParamManager &geoParams,
                                 const fileManagement::MeshParamManager &meshParams,
                                 const bool fixSize)
{
    if(!geoParams.dataIsReady() || !meshParams.dataIsReady())
    {
        m_errorCode = 2;
        return m_errorCode;
    }
    if(fixSize && m_errorCode!=0) return 3;

    int nPtHub = geoParams.m_nPtHub,   nPtShr   = geoParams.m_nPtShr,
        nSpan  = geoParams.m_nSpanSec, nPtTheta = geoParams.m_nPtTheta,
        nPtThick   = geoParams.m_nPtThick;
    m_nIndepVars = 2*(nPtHub+nPtShr)+nSpan*(2+nPtTheta);

    #define ADZERO adtype(0.0,VectorXd::Zero(m_nIndepVars))

    // -------- INITIALIZE VARIABLES FOR DIFFERENTIATION ---------- //

    // hub and shroud control points
    std::vector<adtype> zHubCP(nPtHub,ADZERO), rHubCP(nPtHub,ADZERO), tHub(nPtHub,ADZERO);
    for(int i=0; i<nPtHub; i++)
    {
        zHubCP[i] = adtype(geoParams.m_xHubPt[i],m_nIndepVars,2*i  );
        rHubCP[i] = adtype(geoParams.m_yHubPt[i],m_nIndepVars,2*i+1);
        //parametric variable
        tHub[i] = adtype(double(i)/double(nPtHub-1),VectorXd::Zero(m_nIndepVars));
    }
    std::vector<adtype> zShrCP(nPtShr,ADZERO), rShrCP(nPtShr,ADZERO), tShr(nPtShr,ADZERO);
    for(int i=0; i<nPtShr; i++)
    {
        zShrCP[i] = adtype(geoParams.m_xShrPt[i],m_nIndepVars,2*nPtHub+2*i  );
        rShrCP[i] = adtype(geoParams.m_yShrPt[i],m_nIndepVars,2*nPtHub+2*i+1);
        tShr[i] = adtype(double(i)/double(nPtShr-1),VectorXd::Zero(m_nIndepVars));
    }
    // le and te position
    std::vector<adtype> mLeCP(nSpan,ADZERO), mTeCP(nSpan,ADZERO);
    for(int i=0; i<nSpan; i++)
    {
        mLeCP[i] = adtype(geoParams.m_yLePt[i],m_nIndepVars,2*(nPtHub+nPtShr)+2*i  );
        mTeCP[i] = adtype(geoParams.m_yTePt[i],m_nIndepVars,2*(nPtHub+nPtShr)+2*i+1);
    }
    // theta control points
    std::vector<std::vector<adtype> > wBldCP(nSpan,std::vector<adtype>(nPtTheta,ADZERO));
    for(int i=0; i<nSpan; i++)
      for(int j=0; j<nPtTheta; j++)
        wBldCP[i][j] = adtype(geoParams.m_yThetaPt[i][j],m_nIndepVars,
                              2*(nPtHub+nPtShr+nSpan)+i*nPtTheta+j);

    // thickness is not used for differentiation but still needs to be of type F
    // because the meridional location of the control points depends on inputs
    std::vector<std::vector<adtype> > xThickCP(nSpan,std::vector<adtype>(nPtThick,ADZERO)),
                                      yThickCP(nSpan,std::vector<adtype>(nPtThick,ADZERO));
    for(int i=0; i<nSpan; i++)
    {
        for(int j=0; j<nPtThick; j++)
        {
            xThickCP[i][j] = geoParams.m_xThickPt[i][j];
            yThickCP[i][j] = geoParams.m_yThickPt[i][j];
        }
    }

    // -------------------- GENERATE GEOMETRY ---------------------- //

    // ### create hub and shroud splines ###
    m_zHubSpline.setData(tHub,zHubCP); m_rHubSpline.setData(tHub,rHubCP);
    m_zShrSpline.setData(tShr,zShrCP); m_rShrSpline.setData(tShr,rShrCP);

    // ### meridional to parametric coordinate maps ###
    std::vector<adtype> meriLen(nSpan,ADZERO);
    m_meri2tSplines.resize(nSpan);
    m_buildMeri2tSplines(geoParams,m_zHubSpline,m_rHubSpline,m_zShrSpline,m_rShrSpline,
                         m_meri2tSplines,meriLen);
    // ### mesh blade surface ###
    Cspline<adtype> pBldSpline, thickSpline;
    std::vector<adtype> mBldCP(nPtTheta,ADZERO), pBldCP(nPtTheta,ADZERO), mThickCP(nPtThick,ADZERO);

    int nPtSeed = geoParams.m_advPar.nPtBldSurfInt;
    std::vector<adtype> betaLE(nSpan,ADZERO), betaTE(nSpan,ADZERO);

    int nPtMeshBld = meshParams.m_edgePar.nCellBld, nMg = meshParams.m_advPar.nMultiGrid;
    // make number of cells (Ncell=Npt-1) divisible by 2^nMg
    nPtMeshBld /= pow(2,nMg); nPtMeshBld *= pow(2,nMg); nPtMeshBld++;
    double alphaBld = meshParams.m_edgePar.alphaBld, alphaLeTe = meshParams.m_edgePar.alphaLeTe,
           pitch = 2.0*M_PI/geoParams.m_nBlades;
    std::vector<adtype> mMeshLe(nSpan,ADZERO), pMeshLe(nSpan,ADZERO), mMeshTe(nSpan,ADZERO),
                        pMeshTe(nSpan,ADZERO), ds0Mesh(nSpan,ADZERO), tMeshPts(nPtMeshBld,ADZERO);

    std::vector<std::vector<adtype> > mMeshLowPts(nSpan,std::vector<adtype>(nPtMeshBld,ADZERO)),
                                      pMeshLowPts(nSpan,std::vector<adtype>(nPtMeshBld,ADZERO)),
                                      mMeshUpPts (nSpan,std::vector<adtype>(nPtMeshBld,ADZERO)),
                                      pMeshUpPts (nSpan,std::vector<adtype>(nPtMeshBld,ADZERO));

    // mixed linear-raised cosine spacing for meshing
    for(int i=0; i<nPtMeshBld; i++)
    {
        tMeshPts[i] = double(i)/double(nPtMeshBld-1);
        tMeshPts[i]-= (1.0-alphaBld)*sin(2.0*M_PI*tMeshPts[i])/(2.0*M_PI);
    }

    for(int i=0; i<nSpan; i++)
    {
        // splines for wrap angle and thickness
        double span = double(i)/double(nSpan-1);
        for(int j=0; j<nPtTheta; j++)
            mBldCP[j] = (double(j)/double(nPtTheta-1)*(mTeCP[i]-mLeCP[i])+mLeCP[i])*meriLen[i];
        std::vector<adtype> tBldCP = m_meri2tSplines[i](mBldCP),
                            rHubBldCP = m_rHubSpline(tBldCP),
                            rShrBldCP = m_rShrSpline(tBldCP);
        for(int j=0; j<nPtTheta; j++)
            pBldCP[j] = wBldCP[i][j]*((1.0-span)*rHubBldCP[j]+span*rShrBldCP[j]);
        pBldSpline.setData(mBldCP,pBldCP);

        for(int j=0; j<nPtThick; j++)
            mThickCP[j] = (xThickCP[i][j]*(mTeCP[i]-mLeCP[i])+mLeCP[i])*meriLen[i];
        thickSpline.setData(mThickCP,yThickCP[i]);

        // seed blade surfaces
        std::vector<adtype> mSeedMidPts(nPtSeed,ADZERO);
        for(int j=0; j<nPtSeed; j++)
            mSeedMidPts[j] = (0.5*(1.0-cos(double(j)/double(nPtSeed-1)*M_PI))*(mTeCP[i]-mLeCP[i])+mLeCP[i])*meriLen[i];
        std::vector<adtype> pSeedMidPts  = pBldSpline(mSeedMidPts),
                            thickSeedPts = thickSpline(mSeedMidPts),
                            slopeSeedPts = pBldSpline.deriv(mSeedMidPts);
        std::vector<adtype> mSeedLowPts(nPtSeed,ADZERO), mSeedUpPts(nPtSeed,ADZERO),
                            pSeedLowPts(nPtSeed,ADZERO), pSeedUpPts(nPtSeed,ADZERO),
                            betaSeedPts(nPtSeed,ADZERO);
        for(int j=0; j<nPtSeed; j++)
        {
            betaSeedPts[j] = atan2(slopeSeedPts[j],adtype(1,VectorXd::Zero(m_nIndepVars)));
            mSeedLowPts[j] = mSeedMidPts[j]+0.5*thickSeedPts[j]*sin(betaSeedPts[j]);
            pSeedLowPts[j] = pSeedMidPts[j]-0.5*thickSeedPts[j]*cos(betaSeedPts[j]);
            mSeedUpPts[j]  = mSeedMidPts[j]-0.5*thickSeedPts[j]*sin(betaSeedPts[j]);
            pSeedUpPts[j]  = pSeedMidPts[j]+0.5*thickSeedPts[j]*cos(betaSeedPts[j]);
        }
        betaLE[i] = betaSeedPts[0]; // le/te angle for inlet outlet sections
        betaTE[i] = betaSeedPts[nPtSeed-1];

        // coordinate along blade surfaces
        std::vector<adtype> sSeedLowPts(nPtSeed,ADZERO), sSeedUpPts(nPtSeed,ADZERO);
        sSeedLowPts[0] = 0.0; sSeedUpPts[0] = 0.0;
        for(int j=1; j<nPtSeed; j++)
        {
            sSeedLowPts[j] = sSeedLowPts[j-1]+sqrt(
                pow(mSeedLowPts[j]-mSeedLowPts[j-1],2.0)+
                pow(pSeedLowPts[j]-pSeedLowPts[j-1],2.0));
            sSeedUpPts[j] = sSeedUpPts[j-1]+sqrt(
                pow(mSeedUpPts[j]-mSeedUpPts[j-1],2.0)+
                pow(pSeedUpPts[j]-pSeedUpPts[j-1],2.0));
        }

        // mesh points
        std::vector<adtype> sMeshLowPts(nPtMeshBld,ADZERO), sMeshUpPts(nPtMeshBld,ADZERO);
        for(int j=0; j<nPtMeshBld; j++)
        {
            sMeshLowPts[j] = tMeshPts[j]*sSeedLowPts[nPtSeed-1];
            sMeshUpPts[j]  = tMeshPts[j]*sSeedUpPts[nPtSeed-1];
        }
        ds0Mesh[i] = 0.5*(sMeshLowPts[1]+sMeshUpPts[1]);
        mMeshLowPts[i] = splineInterp(sSeedLowPts,mSeedLowPts,sMeshLowPts);
        pMeshLowPts[i] = splineInterp(sSeedLowPts,pSeedLowPts,sMeshLowPts);
        mMeshUpPts[i]  = splineInterp(sSeedUpPts, mSeedUpPts, sMeshUpPts);
        pMeshUpPts[i]  = splineInterp(sSeedUpPts, pSeedUpPts, sMeshUpPts);
        mMeshLe[i] = 0.5*(mMeshLowPts[i][0]+mMeshUpPts[i][0]);
        pMeshLe[i] = 0.5*(pMeshLowPts[i][0]+pMeshUpPts[i][0]);
        mMeshTe[i] = 0.5*(mMeshLowPts[i][nPtMeshBld-1]+mMeshUpPts[i][nPtMeshBld-1]);
        pMeshTe[i] = 0.5*(pMeshLowPts[i][nPtMeshBld-1]+pMeshUpPts[i][nPtMeshBld-1]);

        // separate up and low by one pitch
        std::vector<adtype> tMeshLowPts = m_meri2tSplines[i](mMeshLowPts[i]);
        {
        std::vector<adtype> rMeshHubPts = m_rHubSpline(tMeshLowPts),
                            rMeshShrPts = m_rShrSpline(tMeshLowPts);
        for(int j=0; j<nPtMeshBld; j++)
            pMeshLowPts[i][j] += 0.5*pitch*((1.0-span)*rMeshHubPts[j]+span*rMeshShrPts[j]);
        }
        std::vector<adtype> tMeshUpPts = m_meri2tSplines[i](mMeshUpPts[i]);
        {
        std::vector<adtype> rMeshHubPts = m_rHubSpline(tMeshUpPts),
                            rMeshShrPts = m_rShrSpline(tMeshUpPts);
        for(int j=0; j<nPtMeshBld; j++)
            pMeshUpPts[i][j] -= 0.5*pitch*((1.0-span)*rMeshHubPts[j]+span*rMeshShrPts[j]);
        }
    }

    // ### inlet section ###
    // determine number of cells, the number of cells is a dependent variable but its calculation
    // is non differentiable because it involves rounding
    double maxRatio = 0.0;
    std::vector<adtype> inletLen(nSpan,ADZERO);
    for(int j=0; j<nSpan; j++)
    {
        inletLen[j] = meriLen[j]*mLeCP[j]/cos(betaLE[j]);
        maxRatio = std::max(maxRatio,inletLen[j].value()/ds0Mesh[j].value());
    }
    int nPtMeshIn;
    if(fixSize) nPtMeshIn = m_leIndex;
    else        nPtMeshIn = int(log(1.0-maxRatio*(1.0-alphaLeTe))/log(alphaLeTe));
    nPtMeshIn /= pow(2,nMg); nPtMeshIn *= pow(2,nMg); //nPtMeshIn++;
    std::vector<std::vector<adtype> > mMeshInPts(nSpan,std::vector<adtype>(nPtMeshIn+1,ADZERO)),
                                      pMeshInPts(nSpan,std::vector<adtype>(nPtMeshIn+1,ADZERO));
    {
    std::vector<adtype> cumSum(nPtMeshIn+1,ADZERO);
    cumSum[0] = 0.0;
    for(int j=1; j<nPtMeshIn+1; j++)
        cumSum[j] = cumSum[j-1]+pow(alphaLeTe,j-1);
    for(int j=0; j<nSpan; j++)
    {
        for(int k=0; k<nPtMeshIn+1; k++)
        {
            adtype sMeshInOut = cumSum[nPtMeshIn-k]*inletLen[j]/cumSum[nPtMeshIn];
            mMeshInPts[j][k] = mMeshLe[j]-sMeshInOut*cos(betaLE[j]);
            pMeshInPts[j][k] = pMeshLe[j]-sMeshInOut*sin(betaLE[j]);
        }
    }
    }
    // ### outlet section ###
    maxRatio = 0.0;
    std::vector<adtype> outletLen(nSpan,ADZERO);
    for(int j=0; j<nSpan; j++)
    {
        outletLen[j] = meriLen[j]*(1.0-mTeCP[j])/cos(betaTE[j]);
        maxRatio = std::max(maxRatio,outletLen[j].value()/ds0Mesh[j].value());
    }
    int nPtMeshOut;
    if(fixSize) nPtMeshOut = m_nPtMeri-m_teIndex-1;
    else        nPtMeshOut = int(log(1.0-maxRatio*(1.0-alphaLeTe))/log(alphaLeTe));
    nPtMeshOut /= pow(2,nMg); nPtMeshOut *= pow(2,nMg); //nPtMeshOut++;
    std::vector<std::vector<adtype> > mMeshOutPts(nSpan,std::vector<adtype>(nPtMeshOut+1,ADZERO)),
                                      pMeshOutPts(nSpan,std::vector<adtype>(nPtMeshOut+1,ADZERO));
    {
    std::vector<adtype> cumSum(nPtMeshOut+1,ADZERO);
    cumSum[0] = 0.0;
    for(int j=1; j<nPtMeshOut+1; j++)
        cumSum[j] = cumSum[j-1]+pow(alphaLeTe,j-1);
    for(int j=0; j<nSpan; j++)
    {
        for(int k=0; k<nPtMeshOut+1; k++)
        {
            adtype sMeshInOut = cumSum[k]*outletLen[j]/cumSum[nPtMeshOut];
            mMeshOutPts[j][k] = mMeshTe[j]+sMeshInOut*cos(betaTE[j]);
            pMeshOutPts[j][k] = pMeshTe[j]+sMeshInOut*sin(betaTE[j]);
        }
    }
    }
    // ### create up and down for inlet and outlet sections
    std::vector<std::vector<adtype> > pMeshInLowPts (nSpan,std::vector<adtype>(nPtMeshIn+1,ADZERO)),
                                      pMeshInUpPts  (nSpan,std::vector<adtype>(nPtMeshIn+1,ADZERO)),
                                      pMeshOutLowPts(nSpan,std::vector<adtype>(nPtMeshOut+1,ADZERO)),
                                      pMeshOutUpPts (nSpan,std::vector<adtype>(nPtMeshOut+1,ADZERO));

    for(int i=0; i<nSpan; i++)
    {
        double span = double(i)/double(nSpan-1);
        std::vector<adtype> tMeshInOutPts = m_meri2tSplines[i](mMeshInPts[i]),
                            rMeshHubPts = m_rHubSpline(tMeshInOutPts),
                            rMeshShrPts = m_rShrSpline(tMeshInOutPts);
        for(int j=0; j<nPtMeshIn+1; j++)
        {
            pMeshInLowPts[i][j] = pMeshInPts[i][j]+0.5*pitch*((1.0-span)*rMeshHubPts[j]+span*rMeshShrPts[j]);
            pMeshInUpPts[i][j]  = pMeshInPts[i][j]-0.5*pitch*((1.0-span)*rMeshHubPts[j]+span*rMeshShrPts[j]);
        }
    }
    for(int i=0; i<nSpan; i++)
    {
        double span = double(i)/double(nSpan-1);
        std::vector<adtype> tMeshInOutPts = m_meri2tSplines[i](mMeshOutPts[i]),
                            rMeshHubPts = m_rHubSpline(tMeshInOutPts),
                            rMeshShrPts = m_rShrSpline(tMeshInOutPts);
        for(int j=0; j<nPtMeshOut+1; j++)
        {
            pMeshOutLowPts[i][j] = pMeshOutPts[i][j]+0.5*pitch*((1.0-span)*rMeshHubPts[j]+span*rMeshShrPts[j]);
            pMeshOutUpPts[i][j]  = pMeshOutPts[i][j]-0.5*pitch*((1.0-span)*rMeshHubPts[j]+span*rMeshShrPts[j]);
        }
    }

    // ----------------- MERGE INLET BLADE AND OUTLET ---------------- //

    m_nPtMeri = nPtMeshIn+nPtMeshBld+nPtMeshOut;
    m_leIndex = nPtMeshIn;
    m_teIndex = nPtMeshIn+nPtMeshBld-1;
    m_meriCoord.resize(nSpan);
    m_pitchCoord.resize(nSpan);

    for(int i=0; i<nSpan; i++)
    {
        m_meriCoord[i].resize(2*m_nPtMeri,1);
        m_pitchCoord[i].resize(2*m_nPtMeri,1);
        for(int j=0; j<nPtMeshIn; j++)
        {
            m_meriCoord[i](j,0) = mMeshInPts[i][j].value();
            m_meriCoord[i](j+m_nPtMeri,0) = mMeshInPts[i][j].value();
            m_pitchCoord[i](j,0) = pMeshInLowPts[i][j].value();
            m_pitchCoord[i](j+m_nPtMeri,0) = pMeshInUpPts[i][j].value();
        }
        for(int j=0; j<nPtMeshBld; j++)
        {
            m_meriCoord[i](j+nPtMeshIn,0) = mMeshLowPts[i][j].value();
            m_meriCoord[i](j+nPtMeshIn+m_nPtMeri,0) = mMeshUpPts[i][j].value();
            m_pitchCoord[i](j+nPtMeshIn,0) = pMeshLowPts[i][j].value();
            m_pitchCoord[i](j+nPtMeshIn+m_nPtMeri,0) = pMeshUpPts[i][j].value();
        }
        for(int j=0; j<nPtMeshOut; j++)
        {
            m_meriCoord[i](j+nPtMeshIn+nPtMeshBld,0) = mMeshOutPts[i][j+1].value();
            m_meriCoord[i](j+nPtMeshIn+nPtMeshBld+m_nPtMeri,0) = mMeshOutPts[i][j+1].value();
            m_pitchCoord[i](j+nPtMeshIn+nPtMeshBld,0) = pMeshOutLowPts[i][j+1].value();
            m_pitchCoord[i](j+nPtMeshIn+nPtMeshBld+m_nPtMeri,0) = pMeshOutUpPts[i][j+1].value();
        }
    }

    // ----------------------- GET DERIVATIVES ----------------------- //

    m_meriCoordDeriv.resize(nSpan);
    m_pitchCoordDeriv.resize(nSpan);

    for(int i=0; i<nSpan; i++)
    {
        m_meriCoordDeriv[i].resize(m_nPtMeri*2,m_nIndepVars);
        m_pitchCoordDeriv[i].resize(m_nPtMeri*2,m_nIndepVars);
        for(int j=0; j<nPtMeshIn; j++)
        {
            for(int k=0; k<m_nIndepVars; k++)
            {
                m_meriCoordDeriv[i](j,k) = mMeshInPts[i][j].derivatives()(k);
                m_meriCoordDeriv[i](j+m_nPtMeri,k) = mMeshInPts[i][j].derivatives()(k);
                m_pitchCoordDeriv[i](j,k) = pMeshInLowPts[i][j].derivatives()(k);
                m_pitchCoordDeriv[i](j+m_nPtMeri,k) = pMeshInUpPts[i][j].derivatives()(k);
            }
        }
        for(int j=0; j<nPtMeshBld; j++)
        {
            for(int k=0; k<m_nIndepVars; k++)
            {
                m_meriCoordDeriv[i](j+nPtMeshIn,k) = mMeshLowPts[i][j].derivatives()(k);
                m_meriCoordDeriv[i](j+nPtMeshIn+m_nPtMeri,k) = mMeshUpPts[i][j].derivatives()(k);
                m_pitchCoordDeriv[i](j+nPtMeshIn,k) = pMeshLowPts[i][j].derivatives()(k);
                m_pitchCoordDeriv[i](j+nPtMeshIn+m_nPtMeri,k) = pMeshUpPts[i][j].derivatives()(k);
            }
        }
        for(int j=0; j<nPtMeshOut; j++)
        {
            for(int k=0; k<m_nIndepVars; k++)
            {
                m_meriCoordDeriv[i](j+nPtMeshIn+nPtMeshBld,k) = mMeshOutPts[i][j+1].derivatives()(k);
                m_meriCoordDeriv[i](j+nPtMeshIn+nPtMeshBld+m_nPtMeri,k) = mMeshOutPts[i][j+1].derivatives()(k);
                m_pitchCoordDeriv[i](j+nPtMeshIn+nPtMeshBld,k) = pMeshOutLowPts[i][j+1].derivatives()(k);
                m_pitchCoordDeriv[i](j+nPtMeshIn+nPtMeshBld+m_nPtMeri,k) = pMeshOutUpPts[i][j+1].derivatives()(k);
            }
        }
    }
    #undef ADZERO
    m_errorCode = 0;
    return 0;
}

void BladeGeometry::m_buildMeri2tSplines(const fileManagement::GeometryParamManager &geoParams,
                                         const Cspline<adtype> &zHubSpline,
                                         const Cspline<adtype> &rHubSpline,
                                         const Cspline<adtype> &zShrSpline,
                                         const Cspline<adtype> &rShrSpline,
                                         std::vector<Cspline<adtype> > &meri2tSplines,
                                         std::vector<adtype> &meriLen)
{
    int downsampleFactor = geoParams.m_advPar.downsampleFactor,
        nPtInt = geoParams.m_advPar.nPtMeriIntegral,
        nSpan = geoParams.m_nSpanSec;
    std::vector<int> dwnsmplIndx;
    std::vector<adtype> zIntPtsHub(nPtInt), rIntPtsHub(nPtInt), tIntPts(nPtInt),
                        zIntPtsShr(nPtInt), rIntPtsShr(nPtInt), mIntPts(nPtInt);

    // downsampled vector to create splines
    int cursor = 0;
    while(cursor<nPtInt-1)
    {
        dwnsmplIndx.push_back(cursor);
        cursor += downsampleFactor;
    }
    dwnsmplIndx.push_back(nPtInt-1);
    int nPtIntDwnsmpl = dwnsmplIndx.size();
    std::vector<adtype> tIntPtsDwnsmpl(nPtIntDwnsmpl), mIntPtsDwnsmpl(nPtIntDwnsmpl);

    // sample hub and shroud curves
    for(int i=0; i<nPtInt; i++)
        tIntPts[i] = adtype(double(i)/double(nPtInt-1),VectorXd::Zero(m_nIndepVars));
    for(int i=0; i<nPtIntDwnsmpl; i++)
        tIntPtsDwnsmpl[i] = tIntPts[dwnsmplIndx[i]];
    zIntPtsHub = zHubSpline(tIntPts);
    rIntPtsHub = rHubSpline(tIntPts);
    zIntPtsShr = zShrSpline(tIntPts);
    rIntPtsShr = rShrSpline(tIntPts);

    // evenly spaced spanwise sections, shape given by linear combination of hub and shroud
    for(int i=0; i<nSpan; i++)
    {
        double span = double(i)/double(nSpan-1);
        mIntPts[0] = 0.0;
        for(int j=1; j<nPtInt; j++)
        {
            mIntPts[j] = mIntPts[j-1]+sqrt(
                pow((1.0-span)*(zIntPtsHub[j]-zIntPtsHub[j-1])+span*(zIntPtsShr[j]-zIntPtsShr[j-1]),2.0)+
                pow((1.0-span)*(rIntPtsHub[j]-rIntPtsHub[j-1])+span*(rIntPtsShr[j]-rIntPtsShr[j-1]),2.0));
        }
        meriLen[i] = mIntPts[nPtInt-1];

        for(int j=0; j<nPtIntDwnsmpl; j++)
            mIntPtsDwnsmpl[j] = mIntPts[dwnsmplIndx[j]];

        meri2tSplines[i].setData(mIntPtsDwnsmpl,tIntPtsDwnsmpl);
    }
}

int BladeGeometry::saveGeometry(const std::string &filePath) const
{
    if(m_errorCode!=0)
        return 1;
    std::ofstream outFile;
    outFile.open(filePath.c_str());
    if(!outFile.is_open())
        return 2;
    for(size_t i=0; i<m_meriCoord.size(); i++)
    {
        outFile << "SPANWISE SECTION: " << i << "," << std::endl;
        outFile << "Streamwise Coordinate," << std::endl;
        for(int j=0; j<2*m_nPtMeri; j++)
        {
            outFile << m_meriCoord[i](j,0) << "," << std::endl;
        }
        outFile << "Pitchwise Coordinate," << std::endl;
        for(int j=0; j<2*m_nPtMeri; j++)
        {
            outFile << m_pitchCoord[i](j,0) << "," << std::endl;
        }
    }
    outFile.close();
    return 0;
}

int BladeGeometry::saveJacobian(const std::string &filePath) const
{
    if(m_errorCode!=0)
        return 1;
    std::ofstream outFile;
    outFile.open(filePath.c_str());
    if(!outFile.is_open())
        return 2;
    for(size_t i=0; i<m_meriCoordDeriv.size(); i++)
    {
        outFile << "SPANWISE SECTION: " << i << "," << std::endl;
        outFile << "Streamwise Coordinate Derivatives," << std::endl;
        for(int j=0; j<2*m_nPtMeri; j++)
        {
            for(int k=0; k<m_nIndepVars; k++)
            {
                outFile << m_meriCoordDeriv[i](j,k) << ",";
            }
            outFile << std::endl;
        }
        outFile << "Pitchwise Coordinate Derivatives," << std::endl;
        for(int j=0; j<2*m_nPtMeri; j++)
        {
            for(int k=0; k<m_nIndepVars; k++)
            {
                outFile << m_pitchCoordDeriv[i](j,k) << ",";
            }
            outFile << std::endl;
        }
    }
    outFile.close();
    return 0;
}
}
