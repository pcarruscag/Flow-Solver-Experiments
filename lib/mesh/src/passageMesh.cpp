//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "passageMesh.h"

#include "../../mathUtils/src/splines.h"


namespace mesh
{
PassageMesh::PassageMesh(const bool minimumMemory)
{
    m_isMeshed = false;
    m_minMemory= minimumMemory;
    m_isScaled = false;
    m_bladeGeo = NULL;
}

int PassageMesh::mesh(const geometryGeneration::BladeGeometry &bladeGeo,
                      const fileManagement::MeshParamManager  &meshParams)
{
    m_isMeshed = false;
    if(bladeGeo.m_errorCode != 0 or !meshParams.dataIsReady())
        return 1;

    m_nSpanGrdPts  = meshParams.m_volumePar.nCellSpan;
    m_nearWallSize = meshParams.m_volumePar.nearWallSize;

    bool verbose = true;
    if(omp_get_max_threads()>1) verbose = false;
    int nSpan = bladeGeo.m_meriCoord.size(),
        M = meshParams.m_layerPar.nCellPitch,
        N = bladeGeo.m_nPtMeri;
    m_jLE = bladeGeo.m_leIndex;
    m_jTE = bladeGeo.m_teIndex;

    std::vector<int> errorCode(nSpan,0);
    // Coarse level matrices
    MatrixXd Xc, Yc;

    // ### GENERATE MESH FOR LAYERS ###
    #ifdef MESH_VERBOSE
    std::cout << "\n### Meshing Layers ###\n" << std::endl;
    #endif
    m_layerMeri.resize(nSpan);
    m_layerPitch.resize(nSpan);
    #pragma omp parallel for schedule(static,1) private(Xc,Yc)
    for(int i=0; i<nSpan; i++)
    {
        m_layerMeri[i].resize(M,N);
        m_layerPitch[i].resize(M,N);
        // Set boundary values
        for(int j=0; j<N; j++)
        {
            m_layerMeri[i](0,j)    = bladeGeo.m_meriCoord[i](j,0);
            m_layerMeri[i](M-1,j)  = bladeGeo.m_meriCoord[i](N+j,0);
            m_layerPitch[i](0,j)   = bladeGeo.m_pitchCoord[i](j,0);
            m_layerPitch[i](M-1,j) = bladeGeo.m_pitchCoord[i](N+j,0);
        }
        // Algebraic initialization
        double weight;
        for(int k=1; k<M-1; k++){
            weight = 1.0-double(k)/double(M-1);
            for(int j=0; j<N; j++){
                m_layerMeri[i](k,j)  = m_layerMeri[i](0,j)*weight+
                                       m_layerMeri[i](M-1,j)*(1.0-weight);
                m_layerPitch[i](k,j) = m_layerPitch[i](0,j)*weight+
                                       m_layerPitch[i](M-1,j)*(1.0-weight);
            }
        }
        // Grid sequencing
        int jLEc, jTEc;
        for(int k=meshParams.m_advPar.nMultiGrid; k>0 && errorCode[i]==0; k--) {
            #ifdef MESH_VERBOSE
            #pragma omp critical
            {std::cout << "  Layer " << i << " - Coarse Grid " << meshParams.m_advPar.nMultiGrid-k+1 << std::endl;}
            #endif
            jLEc = (m_jLE-1)/pow(2,k)+1;
            jTEc = (m_jTE-1)/pow(2,k)+1;
            m_restrict(&Xc,&Yc,m_layerMeri[i],m_layerPitch[i],k);
            errorCode[i] = m_meshLayer(Xc,Yc,jLEc,jTEc,i,meshParams,verbose);
            m_interpolate(Xc,Yc,m_layerMeri[i],m_layerPitch[i],k);
        }
        if(errorCode[i]==0) {
            #ifdef MESH_VERBOSE
            #pragma omp critical
            {std::cout << "  Layer " << i << " - Fine Grid" << std::endl;}
            #endif
            errorCode[i] = m_meshLayer(m_layerMeri[i],m_layerPitch[i],m_jLE,m_jTE,i,meshParams,verbose);
        }
        if(errorCode[i]==0) {
            #ifdef MESH_VERBOSE
            #pragma omp critical
            {std::cout << "  Layer " << i << " - Expansion Layer" << std::endl;}
            #endif
            m_expansionLayer(m_layerMeri[i],m_layerPitch[i],meshParams.m_layerPar.nearWallSize,
                             meshParams.m_layerPar.nearInOutSizeRatio);
        }
    }
    for(int i=0; i<nSpan; i++) if(errorCode[i]!=0) return 2;

    // ### SHARED OPERATIONS ###
    if(m_volumeMesh(bladeGeo)!=0) return 3;

    m_isMeshed = true;
    return 0;
}

int PassageMesh::morph(const geometryGeneration::BladeGeometry& bladeGeo)
{
    if(bladeGeo.m_errorCode!=0 || !m_isMeshed) return 1;

    int M = m_layerMeri[0].rows(),
        N = m_layerMeri[0].cols(),
        NN = M*N;

    if(bladeGeo.m_nPtMeri != N) return 2;

    m_isMeshed = false;

    MatrixXd bndDef(2*N,2), layerDef(NN,2);
    #ifdef MESH_VERBOSE
    std::cout << "\n### Morphing Layers ###" << std::endl;
    #endif
    for(size_t layer=0; layer<m_layerMeri.size(); layer++)
    {
        // Calculate boundary displacement
        for(int j=0; j<N; j++)
        {
            bndDef( j ,0) = bladeGeo.m_meriCoord [layer]( j ,0)-m_layerMeri [layer]( 0 ,j);
            bndDef(j+N,0) = bladeGeo.m_meriCoord [layer](j+N,0)-m_layerMeri [layer](M-1,j);
            bndDef( j ,1) = bladeGeo.m_pitchCoord[layer]( j ,0)-m_layerPitch[layer]( 0 ,j);
            bndDef(j+N,1) = bladeGeo.m_pitchCoord[layer](j+N,0)-m_layerPitch[layer](M-1,j);
        }
        // Multiply by jacobian to get inner point displacement
        layerDef = m_layerJacobian[layer]*bndDef;

        // Deform layers
        for(int i=0; i<M; i++){
            for(int j=0; j<N; j++){
                int n=i*N+j;
                m_layerMeri[layer](i,j)  += layerDef(n,0);
                m_layerPitch[layer](i,j) += layerDef(n,1);
            }
        }
    }
    // ### SHARED OPERATIONS ###
    if(m_volumeMesh(bladeGeo)!=0) return 3;

    m_isMeshed = true;
    return 0;
}

int PassageMesh::scale(const float factor, const int nMeriVars)
{
    if(!m_isMeshed)
        return 1;
    int nVars = m_xCoordJacobian.cols();

    assert(factor>0.0f && "scale factor must be positive");
    if(nVars!=0)
    assert(nMeriVars>=0 && nMeriVars<=nVars && "invalid number of meridional variables");

    m_vertexCoords *= factor;
    if(nVars!=0) {
    m_xCoordJacobian.rightCols(nVars-nMeriVars) *= factor;
    m_yCoordJacobian.rightCols(nVars-nMeriVars) *= factor;
    m_zCoordJacobian.rightCols(nVars-nMeriVars) *= factor;
    }
    m_isScaled = true;
    m_scale_nMeriVars = nMeriVars;
    m_scale_factor = factor;

    return 0;
}

int PassageMesh::m_volumeMesh(const geometryGeneration::BladeGeometry& bladeGeo)
{
    m_isScaled = false;

    int nSpan = m_layerMeri.size();
    std::vector<int> errorCode(nSpan);
    // Calculate the Jacobian for each layer
    if(m_layerJacobian.size()==0)
        m_layerJacobian.resize(nSpan);
    #ifdef MESH_VERBOSE
    std::cout << std::endl << "### Calculating Jacobian for Layers ###" << std::endl;
    #endif
    #pragma omp parallel for schedule(dynamic,1)
    for(int i=0; i<nSpan; i++)
        errorCode[i] = m_computeLayerJacobian(m_layerMeri[i],m_layerPitch[i],i);
    for(int i=0; i<nSpan; i++) if(errorCode[i]!=0) return 1;

    // Generate volume mesh
    m_bladeGeo = &bladeGeo; // store for when class is used to compute the jacobians
    m_spanwiseDriver(!m_minMemory); // if saving memory, don't compute jacobians now
    return 0;
}

int PassageMesh::m_spanwiseDriver(const bool computeJacobians)
{
    int nVars = m_bladeGeo->m_nIndepVars*int(computeJacobians),
        nSpan = m_layerMeri.size(),
        nVertex = m_nSpanGrdPts*m_layerJacobian[0].rows();
    m_vertexCoords.resize(nVertex,3);
    m_xCoordJacobian.resize(nVertex,nVars);
    m_yCoordJacobian.resize(nVertex,nVars);
    m_zCoordJacobian.resize(nVertex,nVars);

    #ifdef MESH_VERBOSE
    if(!computeJacobians || !m_minMemory)
    std::cout << std::endl << "### Generating Volume Mesh ###" << std::endl;
    else
    std::cout << std::endl << "### Computing Volume Mesh Jacobians ###" << std::endl;
    #endif
    int M = m_layerMeri[0].rows(),
        N = m_layerMeri[0].cols();
    #pragma omp parallel for collapse(2) schedule(dynamic,1000)
    for(int i=0; i<M; i++)
      for(int j=0; j<N; j++)
        m_meshSpanwise(i,j,M,N,nSpan,computeJacobians);

    return 0;
}

void PassageMesh::m_restrict(MatrixXd* Xc, MatrixXd* Yc,
                             const Ref<const MatrixXd> Xf, const Ref<const MatrixXd> Yf,
                             int mgLvl) const
{
    int Mc, Nc, Mf, Nf, step;
    Mf = Xf.rows();
    Nf = Xf.cols();
    step = pow(2,mgLvl);
    Mc = (Mf-1)/step+1;
    Nc = (Nf-1)/step+1;
    Xc->resize(Mc,Nc);
    Yc->resize(Mc,Nc);

    for(int ic=0; ic<Mc; ic++){
        for(int jc=0; jc<Nc; jc++){
            (*Xc)(ic,jc) = Xf(ic*step,jc*step);
            (*Yc)(ic,jc) = Yf(ic*step,jc*step);
        }
    }
}

void PassageMesh::m_interpolate(const Ref<const MatrixXd> Xc, const Ref<const MatrixXd> Yc,
                                Ref<MatrixXd> Xf, Ref<MatrixXd> Yf, int mgLvl) const
{
    int Mc, Nc, Mf, step, i, j;
    Mc = Xc.rows();
    Nc = Xc.cols();
    Mf = Xf.rows();
    step = pow(2,mgLvl);

    // Copy values from coarse to fine grid
    for(int ic=0; ic<Mc; ic++){
        for(int jc=0; jc<Nc; jc++){
            Xf(ic*step,jc*step) = Xc(ic,jc);
            Yf(ic*step,jc*step) = Yc(ic,jc);
        }
    }
    // Interpolate rows
    for(int ic=0; ic<Mc-1; ic++){
        i = ic*step+step/2;
        Xf.row(i) = 0.5*(Xf.row(i+step/2)+Xf.row(i-step/2));
        Yf.row(i) = 0.5*(Yf.row(i+step/2)+Yf.row(i-step/2));
    }
    // Interpolate columns
    for(int jc=0; jc<Nc-1; jc++){
        j = jc*step+step/2;
        Xf.block(1,j,Mf-2,1) = 0.5*(Xf.block(1,j+step/2,Mf-2,1)+
                                    Xf.block(1,j-step/2,Mf-2,1));
        Yf.block(1,j,Mf-2,1) = 0.5*(Yf.block(1,j+step/2,Mf-2,1)+
                                    Yf.block(1,j-step/2,Mf-2,1));
    }
}

void PassageMesh::m_expansionLayer(Ref<MatrixXd> X, Ref<MatrixXd> Y,
                                   const double ds_bld, const double ds_inOut) const
{
    int M=X.rows(), N=X.cols();
    std::vector<double> t(M,0.0), s(M,0.0), si(M), x(M), y(M), xi, yi;

    // determine cell size at inlet and outlet
    double ds_in, ds_out;
    if(ds_inOut != 0.0)
    {
        ds_in  = std::max(ds_inOut*abs(Y(M-1, 0 )-Y(0, 0 ))/(M-1),ds_bld),
        ds_out = std::max(ds_inOut*abs(Y(M-1,N-1)-Y(0,N-1))/(M-1),ds_bld);
    } else
        ds_in = ds_out = ds_bld;

    // expansion rate at inlet and outlet
    double alfa_in  = std::pow(ds_in /ds_bld,1.0/double(m_jLE)),
           alfa_out = std::pow(ds_out/ds_bld,1.0/double(N-1-m_jTE));

    for(int i=0; i<M; i++) t[i] = double(i)/double(M-1);

    for(int j=0; j<N; j++) {
        for(int i=1; i<M; i++)
            s[i] = s[i-1]+sqrt(pow(X(i,j)-X(i-1,j),2.0)+pow(Y(i,j)-Y(i-1,j),2.0));

        double ds;
        if(j<m_jLE)
            ds = ds_in*std::pow(alfa_in,double(-j));
        else if(j<=m_jTE)
            ds = ds_bld;
        else
            ds = ds_bld*std::pow(alfa_out,double(j-m_jTE));

        double alfa = 1.0+(ds/s[M-1]-t[1])*2.0*M_PI/sin(2*M_PI*t[1]);

        for(int i=0; i<M; i++) {
            si[i] = s[M-1]*(t[i]-(1.0-alfa)*sin(2*M_PI*t[i])/M_PI*0.5);
            x[i]  = X(i,j);     y[i]  = Y(i,j);
        }
        xi = mathUtils::interpolation::splineInterp(s,x,si);
        yi = mathUtils::interpolation::splineInterp(s,y,si);

        for(int i=0; i<M; i++) {
            X(i,j) = xi[i];     Y(i,j) = yi[i];
        }
    }
}
}
