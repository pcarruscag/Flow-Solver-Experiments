//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "passageMesh.h"

#include "../../mathUtils/src/splines.h"
#include "../../mathUtils/src/solvers.h"
#include "../../mathUtils/src/preconditioners.h"
#include <math.h>


namespace mesh
{
int PassageMesh::m_meshLayer(Ref<MatrixXd> X, Ref<MatrixXd> Y, const int jLE, const int jTE, const int layer,
                             const fileManagement::MeshParamManager &meshParams, const bool verbose) const
{
    using std::pow;
    using std::sqrt;

    // Matrix size
    int M=X.rows(), N=X.cols();
    // Number of equations
    int NN = (M-2)*(N-2);

    // Global numbering matrix, row and column of coordinate matrices,
    MatrixXi NUM;
    NUM = MatrixXi::Constant(M,N,-1); //default value of -1 since indexes start at 0
    VectorXi COL(NN), LIN(NN);
    for(int i=0; i<M-2; i++){
        for(int j=0; j<N-2; j++){
            int n = i*(N-2)+j;
            LIN(n) = i+1;
            COL(n) = j+1;
            NUM(i+1,j+1) = n;
        }
    }
    // Indexes of the coefficients
    std::vector<TripletD> COEF;
    for(int n=0; n<NN; n++)
        for(int j=COL(n)-1; j<=COL(n)+1; j++)
            for(int i=LIN(n)-1; i<=LIN(n)+1; i++)
                if(NUM(i,j)!=-1)
                    COEF.push_back(TripletD(n,NUM(i,j)));

    // First cell thickness
    MatrixXd Seta_1(1,N), Seta_M(1,N);
    Seta_1 = (Y.row(M-1)-Y.row(0))/double(M-1);
    Seta_M = Seta_1;

    // Orthogonality angles at lower/upper boundary
    MatrixXd teta_1(1,N), teta_M(1,N);
    std::vector<double> orthogCPj;
    {
        std::vector<double> x_ort(4), y_ort(4), xi_ort(4);
        x_ort[0] = 0.0; y_ort[0] = 0.0;
        x_ort[1] = 1.0; y_ort[1] = double(jLE);
        x_ort[2] = 2.0; y_ort[2] = double(jTE);
        x_ort[3] = 3.0; y_ort[3] = double(N-1);
        for(int i=0; i<4; i++) xi_ort[i] = meshParams.m_layerPar.orthogCtrl[layer].CP[i];
        orthogCPj = mathUtils::interpolation::pwiseLinInterp<double>(x_ort,y_ort,xi_ort);
    }
    orthogCPj.push_back(double(N-1));
    orthogCPj.insert(orthogCPj.begin(),0.0);
    {
        std::vector<double> y_ort(6,0.0), xi_ort(N), teta_temp(N);
        y_ort[2] = meshParams.m_layerPar.orthogCtrl[layer].angle;
        y_ort[3] = y_ort[2];
        for(int i=0; i<N; i++) xi_ort[i] = double(i);
        // Lower
        double alfaLE = atan((Y(0,jLE)-Y(0, 0 ))/((X(0,jLE)-X(0, 0 )))),
               alfaTE = atan((Y(0,N-1)-Y(0,jTE))/((X(0,N-1)-X(0,jTE))));
        y_ort[0] = alfaLE; y_ort[1] = alfaLE; y_ort[4] = alfaTE; y_ort[5] = alfaTE;
        teta_temp = mathUtils::interpolation::pwiseLinInterp(orthogCPj,y_ort,xi_ort);
        for(int i=0; i<N; i++) teta_1(0,i) = M_PI_2+teta_temp[i];
        // Upper
        alfaLE = atan((Y(M-1,jLE)-Y(M-1,0))/((X(M-1,jLE)-X(M-1,0)))),
        alfaTE = atan((Y(M-1,N-1)-Y(M-1,jTE))/((X(M-1,N-1)-X(M-1,jTE))));
        y_ort[0] = alfaLE; y_ort[1] = alfaLE; y_ort[4] = alfaTE; y_ort[5] = alfaTE;
        teta_temp = mathUtils::interpolation::pwiseLinInterp(orthogCPj,y_ort,xi_ort);
        for(int i=0; i<N; i++) teta_M(0,i) = M_PI_2+teta_temp[i];
    }
    // Qsi derivatives at upper and lower boundary
    MatrixXd Xqsi_1(1,N), Xqsi_M(1,N), Xqsi2_1(1,N), Xqsi2_M(1,N),
             Yqsi_1(1,N), Yqsi_M(1,N), Yqsi2_1(1,N), Yqsi2_M(1,N);
    Xqsi2_1.setZero(); Xqsi2_M.setZero(); Yqsi2_1.setZero(); Yqsi2_M.setZero();

    Xqsi_1(0,0) = X( 0 ,1)-X( 0 ,0);  Xqsi_1(0,N-1) = X( 0 ,N-1)-X( 0 ,N-2);
    Xqsi_1.block(0,1,1,N-2) = 0.5*(X.block( 0 ,2,1,N-2)-X.block( 0 ,0,1,N-2));

    Xqsi_M(0,0) = X(M-1,1)-X(M-1,0);  Xqsi_M(0,N-1) = X(M-1,N-1)-X(M-1,N-2);
    Xqsi_M.block(0,1,1,N-2) = 0.5*(X.block(M-1,2,1,N-2)-X.block(M-1,0,1,N-2));

    Xqsi2_1.block(0,1,1,N-2) = X.block( 0 ,0,1,N-2)-2.0*X.block( 0 ,1,1,N-2)+X.block( 0 ,2,1,N-2);
    Xqsi2_M.block(0,1,1,N-2) = X.block(M-1,0,1,N-2)-2.0*X.block(M-1,1,1,N-2)+X.block(M-1,2,1,N-2);

    Yqsi_1(0,0) = Y( 0 ,1)-Y( 0 ,0);  Yqsi_1(0,N-1) = Y( 0 ,N-1)-Y( 0 ,N-2);
    Yqsi_1.block(0,1,1,N-2) = 0.5*(Y.block( 0 ,2,1,N-2)-Y.block( 0 ,0,1,N-2));

    Yqsi_M(0,0) = Y(M-1,1)-Y(M-1,0);  Yqsi_M(0,N-1) = Y(M-1,N-1)-Y(M-1,N-2);
    Yqsi_M.block(0,1,1,N-2) = 0.5*(Y.block(M-1,2,1,N-2)-Y.block(M-1,0,1,N-2));

    Yqsi2_1.block(0,1,1,N-2) = Y.block( 0 ,0,1,N-2)-2.0*Y.block( 0 ,1,1,N-2)+Y.block( 0 ,2,1,N-2);
    Yqsi2_M.block(0,1,1,N-2) = Y.block(M-1,0,1,N-2)-2.0*Y.block(M-1,1,1,N-2)+Y.block(M-1,2,1,N-2);

    // alpha, beta, gamma coefficients and eta derivatives
    MatrixXd ALFA_1(1,N), BETA_1(1,N), GAMA_1(1,N), Xeta_1(1,N), Yeta_1(1,N),
             ALFA_M(1,N), BETA_M(1,N), GAMA_M(1,N), Xeta_M(1,N), Yeta_M(1,N);
    for(int i=0; i<N; i++)
    {
        ALFA_1(0,i) = pow(Seta_1(0,i),2.0);
        GAMA_1(0,i) = pow(Xqsi_1(0,i),2.0)+pow(Yqsi_1(0,i),2.0);
        BETA_1(0,i) =-sqrt(ALFA_1(0,i)*GAMA_1(0,i))*cos(teta_1(0,i));
        Xeta_1(0,i) =-Seta_1(0,i)*(Xqsi_1(0,i)*cos(teta_1(0,i))+Yqsi_1(0,i)*sin(teta_1(0,i)))/sqrt(GAMA_1(0,i));
        Yeta_1(0,i) = Seta_1(0,i)*(Xqsi_1(0,i)*sin(teta_1(0,i))-Yqsi_1(0,i)*cos(teta_1(0,i)))/sqrt(GAMA_1(0,i));
        ALFA_M(0,i) = pow(Seta_M(0,i),2.0);
        GAMA_M(0,i) = pow(Xqsi_M(0,i),2.0)+pow(Yqsi_M(0,i),2.0);
        BETA_M(0,i) =-sqrt(ALFA_M(0,i)*GAMA_M(0,i))*cos(teta_M(0,i));
        Xeta_M(0,i) =-Seta_M(0,i)*(Xqsi_M(0,i)*cos(teta_M(0,i))+Yqsi_M(0,i)*sin(teta_M(0,i)))/sqrt(GAMA_M(0,i));
        Yeta_M(0,i) = Seta_M(0,i)*(Xqsi_M(0,i)*sin(teta_M(0,i))-Yqsi_M(0,i)*cos(teta_M(0,i)))/sqrt(GAMA_M(0,i));
    }

    // Eta and mixed derivatives at upper and lower boundary
    MatrixXd Xmix_1(1,N), Xmix_M(1,N), Ymix_1(1,N), Ymix_M(1,N);
    Xmix_1(0, 0 ) = 0.0; Xmix_M(0, 0 ) = 0.0; Ymix_1(0, 0 ) = 0.0; Ymix_M(0, 0 ) = 0.0;
    Xmix_1(0,N-1) = 0.0; Xmix_M(0,N-1) = 0.0; Ymix_1(0,N-1) = 0.0; Ymix_M(0,N-1) = 0.0;
    Xmix_1.block(0,1,1,N-2) = 0.5*(Xeta_1.block(0,2,1,N-2)-Xeta_1.block(0,0,1,N-2));
    Xmix_M.block(0,1,1,N-2) = 0.5*(Xeta_M.block(0,2,1,N-2)-Xeta_M.block(0,0,1,N-2));
    Ymix_1.block(0,1,1,N-2) = 0.5*(Yeta_1.block(0,2,1,N-2)-Yeta_1.block(0,0,1,N-2));
    Ymix_M.block(0,1,1,N-2) = 0.5*(Yeta_M.block(0,2,1,N-2)-Yeta_M.block(0,0,1,N-2));

    // Jacobian
    MatrixXd J_1(1,N), J_M(1,N);
    for(int i=0; i<N; i++)
    {
        J_1(0,i) = 1.0/(Xqsi_1(0,i)*Yeta_1(0,i)-Xeta_1(0,i)*Yqsi_1(0,i));
        J_M(0,i) = 1.0/(Xqsi_M(0,i)*Yeta_M(0,i)-Xeta_M(0,i)*Yqsi_M(0,i));
    }

    // Matrices for mesh metrics
    MatrixXd Xqsi, Yqsi, Xeta, Yeta, J, ALFA, BETA, GAMA,
             Xeta2_1(1,N), Yeta2_1(1,N), R1_1(1,N), R2_1(1,N), P_1(1,N), Q_1(1,N),
             Xeta2_M(1,N), Yeta2_M(1,N), R1_M(1,N), R2_M(1,N), P_M(1,N), Q_M(1,N);
    Xqsi = MatrixXd::Zero(M,N); Yqsi = MatrixXd::Zero(M,N);
    Xeta = MatrixXd::Zero(M,N); Yeta = MatrixXd::Zero(M,N);
    J    = MatrixXd::Zero(M,N); ALFA = MatrixXd::Zero(M,N);
    BETA = MatrixXd::Zero(M,N); GAMA = MatrixXd::Zero(M,N);

    // coordinate, sorce term vectors and coefficient matrix
    MatrixX2d sol(NN,2), src(NN,2);
    SparseMatrix<double,RowMajor> A(NN,NN);
    for(int n=0; n<NN; n++) {
        sol(n,0) = X(LIN(n),COL(n));
        sol(n,1) = Y(LIN(n),COL(n));
    }

    // Setup solver and preconditioner
    mathUtils::solvers::BiCGSTAB<SparseMatrix<double,RowMajor>,MatrixX2d> solver;
    mathUtils::preconditioners::ILU0<MatrixX2d> prec;
    solver.setData(A, src, sol);
    solver.setPreconditioner(&prec);

    // Iterations
    Array2d r, r0;
    int k;
    for(k=0; k<meshParams.m_advPar.iterMax; k++)
    {
        // Relaxation factor for current iteration
        double rel = std::min(meshParams.m_advPar.relax1,meshParams.m_advPar.relax0+
                   double(k)*(meshParams.m_advPar.relax1-meshParams.m_advPar.relax0)/
                   double(meshParams.m_advPar.iter1));
        // Metrics of the transformation and coefficients of the PDE
        Xqsi.block(0,1,M,N-2) = 0.5*(X.block(0,2,M,N-2)-X.block(0,0,M,N-2));
        Yqsi.block(0,1,M,N-2) = 0.5*(Y.block(0,2,M,N-2)-Y.block(0,0,M,N-2));
        Xeta.block(1,0,M-2,N) = 0.5*(X.block(2,0,M-2,N)-X.block(0,0,M-2,N));
        Yeta.block(1,0,M-2,N) = 0.5*(Y.block(2,0,M-2,N)-Y.block(0,0,M-2,N));
        for(int i=1; i<M-1; i++){
            for(int j=1; j<N-1; j++){
                J(i,j) = 1.0/(Xqsi(i,j)*Yeta(i,j)-Xeta(i,j)*Yqsi(i,j));
                ALFA(i,j) = pow(Xeta(i,j),2.0)+pow(Yeta(i,j),2.0);
                BETA(i,j) = Xqsi(i,j)*Xeta(i,j)+Yqsi(i,j)*Yeta(i,j);
                GAMA(i,j) = pow(Xqsi(i,j),2.0)+pow(Yqsi(i,j),2.0);
            }
        }
        // Second eta derivatives at the upper and lower boundaries
        Xeta2_1 =-1.5*Xeta_1-1.25*X.row( 0 )+X.row( 1 )+0.25*X.row( 2 );
        Xeta2_M = 1.5*Xeta_M-1.25*X.row(M-1)+X.row(M-2)+0.25*X.row(M-3);
        Yeta2_1 =-1.5*Yeta_1-1.25*Y.row( 0 )+Y.row( 1 )+0.25*Y.row( 2 );
        Yeta2_M = 1.5*Yeta_M-1.25*Y.row(M-1)+Y.row(M-2)+0.25*Y.row(M-3);

        // Source term coefficients
        for(int i=0; i<N; i++)
        {
            R1_1(0,i) =-pow(J_1(0,i),2.0)*(ALFA_1(0,i)*Xqsi2_1(0,i)-
                        2.0*BETA_1(0,i)*Xmix_1(0,i)+GAMA_1(0,i)*Xeta2_1(0,i));
            R1_M(0,i) =-pow(J_M(0,i),2.0)*(ALFA_M(0,i)*Xqsi2_M(0,i)-
                        2.0*BETA_M(0,i)*Xmix_M(0,i)+GAMA_M(0,i)*Xeta2_M(0,i));
            R2_1(0,i) =-pow(J_1(0,i),2.0)*(ALFA_1(0,i)*Yqsi2_1(0,i)-
                        2.0*BETA_1(0,i)*Ymix_1(0,i)+GAMA_1(0,i)*Yeta2_1(0,i));
            R2_M(0,i) =-pow(J_M(0,i),2.0)*(ALFA_M(0,i)*Yqsi2_M(0,i)-
                        2.0*BETA_M(0,i)*Ymix_M(0,i)+GAMA_M(0,i)*Yeta2_M(0,i));
            P_1(0,i)  = J_1(0,i)*(Yeta_1(0,i)*R1_1(0,i)-Xeta_1(0,i)*R2_1(0,i));
            P_M(0,i)  = J_M(0,i)*(Yeta_M(0,i)*R1_M(0,i)-Xeta_M(0,i)*R2_M(0,i));
            Q_1(0,i)  = J_1(0,i)*(Xqsi_1(0,i)*R2_1(0,i)-Yqsi_1(0,i)*R1_1(0,i));
            Q_M(0,i)  = J_M(0,i)*(Xqsi_M(0,i)*R2_M(0,i)-Yqsi_M(0,i)*R1_M(0,i));
        }
        // Matrix of coefficients and vectors of source terms
        m_meshLayer_fillStencil(X,Y,Xqsi,Xeta,Yqsi,Yeta,J,ALFA,BETA,GAMA,
            P_1,P_M,Q_1,Q_M,LIN,COL,NUM,meshParams.m_advPar.decayFactor,
            meshParams.m_advPar.disableOrthogAlfa,meshParams.m_advPar.disableOrthogFraction,
            meshParams.m_layerPar.orthogCtrl[layer].weight,jLE,jTE,COEF,src);
        A.setFromTriplets(COEF.begin(),COEF.end());

        // Solve equations
        r = (src-A*sol).colwise().norm();
        prec.compute(A);
        solver.solve(meshParams.m_advPar.linSolItMax, meshParams.m_advPar.linSolTol);

        // Update coordinates
        for(int n=0; n<NN; n++) {
            X(LIN(n),COL(n)) += rel*(sol(n,0)-X(LIN(n),COL(n)));
            Y(LIN(n),COL(n)) += rel*(sol(n,1)-Y(LIN(n),COL(n)));
        }
        // Check convergence
        if(k==0) r0 = r;
        r /= r0;
        #ifdef MESH_VERBOSE
        if(verbose)
            std::cout << "  X: " << r(0) << "\tY: " << r(1) << std::endl;
        #endif

        if((r<meshParams.m_advPar.tol).all() && k>meshParams.m_advPar.iter1)
            break;
    }
    if(k==meshParams.m_advPar.iterMax)
        return 1;
    else
        return 0;
}

void PassageMesh::m_meshLayer_fillStencil(const Ref<const MatrixXd> X,    const Ref<const MatrixXd> Y,
          const Ref<const MatrixXd> Xqsi, const Ref<const MatrixXd> Xeta, const Ref<const MatrixXd> Yqsi,
          const Ref<const MatrixXd> Yeta, const Ref<const MatrixXd> J,    const Ref<const MatrixXd> ALFA,
          const Ref<const MatrixXd> BETA, const Ref<const MatrixXd> GAMA, const Ref<const MatrixXd> P_1,
          const Ref<const MatrixXd> P_M,  const Ref<const MatrixXd> Q_1,  const Ref<const MatrixXd> Q_M,
          const Ref<const VectorXi> LIN,  const Ref<const VectorXi> COL,  const Ref<const MatrixXi> NUM,
          const double decayFactor,       const double disableOrthogAlfa, const double disableOrthogFraction,
          const double weight,            const int jLE,                  const int jTE,
          std::vector<TripletD> &COEF,    Ref<MatrixX2d> src) const
{
    using std::atan; using std::exp; using std::pow; using std::tanh;

    int M = X.rows(), N = X.cols(), NN = LIN.rows(), i, j, k=0;
    src.setZero();
    Matrix3d STC;
    double P, Q, lowWeight, upWeight;

    double leTeDecay = -2.3/pow(disableOrthogFraction*(jTE-jLE),2.0),
           alfaLE = atan((Y(0,jLE)-Y(0, 0 )+Y(M-1,jLE)-Y(M-1, 0 ))/
                         (X(0,jLE)-X(0, 0 )+X(M-1,jLE)-X(M-1, 0 ))),
           alfaTE = atan((Y(0,N-1)-Y(0,jTE)+Y(M-1,N-1)-Y(M-1,jTE))/
                         (X(0,N-1)-X(0,jTE)+X(M-1,N-1)-X(M-1,jTE)));

    double leLowSwitch = 0.5*(1.0+tanh(180.0/M_PI*(alfaLE-disableOrthogAlfa)+2.0)),
           leUpSwitch  = 0.5*(1.0-tanh(180.0/M_PI*(alfaLE+disableOrthogAlfa)-2.0)),
           teLowSwitch = 0.5*(1.0-tanh(180.0/M_PI*(alfaTE+disableOrthogAlfa)-2.0)),
           teUpSwitch  = 0.5*(1.0+tanh(180.0/M_PI*(alfaTE-disableOrthogAlfa)+2.0));

    for(int n=0; n<NN; n++){
        i = LIN(n); j = COL(n);
        STC << -BETA(i,j)*0.5,       GAMA(i,j),        BETA(i,j)*0.5,
                ALFA(i,j), -2.0*(ALFA(i,j)+GAMA(i,j)), ALFA(i,j),
                BETA(i,j)*0.5,       GAMA(i,j),       -BETA(i,j)*0.5;
        for(int q=0; q<3; q++){
            for(int p=0; p<3; p++){
                if(NUM(i-1+p,j-1+q)!=-1){
                    COEF[k].m_value = STC(p,q);
                    k++;
                } else{
                    src(n,0) -= STC(p,q)*X(i-1+p,j-1+q);
                    src(n,1) -= STC(p,q)*Y(i-1+p,j-1+q);
                }
            }
        }
        lowWeight = weight*
                    (1.0-leLowSwitch*exp(leTeDecay*pow(j-jLE,2)))*
                    (1.0-teLowSwitch*exp(leTeDecay*pow(j-jTE,2)));
        upWeight  = weight*
                    (1.0-leUpSwitch*exp(leTeDecay*pow(j-jLE,2)))*
                    (1.0-teUpSwitch*exp(leTeDecay*pow(j-jTE,2)));
        P = P_1(0,j)*exp(-decayFactor*double(  i  )/double(M-1))*lowWeight+
            P_M(0,j)*exp(-decayFactor*double(M-1-i)/double(M-1))*upWeight;
        Q = Q_1(0,j)*exp(-decayFactor*double(  i  )/double(M-1))*lowWeight+
            Q_M(0,j)*exp(-decayFactor*double(M-1-i)/double(M-1))*upWeight;
        src(n,0) -= (P*Xqsi(i,j)+Q*Xeta(i,j))/pow(J(i,j),2.0);
        src(n,1) -= (P*Yqsi(i,j)+Q*Yeta(i,j))/pow(J(i,j),2.0);
    }
}
}
