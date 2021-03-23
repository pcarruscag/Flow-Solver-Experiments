//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "passageMesh.h"

namespace mesh
{
int PassageMesh::m_computeLayerJacobian(const Ref<const MatrixXd> X,
                                        const Ref<const MatrixXd> Y,
                                        const int layerIndex)
{
    int M = X.rows(), N = X.cols();

    // Mesh metrics
    MatrixXd Xqsi(M,N), Xeta(M,N), Xqsi2(M,N), Xeta2(M,N), Xmix(M,N),
             Yqsi(M,N), Yeta(M,N), Yqsi2(M,N), Yeta2(M,N), Ymix(M,N);
    Xqsi.setZero(); Xeta.setZero(); Xqsi2.setZero(); Xeta2.setZero(); Xmix.setZero();
    Yqsi.setZero(); Yeta.setZero(); Yqsi2.setZero(); Yeta2.setZero(); Ymix.setZero();

    Xqsi.block(0,1,M,N-2) = 0.5*(X.block(0,2,M,N-2)-X.block(0,0,M,N-2));
    Yqsi.block(0,1,M,N-2) = 0.5*(Y.block(0,2,M,N-2)-Y.block(0,0,M,N-2));
    Xeta.block(1,0,M-2,N) = 0.5*(X.block(2,0,M-2,N)-X.block(0,0,M-2,N));
    Yeta.block(1,0,M-2,N) = 0.5*(Y.block(2,0,M-2,N)-Y.block(0,0,M-2,N));

    Xmix.block(1,0,M-2,N) = 0.5*(Xqsi.block(2,0,M-2,N)-Xqsi.block(0,0,M-2,N));
    Ymix.block(1,0,M-2,N) = 0.5*(Yqsi.block(2,0,M-2,N)-Yqsi.block(0,0,M-2,N));

    Xqsi2.block(0,1,M,N-2) = X.block(0,0,M,N-2)-2.0*X.block(0,1,M,N-2)+X.block(0,2,M,N-2);
    Yqsi2.block(0,1,M,N-2) = Y.block(0,0,M,N-2)-2.0*Y.block(0,1,M,N-2)+Y.block(0,2,M,N-2);
    Xeta2.block(1,0,M-2,N) = X.block(0,0,M-2,N)-2.0*X.block(1,0,M-2,N)+X.block(2,0,M-2,N);
    Yeta2.block(1,0,M-2,N) = Y.block(0,0,M-2,N)-2.0*Y.block(1,0,M-2,N)+Y.block(2,0,M-2,N);

    // Coefficients of transformed equation
    MatrixXd A(M,N), B(M,N), C(M,N), D(M,N), E(M,N);
    A.setZero(); B.setZero(); C.setZero(); D.setZero(); E.setZero();

    for(int j=1; j<N-1; j++)
    {
        for(int i=1; i<M-1; i++)
        {
            double J = 1.0/(Xqsi(i,j)*Yeta(i,j)-Xeta(i,j)*Yqsi(i,j));
            A(i,j) = pow(J,2.0)*(pow(Xeta(i,j),2.0)+pow(Yeta(i,j),2.0));
            B(i,j) = -2.0*pow(J,2.0)*(Xqsi(i,j)*Xeta(i,j)+Yqsi(i,j)*Yeta(i,j));
            C(i,j) = pow(J,2.0)*(pow(Xqsi(i,j),2.0)+pow(Yqsi(i,j),2.0));
            double temp1 = A(i,j)*Yqsi2(i,j)+B(i,j)*Ymix(i,j)+C(i,j)*Yeta2(i,j),
                   temp2 = A(i,j)*Xqsi2(i,j)+B(i,j)*Xmix(i,j)+C(i,j)*Xeta2(i,j);
            D(i,j) = J*(temp1*Xeta(i,j)-temp2*Yeta(i,j));
            E(i,j) = J*(temp2*Yqsi(i,j)-temp1*Xqsi(i,j));
        }
    }
    // Global numbering matrix, row and column of coordinate matrices,
    int NN = M*N;
    MatrixXi NUM(M,N);
    VectorXi COL(NN), LIN(NN);
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            int n = i*N+j;
            LIN(n) = i;
            COL(n) = j;
            NUM(i,j) = n;
        }
    }
    // Coefficients of matrix
    std::vector<TripletD> COEF;
    Matrix3d STC;
    for(int n=0; n<NN; n++) {
        int i = LIN(n),
            j = COL(n);
        if(i==0 || i==M-1) {
            COEF.push_back(TripletD(n,n,1.0));
        } else if(j==0 || j==N-1) {
            COEF.push_back(TripletD(n,n,1.0));
            double alfa = (Y(i,j)-Y(0,j))/(Y(M-1,j)-Y(0,j));
            COEF.push_back(TripletD(n,NUM( 0 ,j),alfa-1.0));
            COEF.push_back(TripletD(n,NUM(M-1,j),-alfa));
        } else {
            STC <<     0.25*B(i,j),     C(i,j)-0.5*E(i,j),    -0.25*B(i,j),
                   A(i,j)-0.5*D(i,j), -2.0*(A(i,j)+C(i,j)), A(i,j)+0.5*D(i,j),
                      -0.25*B(i,j),     C(i,j)+0.5*E(i,j),     0.25*B(i,j);

            for(int q=0; q<3; q++)
                for(int p=0; p<3; p++)
                    COEF.push_back(TripletD(n,NUM(i-1+p,j-1+q),STC(p,q)));
        }
    }
    // Matrix of coefficients
    SparseMatrix<double> coeffMat(NN,NN);
    coeffMat.setFromTriplets(COEF.begin(),COEF.end());
    COEF.empty();

    // Factorize matrix
    SparseLU<SparseMatrix<double>,COLAMDOrdering<int> > solver;
    solver.analyzePattern(coeffMat);
    solver.factorize(coeffMat);
    if(solver.info() != Success) return 1;

    // Prepare rhs
    MatrixXd rhs(NN,2*N);
    rhs.setZero();
    for(int j=0; j<N; j++) rhs(NUM( 0 ,j), j ) = 1.0;
    for(int j=0; j<N; j++) rhs(NUM(M-1,j),N+j) = 1.0;

    // Jacobian
    m_layerJacobian[layerIndex].resize(M*N,2*N);
    m_layerJacobian[layerIndex] = solver.solve(rhs);
    if(solver.info() != Success) return 2;

    return 0;
}
}
