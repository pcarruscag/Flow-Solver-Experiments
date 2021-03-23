//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "mathUtilsTests.h"

#include <vector>
#include <iostream>
#include <cassert>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "../src/splines.h"
#include "../src/solvers.h"
#include "../src/matrix.h"
#include "../src/preconditioners.h"

using namespace Eigen;

namespace mathUtils
{
void testSuite()
{
    std::cout << std::endl;
    _splineTests();
    std::cout << std::endl;
    _preconditionerTests();
    std::cout << std::endl;
    _solverTests();
    std::cout << std::endl;
    _utilitiesTests();
}

void _splineTests()
{
    std::cout << "### SPLINE INTERPOLATION TESTS ###" << std::endl;
    std::vector<double> x(5), y(5);
    double xi;

    for (int i=0; i<5; i++)
    {
        x[i]=double(i);
        y[i]=pow(x[i],2.0);
    }

    interpolation::Cspline<double> spline(x,y,1.0);
    y = spline(x);
    assert(y[0]==0.0);
    assert(y[4]==16.0);
    for (int i=10; i<40; i++)
    {
        xi = double(i)/40.0*4.0;
        assert(abs((spline(xi)/(xi*xi)-1.0)*100.0)<1.0);
    }
}

void _preconditionerTests()
{
    std::cout << "### PRECONDITIONER TESTS ###" << std::endl;

    // load a matrix from file
    std::string path("../..//lib/mathUtils/tests/testMatrix.txt");
    std::ifstream file;
    std::ofstream outFile;

    file.open(path.c_str());
    int nnz = 14280, N = 2160;
    std::vector<matrix::Triplet<double> > triplets(nnz);
    for(int i=0; i<nnz; ++i)
    {
        std::string line;
        std::stringstream stream;
        getline(file,line);
        stream.str(line);
        stream >> triplets[i].m_row;
        stream >> triplets[i].m_column;
        stream >> triplets[i].m_value;
    }
    file.close();

    SparseMatrix<double,RowMajor> mat(N,N);
    mat.setFromTriplets(triplets.begin(),triplets.end());
    assert(mat.nonZeros()==nnz);

    VectorXd rhs(N), lhs(N);
    rhs.setOnes();

    preconditioners::ILU0<VectorXd> precond;
    precond.compute(mat);

    path = "../../lib/mathUtils/tests/iluComputeResult.txt";
    outFile.open(path.c_str());
    outFile << precond.m_values << std::endl;
    outFile.close();

    precond.solve(rhs,lhs);

    path = "../../lib/mathUtils/tests/iluSolveResult.txt";
    outFile.open(path.c_str());
    outFile << lhs << std::endl;
    outFile.close();

    struct communication
    {
        int startI, startJ, partI, partJ, length;
    };

    solvers::DistBiCGSTAB<SparseMatrix<double,RowMajor>,VectorXd,communication> solver;

    std::vector<communication> comms;
    std::vector<const VectorXi*> nbIdx_p;
    std::vector<const std::vector<MatrixXd>*> subMats_p;

    std::vector<SparseMatrix<double,RowMajor> > A(1,mat);
    std::vector<VectorXd> b(1,rhs), x(1,rhs);
    std::vector<const preconditioners::PreconditionerBase<SparseMatrix<double,RowMajor>,VectorXd>*> precond_p(1,&precond);

    solver.setData(&A,&b,&x,nbIdx_p,subMats_p,&comms);
    solver.setPreconditioner(precond_p);
    solver.solve(1000,1e-9,true);

    BiCGSTAB<SparseMatrix<double>,IncompleteLUT<double> > solver2;
    solver2.preconditioner().setFillfactor(1);
    solver2.preconditioner().setDroptol(0.0);
    solver2.setTolerance(1e-9);
    solver2.setMaxIterations(1000);

    SparseMatrix<double> mat2(mat);

    solver2.compute(mat2);
    lhs = solver2.solveWithGuess(rhs,rhs);

    assert((x[0]-lhs).norm()<1e-6);
}

void _solverTests()
{
    std::cout << "### LINEAR SOLVER TESTS ###" << std::endl;
    struct communication
    {
        int startI, startJ, partI, partJ, length;
    };

    int numel=100;

    std::vector<matrix::Triplet<float> > triplets;
    triplets.reserve(3*numel);

    triplets.push_back(matrix::Triplet<float>(0,0, 2.0));
    triplets.push_back(matrix::Triplet<float>(0,1,-1.0));
    for(int i=1; i<numel-1; ++i)
    {
        triplets.push_back(matrix::Triplet<float>(i,i-1,-1.0));
        triplets.push_back(matrix::Triplet<float>(i, i , 2.0));
        triplets.push_back(matrix::Triplet<float>(i,i+1,-1.0));
    }
    triplets.push_back(matrix::Triplet<float>(numel-1,numel-1, 2.0));
    triplets.push_back(matrix::Triplet<float>(numel-1,numel-2,-1.0));

    std::vector<SparseMatrix<float,RowMajor> > matrices(2);
    matrices[0].resize(numel,numel);
    matrices[0].setFromTriplets(triplets.begin(),triplets.end());
    matrices[1] = matrices[0].eval();

    std::vector<VectorXf> sources(2), x(2);
    sources[0].setOnes(numel);
    sources[1] = sources[0].eval();
    x[0].setZero(numel);
    x[1] = x[0].eval();

    std::vector<communication> communications(1);
    communications[0].startI = 0;
    communications[0].startJ = 0;
    communications[0].partI = 0;
    communications[0].partJ = 1;
    communications[0].length = 1;

    std::vector<VectorXi> neighbourIndex(2,VectorXi(1));
    neighbourIndex[0] << numel-1;
    neighbourIndex[1] << 0;
    std::vector<const VectorXi *> neighbourIndex_p(2);
    neighbourIndex_p[0] = &neighbourIndex[0];
    neighbourIndex_p[1] = &neighbourIndex[1];

    std::vector<std::vector<MatrixXf> > subMatrices(2,std::vector<MatrixXf>(1,MatrixXf(1,1)));
    subMatrices[0][0] << -1.0;
    subMatrices[1][0] << -1.0;
    std::vector<const std::vector<MatrixXf>*> subMatrices_p(2);
    subMatrices_p[0] = &subMatrices[0];
    subMatrices_p[1] = &subMatrices[1];

    solvers::DistBiCGSTAB<SparseMatrix<float,RowMajor>,VectorXf,communication> solver;
    solver.setData(&matrices,&sources,&x,neighbourIndex_p,subMatrices_p,&communications);
    #pragma omp parallel num_threads(2)
    solver.solve(1000,1e-6,true);
    assert(std::abs(x[0](0)-100)<1e-2f);
    assert(std::abs(x[0](numel-1)-5050)<1e-1f);

    x[0].setZero(); x[1] = x[0].eval();
    preconditioners::Diagonal<SparseMatrix<float,RowMajor>,VectorXf> precond1, precond2;
    precond1.compute(matrices[0]);
    precond2.compute(matrices[1]);
    std::vector<const preconditioners::PreconditionerBase<SparseMatrix<float,RowMajor>,VectorXf>*> precond_p(2);
    precond_p[0] = &precond1;
    precond_p[1] = &precond2;
    solver.setData(&matrices,&sources,&x,neighbourIndex_p,subMatrices_p,&communications);
    solver.setPreconditioner(precond_p);
    #pragma omp parallel num_threads(2)
    solver.solve(1000,1e-6,true);
    assert(std::abs(x[0](0)-100)<1e-2f);
    assert(std::abs(x[0](numel-1)-5050)<1e-1f);

    x[0].setZero(); x[1] = x[0].eval();
    preconditioners::ILU0<VectorXf> precond3, precond4;
    precond3.compute(matrices[0]);
    precond4.compute(matrices[1]);
    precond_p[0] = &precond3;
    precond_p[1] = &precond4;
    solver.setData(&matrices,&sources,&x,neighbourIndex_p,subMatrices_p,&communications);
    solver.setPreconditioner(precond_p);
    #pragma omp parallel num_threads(2)
    solver.solve(1000,1e-6,true);
    assert(std::abs(x[0](0)-100)<1e-2f);
    assert(std::abs(x[0](numel-1)-5050)<1e-1f);
}

void _utilitiesTests()
{
    std::cout << "### UTILITIES TESTS ###" << std::endl;

    using matrix::Triplet;
    using matrix::JoinedIterator;

    std::vector<Triplet<float> > v1(3), v2;
    for(int i=0; i<3; ++i)
        v1[i] =  Triplet<float>(i,i,float(i+1));
    v2.push_back(Triplet<float>(0,1,-1.0f));
    v2.push_back(Triplet<float>(2,2,1.0f));

    std::vector<std::vector<Triplet<float> > const*> v_ptr;
    v_ptr.push_back(&v1);
    v_ptr.push_back(&v2);

    SparseMatrix<float,ColMajor> A(3,3);
    A.setFromTriplets(JoinedIterator<Triplet<float> >(v_ptr),
                      JoinedIterator<Triplet<float> >(v_ptr,1));

    assert(A.coeff(2,2)== 4.0f);
    assert(A.coeff(0,1)==-1.0f);
}

int _getTripletsFromFile(const std::string& filepath, std::vector<matrix::Triplet<float> >& triplets)
{
    int N, Ntrip1, Ntrip2;
    char data[4]; // 4 bytes to read ints and floats from file

    std::cout << "Loading matrix from file..." << std::endl;
    double t0 = -omp_get_wtime();

    std::ifstream file; file.open(filepath.c_str(),std::ios::binary);

    #define GETINT(p)   file.read(data,4); p=*reinterpret_cast<int*>(data)
    #define GETFLOAT(p) file.read(data,4); p=*reinterpret_cast<float*>(data)

    GETINT(N);
    std::cout << "  Number of cells:    " << N << std::endl;
    N *= 4; // matrix size

    GETINT(Ntrip1);  GETINT(Ntrip2);
    std::cout << "  Number of triplets: " << Ntrip1 << "\t" << Ntrip2 << std::endl;
    Ntrip1 += Ntrip2;

    triplets.reserve(Ntrip1);
    for(int i=0; i<Ntrip1; ++i)
    {
        int row, col; float val;
        GETINT(row); GETINT(col); GETFLOAT(val);
        triplets.push_back(matrix::Triplet<float>(row,col,val));
    }
    file.close();

    t0 += omp_get_wtime();
    std::cout << "... read " << Ntrip1 << " triplets in " << t0 << "s" << std::endl;

    #undef GETINT
    #undef GETFLOAT

    return N;
}
}
