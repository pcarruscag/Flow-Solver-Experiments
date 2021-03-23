//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "adjoint.h"

#include "../../mathUtils/src/matrix.h"
#include "../../mathUtils/src/gradients.h"
#include "../../mesh/src/geometricProperties.h"
#include <unsupported/Eigen/AutoDiff>

//#define PROBLEMCELL -1
// how many derivatives we compute at once in fwd mode AD
#define VECTORSIZE 136
#define FIXEDPOINTITER 20

using std::vector;
using std::pair;
using namespace mathUtils::matrix;

namespace adjoint
{
// passive scalar and vector types
typedef float PassiveScalar;
typedef Matrix<PassiveScalar,Dynamic,Dynamic> MatrixX;
typedef Matrix<PassiveScalar,Dynamic,3> MatrixX3;
typedef Matrix<PassiveScalar,Dynamic,1> VectorX;
typedef Matrix<PassiveScalar,3,1> RowVector3;
// and their active counterparts
typedef AutoDiffScalar<Matrix<PassiveScalar,VECTORSIZE,1> > ActiveScalar;
typedef Matrix<ActiveScalar,Dynamic,Dynamic> MatrixXact;
typedef Matrix<ActiveScalar,Dynamic,3> MatrixX3act;
typedef Matrix<ActiveScalar,Dynamic,1> VectorXact;
typedef Matrix<ActiveScalar,3,1> RowVector3act;

// this is a mechanism to change variable types depending on the type of
// variables we are differentiating w.r.t. 2 versions of the function will
// eventually be generated, one for primal variables the other for geometric
// ones (i.e. parameters)
template<PressureBasedCoupledSolverAdjoint::VariableType T> struct types {};

template<> struct types<PressureBasedCoupledSolverAdjoint::FLOW>
{
    using VecXflw    = VectorXact;
    using ScalarGeo  = PassiveScalar;
    using MatXgeo    = MatrixX;
    using MatX3geo   = MatrixX3;
    using VecXgeo    = VectorX;
    using RowVec3geo = RowVector3;
};
template<> struct types<PressureBasedCoupledSolverAdjoint::GEOMETRY>
{
    using VecXflw    = VectorX;
    using ScalarGeo  = ActiveScalar;
    using MatXgeo    = MatrixXact;
    using MatX3geo   = MatrixX3act;
    using VecXgeo    = VectorXact;
    using RowVec3geo = RowVector3act;
};

// template function to seed flow or geometric variables for differentiation
// this needs specialization because passive types do not have the method "derivatives"
template<class TFlw, class TGeo>
inline void seedVariables(int, int, int, TFlw&, TFlw&, TFlw&, TFlw&, TGeo&);

template<>
inline void seedVariables<VectorXact,MatrixX3>(int round, int Nc, int, VectorXact& u, VectorXact& v,
                                               VectorXact& w, VectorXact& p, MatrixX3&)
{
    for(int i=0; i<VECTORSIZE; )
    {
        // clear previous seeds
        int outerIdx = (VECTORSIZE*(round-1)+i)/4;

        if(round>0) {
        u(outerIdx).derivatives()(i  ) = PassiveScalar(0);
        v(outerIdx).derivatives()(i+1) = PassiveScalar(0);
        w(outerIdx).derivatives()(i+2) = PassiveScalar(0);
        p(outerIdx).derivatives()(i+3) = PassiveScalar(0);
        }
        // seed new variables
        outerIdx = (VECTORSIZE*round+i)/4;

        if(outerIdx==Nc) break;
        u(outerIdx).derivatives()(i++) = PassiveScalar(1);
        v(outerIdx).derivatives()(i++) = PassiveScalar(1);
        w(outerIdx).derivatives()(i++) = PassiveScalar(1);
        p(outerIdx).derivatives()(i++) = PassiveScalar(1);
    }
}

template<>
inline void seedVariables<VectorX,MatrixX3act>(int round, int, int Nv, VectorX&, VectorX&,
                                               VectorX&, VectorX&, MatrixX3act& x)
{
    for(int i=0; i<VECTORSIZE; ++i)
    {
        // clear previous seeds
        int outerIdx = (VECTORSIZE*(round-1)+i)/3,
            innerIdx = (VECTORSIZE*(round-1)+i)%3;

        if(round>0)
        x(outerIdx,innerIdx).derivatives()(i) = PassiveScalar(0);

        // seed new variables
        outerIdx = (VECTORSIZE*round+i)/3;
        innerIdx = (VECTORSIZE*round+i)%3;

        if(outerIdx==Nv) break;
        x(outerIdx,innerIdx).derivatives()(i) = PassiveScalar(1);
    }
}

// the main event, compute the residual (u,v,w,p) for a single cell and differentiate
// w.r.t. either the state (FLOW / u,v,w,p) or the control (GEOMETRY / x,y,z) variables
template<PressureBasedCoupledSolverAdjoint::VariableType T>
int PressureBasedCoupledSolverAdjoint::m_computeResidualForCell(
    const int cell,     Vector4f& residual,         Vector4f& residualNorm,
    vector<int>& cells, vector<int>& cellDegEndIdx, vector<int>& vertices,
    Matrix<numPrec,Dynamic,4>& deriv) const
{
    using mathUtils::gradients::TypeBasedBoundaryIndicator;

    using VecXflw    = typename types<T>::VecXflw;
    using ScalarGeo  = typename types<T>::ScalarGeo;
    using MatXgeo    = typename types<T>::MatXgeo;
    using MatX3geo   = typename types<T>::MatX3geo;
    using VecXgeo    = typename types<T>::VecXgeo;
    using RowVec3geo = typename types<T>::RowVec3geo;

    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Build local domain" << std::endl;
    #endif
    // Obtain the relevant neighbours for the cell
    int maxDeg = 2;
    vector<int> faces, faceDegEndIdx;

    m_gatherNeighbourhood(cell,maxDeg,vertices,faces,faceDegEndIdx,cells,cellDegEndIdx);
    int Nv = vertices.size(),
        Nf = faces.size(),
        Nf_inner = faceDegEndIdx[maxDeg-1],
        Nc = cells.size();

    // ************************ BUILD LOCAL DOMAIN ************************* //
    vector<int> verticesStart, verticesIndex;
    vector<pair<int,int> > connectivity;
    for(int face_l=0; face_l<Nf; ++face_l)
    {
        int face_g = faces[face_l];

        // face definition
        verticesStart.push_back(verticesIndex.size());
        for(int i=m_inData_p->verticesStart[face_g];
                i<m_inData_p->verticesStart[face_g+1]; ++i)
        {
            int vertex_g = m_inData_p->verticesIndex[i];
            int vertex_l = findInVector(vertices,vertex_g);
            assert(vertex_l!=-1 && "could not map a vertex");
            verticesIndex.push_back(vertex_l);
        }

        // connectivity
        if(face_l < Nf_inner)
        {
            int C0_g = m_inData_p->connectivity[face_g].first;
            int C0_l = findInVector(cells,C0_g);
            if(C0_l == -1)
            {
                std::cout << "Error building local domain for cell " << cell << std::endl;
                std::cout << "Inner face connectivity for face:" << std::endl;
                std::cout << "Local idx: " << face_l << " Global idx: " << face_g << std::endl;
                std::cout << "Could not find cell " << C0_g << std::endl;
                assert(false);
            }
            int C1_g = m_inData_p->connectivity[face_g].second, C1_l;
            if(C1_g == -1)
                C1_l = -1;
            else
            {
                C1_l = findInVector(cells,C1_g);
                if(C1_l == -1)
                {
                std::cout << "Error building local domain for cell " << cell << std::endl;
                std::cout << "Inner face connectivity for face:" << std::endl;
                std::cout << "Local idx: " << face_l << " Global idx: " << face_g << std::endl;
                std::cout << "Could not find cell " << C1_g << std::endl;
                assert(false);
                }
            }
            connectivity.push_back(std::make_pair(C0_l,C1_l));
        }
        else
        {
            int C0_g = m_inData_p->connectivity[face_g].first;
            int C1_g = m_inData_p->connectivity[face_g].second;

            int C0_l = findInVector(cells,C0_g);
            if(C0_l == -1)
                C0_l = findInVector(cells,C1_g);
            if(C0_l == -1)
            {
                std::cout << "Error building local domain for cell " << cell << std::endl;
                std::cout << "Outer face connectivity for face:" << std::endl;
                std::cout << "Local idx: " << face_l << " Global idx: " << face_g << std::endl;
                if(face_g < m_inData_p->boundaryNumber)
                    std::cout << "Boundary type: " << m_inData_p->boundaryType(face_g) << std::endl;
                std::cout << "Could not find cell " << C0_g << " or " << C1_g << std::endl;
                assert(false);
            }
            connectivity.push_back(std::make_pair(C0_l,-1));
        }
    }
    verticesStart.push_back(verticesIndex.size());

    ArrayXi  boundaryType(Nf);
    ArrayXXi boundaryConditions_int(Nf,6);
    MatXgeo  boundaryConditions_bad(Nf,6);
    for(int face_l=0; face_l<Nf; ++face_l)
    {
        int face_g = faces[face_l];
        if(face_g < m_inData_p->boundaryNumber)
        {
            boundaryType(face_l) = m_inData_p->boundaryType(face_g);
            for(int i=0; i<6; ++i)
            {
                boundaryConditions_bad(face_l,i) = m_inData_p->boundaryConditions(face_g,i);
                boundaryConditions_int(face_l,i) = m_inData_p->boundaryConditions(face_g,i);
            }
            if(boundaryType(face_l)==BoundaryType::PERIODIC)
            {
                int C1_l = findInVector(cells,boundaryConditions_int(face_l,0));
                int Fj_l = findInVector(faces,boundaryConditions_int(face_l,4));

                // if outer face it may or may not be needed
                if(face_l >= Nf_inner)
                {
                    // if the cell is found so must be the face and v.v.
                    if(((C1_l != -1) && (Fj_l == -1)) || ((C1_l == -1) && (Fj_l != -1)))
                    {
                        std::cout << "Error building local domain for cell " << cell << std::endl;
                        std::cout << "Periodic connectivity for outer face:" << std::endl;
                        std::cout << "Local idx: " << face_l << " Global idx: " << face_g << std::endl;
                        std::cout << "Could not find cell " << boundaryConditions_int(face_l,0)
                                  << " or face " << boundaryConditions_int(face_l,4) << std::endl;
                        assert(false);
                    }
                }
                else // otherwise it is needed
                {
                    if((C1_l == -1) || (Fj_l == -1))
                    {
                        std::cout << "Error building local domain for cell " << cell << std::endl;
                        std::cout << "Periodic connectivity for inner face:" << std::endl;
                        std::cout << "Local idx: " << face_l << " Global idx: " << face_g << std::endl;
                        std::cout << "Could not find cell " << boundaryConditions_int(face_l,0)
                                  << " or face " << boundaryConditions_int(face_l,4) << std::endl;
                        assert(false);
                    }
                }
                boundaryConditions_int(face_l,0) = C1_l;
                boundaryConditions_int(face_l,4) = Fj_l;
            }
        } else
        {
            boundaryType(face_l) = BoundaryType::INTERNAL;
            boundaryConditions_int.row(face_l) = 0;
            for(int i=0; i<6; ++i) boundaryConditions_bad(face_l,i) = 0.0;
        }
    }
    TypeBasedBoundaryIndicator bndIndicator(&boundaryType,BoundaryType::INTERNAL);

    // ********************** END BUILD LOCAL DOMAIN *********************** //
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Initialize AD variables" << std::endl;
    #endif
    // Initialize AD types for dependent and independent variables
    VectorXact res(4);
    MatX3geo vertices_coords(Nv,3);
    VecXflw flwC_u(Nc), flwC_v(Nc), flwC_w(Nc), flwC_p(Nc);
    VectorX limit_u(Nc), limit_v(Nc), limit_w(Nc), flwC_mut(Nc), momCoeffs(Nc);

    int Nderiv = (T==GEOMETRY)? 3*Nv : 4*Nc;
    int Nround = (Nderiv+VECTORSIZE-1)/VECTORSIZE;

    deriv.resize(Nderiv,4);

    for(int i=0; i<Nv; ++i)
    {
        int vert = vertices[i];
        vertices_coords(i,0) = m_inData_p->verticesCoord(vert,0);
        vertices_coords(i,1) = m_inData_p->verticesCoord(vert,1);
        vertices_coords(i,2) = m_inData_p->verticesCoord(vert,2);
    }
    for(int i=0; i<Nc; ++i)
    {
        int C0 = cells[i];
        flwC_u(i) = m_inData_p->u(C0);
        flwC_v(i) = m_inData_p->v(C0);
        flwC_w(i) = m_inData_p->w(C0);
        flwC_p(i) = m_inData_p->p(C0);
        limit_u(i) = m_inData_p->ulim(C0);
        limit_v(i) = m_inData_p->vlim(C0);
        limit_w(i) = m_inData_p->wlim(C0);
        flwC_mut(i) = m_inData_p->mut(C0);
        momCoeffs(i) = m_inData_p->mdiag(C0);
    }

    PassiveScalar rho = m_inData_p->rho,
                  mu  = m_inData_p->mu,
                  rpm = m_inData_p->rotationalSpeed;

    // allocate large variables that are used multiple times in the differentiation loops
    MatX3geo faces_area(Nf,3), faces_centroid(Nf,3), cells_centroid(Nc,3), faces_r0(Nf,3), faces_r1(Nf,3);

    VecXgeo faces_alfa0(Nf), faces_alfa1(Nf), faces_wf(Nf), cells_volume(Nc);

    VectorXact flwF_u(Nf_inner), flwF_v(Nf_inner), flwF_w(Nf_inner), flwF_p(Nf_inner);

    MatrixX3act nabla_u(Nc,3), nabla_v(Nc,3), nabla_w(Nc,3), nabla_p(Nc,3);

    // *********************** DIFFERENTIATION LOOPS *********************** //
    for(int derivRound=0; derivRound<Nround; ++derivRound)
    {
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Deriv round: " << derivRound+1 << "/" << Nround << std::endl;
    #endif
    // seed variables, something goes wrong if VECTORSIZE is not multiple of 4
    seedVariables(derivRound,Nc,Nv,flwC_u,flwC_v,flwC_w,flwC_p,vertices_coords);

    // only recompute geometric properties for geometric variables
    if(T==GEOMETRY || derivRound==0)
    {
    // *********************** GEOMETRIC PROPERTIES ************************ //
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Geometric properties" << std::endl;
    #endif
    GEOMETRICPROPERTIES(ScalarGeo, Nc, Nf, connectivity, verticesStart,
      verticesIndex, vertices_coords, faces_centroid, faces_area, faces_r0, faces_r1,
      faces_alfa0, faces_alfa1, faces_wf, cells_centroid, cells_volume)

    // ********************* END GEOMETRIC PROPERTIES ********************** //
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Map periodics" << std::endl;
    #endif
    // *************************** MAP PERIODICS *************************** //
    // recomputation of sector angles needed as they depend on grid coords
    for(int Fi=0; Fi < Nf_inner; ++Fi)
        if(boundaryType(Fi)==BoundaryType::PERIODIC && boundaryConditions_int(Fi,1)==1)
        {
            int Fj = boundaryConditions_int(Fi,4);

            double delta = getVal(faces_centroid(Fi,0))*getVal(faces_centroid(Fj,1))-
                           getVal(faces_centroid(Fi,1))*getVal(faces_centroid(Fj,0));

            ScalarGeo theta = ((delta>0.0)? -1.0 : 1.0) * acos(ScalarGeo(
                              (faces_centroid(Fi,0)*faces_centroid(Fj,0)+
                               faces_centroid(Fi,1)*faces_centroid(Fj,1))/
                              (faces_centroid.block(Fi,0,1,2).norm()*
                               faces_centroid.block(Fj,0,1,2).norm())));

            faces_alfa1(Fi)  = -faces_alfa0(Fj);
            faces_r1.row(Fi) = faces_r0.row(Fj);
            faces_wf(Fi) = 1.0f/(1.0f+faces_r0.row(Fi).norm()/faces_r1.row(Fi).norm());

            faces_alfa1(Fj) = -faces_alfa0(Fi);
            faces_r1.row(Fj) = faces_r0.row(Fi);
            faces_wf(Fj) = 1.0f/(1.0f+faces_r0.row(Fj).norm()/faces_r1.row(Fj).norm());

            boundaryConditions_bad(Fi,2) = cos(theta);
            boundaryConditions_bad(Fi,3) = sin(theta);

            boundaryConditions_bad(Fj,2) = cos(theta);
            boundaryConditions_bad(Fj,3) = -1.0*sin(theta);
        }
    // ************************* END MAP PERIODICS ************************* //
    } // if(T==GEOMETRY || derivRound==0)
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Boundary values" << std::endl;
    #endif
    // ************************** BOUNDARY VALUES ************************** //
    // determine if we need fixed point iterations to make
    // boundary face values and gradients consistent
    bool iterateV = false, iterateP = false;

    for(int j=0; j<Nf_inner; ++j) {
        iterateV |= (boundaryType(j)==BoundaryType::SYMMETRY);
        iterateP |= (boundaryType(j)==BoundaryType::VELOCITY ||
                     boundaryType(j)==BoundaryType::WALL || iterateV);
    }

    if(iterateV) for(int i=0; i<Nc; ++i) {
        nabla_u(i,0) = 0.0;  nabla_u(i,1) = 0.0;  nabla_u(i,2) = 0.0;
        nabla_v(i,0) = 0.0;  nabla_v(i,1) = 0.0;  nabla_v(i,2) = 0.0;
        nabla_w(i,0) = 0.0;  nabla_w(i,1) = 0.0;  nabla_w(i,2) = 0.0;
    }

    if(iterateP) for(int i=0; i<Nc; ++i) {
        nabla_p(i,0) = 0.0;  nabla_p(i,1) = 0.0;  nabla_p(i,2) = 0.0;
    }

    // we dont iterate for the geometry jacobian, as it is not as important
    int nFixPtIter = 1+(FIXEDPOINTITER-1)*int(iterateP && (T==FLOW));

    // START FIXED POINT
    for(int k=0; k<nFixPtIter; ++k)
    {
    for(int j=0; j<Nf_inner; ++j)
    {
        int C0 = connectivity[j].first, C1;
        ActiveScalar u_C1, v_C1;
        PassiveScalar wallRotSpeed;
        RowVector3 d0;
        for(int i=0; i<3; ++i)
          d0(i) = getVal(faces_r0(j,i))-getVal(faces_alfa0(j))*getVal(faces_area(j,i));

        switch(boundaryType(j))
        {
          case BoundaryType::VELOCITY:
            flwF_u(j) = boundaryConditions_bad(j,0);
            flwF_v(j) = boundaryConditions_bad(j,1);
            flwF_w(j) = boundaryConditions_bad(j,2);
            flwF_p(j) = flwC_p(C0)+nabla_p(C0,0)*d0(0)+nabla_p(C0,1)*d0(1)+nabla_p(C0,2)*d0(2);
            break;
          case BoundaryType::TOTALPRESSURE:
            assert(false); // NOT IMPLEMENTED YET
          case BoundaryType::PRESSURE:
            flwF_u(j) = flwC_u(C0);
            flwF_v(j) = flwC_v(C0);
            flwF_w(j) = flwC_w(C0);
            flwF_p(j) = boundaryConditions_bad(j,3);
            break;
          case BoundaryType::MASSFLOW:
            assert(false); // NOT IMPLEMENTED YET
          case BoundaryType::WALL:
            wallRotSpeed = getVal(boundaryConditions_bad(j,0));
            flwF_u(j) = -wallRotSpeed*faces_centroid(j,1);
            flwF_v(j) =  wallRotSpeed*faces_centroid(j,0);
            flwF_w(j) = 0.0;
            flwF_p(j) = flwC_p(C0)+nabla_p(C0,0)*d0(0)+nabla_p(C0,1)*d0(1)+nabla_p(C0,2)*d0(2);
            break;
          case BoundaryType::SYMMETRY:
            flwF_u(j) = flwC_u(C0)+(nabla_u(C0,0)*d0(0)+nabla_u(C0,1)*d0(1)+nabla_u(C0,2)*d0(2))*limit_u(C0);
            flwF_v(j) = flwC_v(C0)+(nabla_v(C0,0)*d0(0)+nabla_v(C0,1)*d0(1)+nabla_v(C0,2)*d0(2))*limit_v(C0);
            flwF_w(j) = flwC_w(C0)+(nabla_w(C0,0)*d0(0)+nabla_w(C0,1)*d0(1)+nabla_w(C0,2)*d0(2))*limit_w(C0);
            flwF_p(j) = flwC_p(C0)+ nabla_p(C0,0)*d0(0)+nabla_p(C0,1)*d0(1)+nabla_p(C0,2)*d0(2);
            break;
          case BoundaryType::PERIODIC:
            C1 = boundaryConditions_int(j,0); //Index of the neighbour cell
            // Velocity vector of neighbour cell rotated before interpolation
            u_C1 = flwC_u(C1)*boundaryConditions_bad(j,2)-flwC_v(C1)*boundaryConditions_bad(j,3),
            v_C1 = flwC_u(C1)*boundaryConditions_bad(j,3)+flwC_v(C1)*boundaryConditions_bad(j,2);
            flwF_u(j) = flwC_u(C0)*faces_wf(j)+u_C1*(1.0f-faces_wf(j));
            flwF_v(j) = flwC_v(C0)*faces_wf(j)+v_C1*(1.0f-faces_wf(j));
            flwF_w(j) = flwC_w(C0)*faces_wf(j)+flwC_w(C1)*(1.0f-faces_wf(j));
            flwF_p(j) = flwC_p(C0)*faces_wf(j)+flwC_p(C1)*(1.0f-faces_wf(j));
            break;
          case BoundaryType::GHOST:
            assert(false); // NOT EXPECTED HERE
        }
        // Reference frame change
        if(boundaryType(j)==BoundaryType::VELOCITY ||
           boundaryType(j)==BoundaryType::TOTALPRESSURE) {
            flwF_u(j) += rpm*faces_centroid(j,1);
            flwF_v(j) -= rpm*faces_centroid(j,0);
        }
    }
    // ************************ END BOUNDARY VALUES ************************ //
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Face values and gradients" << std::endl;
    #endif
    // ********************* FACE VALUES AND GRADIENTS ********************* //
    #define COMPUTEGRADIENTS(phi,phiF,grad) mathUtils::gradients::greenGauss \
    <MatX3geo, VecXgeo, VecXflw, VectorXact, MatrixX3act, TypeBasedBoundaryIndicator>\
    (Nc, Nf_inner, connectivity, bndIndicator, faces_area, faces_wf, cells_volume, phi, phiF, grad)
    if(k==0 || iterateV) COMPUTEGRADIENTS(flwC_u,flwF_u,nabla_u);
    if(k==0 || iterateV) COMPUTEGRADIENTS(flwC_v,flwF_v,nabla_v);
    if(k==0 || iterateV) COMPUTEGRADIENTS(flwC_w,flwF_w,nabla_w);
    if(k==0 || iterateP) COMPUTEGRADIENTS(flwC_p,flwF_p,nabla_p);
    #undef COMPUTEGRADIENTS
    } // END FIXED POINT

    if(m_firstOrder) for(int i=0; i<Nc; ++i) {
        nabla_u(i,0) = 0.0;  nabla_u(i,1) = 0.0;  nabla_u(i,2) = 0.0;
        nabla_v(i,0) = 0.0;  nabla_v(i,1) = 0.0;  nabla_v(i,2) = 0.0;
        nabla_w(i,0) = 0.0;  nabla_w(i,1) = 0.0;  nabla_w(i,2) = 0.0;
    }
    // ******************* END FACE VALUES AND GRADIENTS ******************* //
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Mass and Momentum residuals" << std::endl;
    #endif
    // ********************* MASS & MOMENTUM RESIDUALS ********************* //
    // begin by initializing rotational sources
    ScalarGeo a = rho*cells_volume(0)*rpm;
    res(0) = a*(-2.0f*flwC_v(0)-rpm*cells_centroid(0,0));
    res(1) = a*( 2.0f*flwC_u(0)-rpm*cells_centroid(0,1));
    res(2) = 0.0;
    res(3) = 0.0;

    // optionally make the jacobian more diagonally dominant
//    PassiveScalar d = cells_volume(0)*rho/dt;
//    res(0) += d*(flwC_u(0)-getVal(flwC_u(0)));
//    res(1) += d*(flwC_v(0)-getVal(flwC_v(0)));
//    res(2) += d*(flwC_w(0)-getVal(flwC_w(0)));

    residualNorm.setZero();
    for(int i=0; i<faceDegEndIdx[0]; ++i)
    {
        int C0 = connectivity[i].first;

        ScalarGeo anb, anbp;
        ActiveScalar mdot, F[4],
                     u0,  v0,  w0,  u1,  v1,  w1,  uf,  vf,  wf,
                     u0c, v0c, w0c, u1c, v1c, w1c, ufc, vfc, wfc;
        u0c = v0c = w0c = u1c = v1c = w1c = ufc = vfc = wfc = 0.0;
        VectorXact nabla_pf(3);
        PassiveScalar fluxdir = 1.0;

        RowVec3geo d0, d1;

        d0 = faces_r0.row(i)-faces_alfa0(i)*faces_area.row(i);

        if(boundaryType(i)==BoundaryType::INTERNAL)
            d1 = faces_r1.row(i)-faces_alfa1(i)*faces_area.row(i);
        else if(boundaryType(i)==BoundaryType::PERIODIC) {
            int otherFace = boundaryConditions_int(i,4);
            d1 = faces_r1.row(otherFace)-faces_alfa1(otherFace)*faces_area.row(otherFace);
        } else
            for(int j=0; j<3; ++j) d1(j)=0.0;

        if(boundaryType(i)!=BoundaryType::PERIODIC && boundaryType(i)!=BoundaryType::INTERNAL) {
            assert(C0==0);
            anb  = (mu+flwC_mut(0))/faces_alfa0(i);
            anbp = cells_volume(0)/(momCoeffs(0)*faces_alfa0(i));
            u1 = uf = flwF_u(i);
            v1 = vf = flwF_v(i);
            w1 = wf = flwF_w(i);
        } else {
            anb = 0.0; anbp = 0.0;
        }
        mdot = rho*(faces_area(i,0)*flwF_u(i)+
                    faces_area(i,1)*flwF_v(i)+
                    faces_area(i,2)*flwF_w(i));

        if(boundaryType(i)==BoundaryType::VELOCITY || boundaryType(i)==BoundaryType::WALL)
        {
            u0 = flwC_u(0); u0c = nabla_u(0,0)*d0(0)+nabla_u(0,1)*d0(1)+nabla_u(0,2)*d0(2);
            v0 = flwC_v(0); v0c = nabla_v(0,0)*d0(0)+nabla_v(0,1)*d0(1)+nabla_v(0,2)*d0(2);
            w0 = flwC_w(0); w0c = nabla_w(0,0)*d0(0)+nabla_w(0,1)*d0(1)+nabla_w(0,2)*d0(2);
        }
        else if(boundaryType(i)==BoundaryType::PRESSURE || boundaryType(i)==BoundaryType::MASSFLOW)
        {
            u0 = uf;  v0 = vf;  w0 = wf;

            mdot += rho*anbp*(flwC_p(0)+nabla_p.row(0).dot(faces_r0.row(i))-flwF_p(i));
        }
        else if(boundaryType(i)==BoundaryType::SYMMETRY)
        {
            u0 = uf;  v0 = vf;  w0 = wf;
        }
        else if(boundaryType(i)==BoundaryType::PERIODIC && C0==0) // periodic
        {
            // we wait for the face for which this is cell C0, doing the flux for the primary
            // face of the matching pair, and rotating it if needed, introduces fake dependencies
            int C1 = boundaryConditions_int(i,0);

            ScalarGeo mut = flwC_mut(0)*faces_wf(i)+flwC_mut(C1)*(1.0f-faces_wf(i));
            ActiveScalar ur, vr, urc, vrc;

            anb = (mu+mut)/(faces_alfa0(i)-faces_alfa1(i));

            // continuity
            anbp = (cells_volume(0)+cells_volume(C1))/
                   ((momCoeffs(0)+momCoeffs(C1))*(faces_alfa0(i)-faces_alfa1(i)));

            RowVector3act nabla_p_C1;
            nabla_p_C1(0) = nabla_p(C1,0)*boundaryConditions_bad(i,2)-nabla_p(C1,1)*boundaryConditions_bad(i,3);
            nabla_p_C1(1) = nabla_p(C1,0)*boundaryConditions_bad(i,3)+nabla_p(C1,1)*boundaryConditions_bad(i,2);
            nabla_p_C1(2) = nabla_p(C1,2);

            for(int j=0; j<3; ++j)
                nabla_pf(j) = faces_wf(i)*nabla_p(0,j)+(1.0f-faces_wf(i))*nabla_p_C1(j);

            RowVec3geo r1_prime;
            r1_prime(0) = faces_r1(i,0)*boundaryConditions_bad(i,2)-faces_r1(i,1)*boundaryConditions_bad(i,3);
            r1_prime(1) = faces_r1(i,0)*boundaryConditions_bad(i,3)+faces_r1(i,1)*boundaryConditions_bad(i,2);
            r1_prime(2) = faces_r1(i,2);

            mdot += rho*anbp*(flwC_p(0)-flwC_p(C1));
            for(int j=0; j<3; ++j)
                mdot += rho*anbp*nabla_pf(j)*(faces_r0(i,j)-r1_prime(j));

            // centroid values, for diffusion flux
            u0 = flwC_u(0 ); u0c = nabla_u(0 ,0)*d0(0)+nabla_u(0 ,1)*d0(1)+nabla_u(0 ,2)*d0(2);
            v0 = flwC_v(0 ); v0c = nabla_v(0 ,0)*d0(0)+nabla_v(0 ,1)*d0(1)+nabla_v(0 ,2)*d0(2);
            w0 = flwC_w(0 ); w0c = nabla_w(0 ,0)*d0(0)+nabla_w(0 ,1)*d0(1)+nabla_w(0 ,2)*d0(2);

            ur = flwC_u(C1); urc = nabla_u(C1,0)*d1(0)+nabla_u(C1,1)*d1(1)+nabla_u(C1,2)*d1(2);
            vr = flwC_v(C1); vrc = nabla_v(C1,0)*d1(0)+nabla_v(C1,1)*d1(1)+nabla_v(C1,2)*d1(2);
            w1 = flwC_w(C1); w1c = nabla_w(C1,0)*d1(0)+nabla_w(C1,1)*d1(1)+nabla_w(C1,2)*d1(2);

            u1  = ur *boundaryConditions_bad(i,2);
            v1  = vr *boundaryConditions_bad(i,2);
            u1c = urc*boundaryConditions_bad(i,2)-vrc*boundaryConditions_bad(i,3)-vr *boundaryConditions_bad(i,3);
            v1c = urc*boundaryConditions_bad(i,3)+vrc*boundaryConditions_bad(i,2)+ur *boundaryConditions_bad(i,3);

            // face values, for advection flux
            if(mdot > 0.0)
            {
                uf = flwC_u(0); ufc = 0.0; vf = flwC_v(0); vfc = 0.0; wf = flwC_w(0); wfc = 0.0;
                for(int j=0; j<3; ++j) {
                    ufc += limit_u(0)*nabla_u(0,j)*faces_r0(i,j);
                    vfc += limit_v(0)*nabla_v(0,j)*faces_r0(i,j);
                    wfc += limit_w(0)*nabla_w(0,j)*faces_r0(i,j);
                }
            } else {
                ur = flwC_u(C1); urc = 0.0; vr = flwC_v(C1); vrc = 0.0; wf = flwC_w(C1); wfc = 0.0;
                for(int j=0; j<3; ++j) {
                    urc += limit_u(C1)*nabla_u(C1,j)*faces_r1(i,j);
                    vrc += limit_v(C1)*nabla_v(C1,j)*faces_r1(i,j);
                    wfc += limit_w(C1)*nabla_w(C1,j)*faces_r1(i,j);
                }
                uf  = ur *boundaryConditions_bad(i,2);
                vf  = vr *boundaryConditions_bad(i,2);
                ufc = urc*boundaryConditions_bad(i,2)-vrc*boundaryConditions_bad(i,3)-vr *boundaryConditions_bad(i,3);
                vfc = urc*boundaryConditions_bad(i,3)+vrc*boundaryConditions_bad(i,2)+ur *boundaryConditions_bad(i,3);
            }
        }
        else if(boundaryType(i)==BoundaryType::INTERNAL) // internal face
        {
            int C1 = connectivity[i].second;
            if(C0!=0) {
                assert(C1==0); // one of the cells needs to be the 0th
                fluxdir = -1.0;
            }
            ScalarGeo mut = flwC_mut(C0)*faces_wf(i)+flwC_mut(C1)*(1.0f-faces_wf(i));

            anb = (mu+mut)/(faces_alfa0(i)-faces_alfa1(i));

            // continuity
            anbp = (cells_volume(C0)+cells_volume(C1))/
                   ((momCoeffs(C0)+momCoeffs(C1))*(faces_alfa0(i)-faces_alfa1(i)));

            for(int j=0; j<3; ++j)
                nabla_pf(j) = faces_wf(i)*nabla_p(C0,j)+(1.0f-faces_wf(i))*nabla_p(C1,j);

            mdot += rho*anbp*(flwC_p(C0)-flwC_p(C1)+nabla_pf.dot(faces_r0.row(i)-faces_r1.row(i)));

            // centroid values, for diffusion flux
            u0 = flwC_u(C0); u0c = nabla_u(C0,0)*d0(0)+nabla_u(C0,1)*d0(1)+nabla_u(C0,2)*d0(2);
            v0 = flwC_v(C0); v0c = nabla_v(C0,0)*d0(0)+nabla_v(C0,1)*d0(1)+nabla_v(C0,2)*d0(2);
            w0 = flwC_w(C0); w0c = nabla_w(C0,0)*d0(0)+nabla_w(C0,1)*d0(1)+nabla_w(C0,2)*d0(2);

            u1 = flwC_u(C1); u1c = nabla_u(C1,0)*d1(0)+nabla_u(C1,1)*d1(1)+nabla_u(C1,2)*d1(2);
            v1 = flwC_v(C1); v1c = nabla_v(C1,0)*d1(0)+nabla_v(C1,1)*d1(1)+nabla_v(C1,2)*d1(2);
            w1 = flwC_w(C1); w1c = nabla_w(C1,0)*d1(0)+nabla_w(C1,1)*d1(1)+nabla_w(C1,2)*d1(2);

            // face values, for advection flux
            if(mdot > 0.0)
            {
                uf = flwC_u(C0); ufc = 0.0; vf = flwC_v(C0); vfc = 0.0; wf = flwC_w(C0); wfc = 0.0;
                for(int j=0; j<3; ++j) {
                    ufc += limit_u(C0)*nabla_u(C0,j)*faces_r0(i,j);
                    vfc += limit_v(C0)*nabla_v(C0,j)*faces_r0(i,j);
                    wfc += limit_w(C0)*nabla_w(C0,j)*faces_r0(i,j);
                }
            } else {
                uf = flwC_u(C1); ufc = 0.0; vf = flwC_v(C1); vfc = 0.0; wf = flwC_w(C1); wfc = 0.0;
                for(int j=0; j<3; ++j) {
                    ufc += limit_u(C1)*nabla_u(C1,j)*faces_r1(i,j);
                    vfc += limit_v(C1)*nabla_v(C1,j)*faces_r1(i,j);
                    wfc += limit_w(C1)*nabla_w(C1,j)*faces_r1(i,j);
                }
            }
        }
        else
            assert(false && "unknown/unhandled boundary type");

        // now we do fluxes
        F[0] = fluxdir*(mdot*(uf+ufc)-anb*(u1+u1c-u0-u0c)+faces_area(i,0)*flwF_p(i));
        F[1] = fluxdir*(mdot*(vf+vfc)-anb*(v1+v1c-v0-v0c)+faces_area(i,1)*flwF_p(i));
        F[2] = fluxdir*(mdot*(wf+wfc)-anb*(w1+w1c-w0-w0c)+faces_area(i,2)*flwF_p(i));
        F[3] = fluxdir*(mdot/rho);

        for(int j=0; j<4; ++j) {
            res(j) += F[j];
            residualNorm(j) += std::abs(F[j].value());
        }
    }
    // ******************* END MASS & MOMENTUM RESIDUALS ******************* //
    #ifdef PROBLEMCELL
    if(cell==PROBLEMCELL) std::cout << "Differentiation" << std::endl;
    #endif
    for(int i=0; i<4; ++i) residual(i) = res(i).value();

    for(int i=0; (i<VECTORSIZE) && (VECTORSIZE*derivRound+i < Nderiv); ++i)
      for(int k=0; k<4; ++k)
        deriv(VECTORSIZE*derivRound+i,k) = res(k).derivatives()(i);

    } // diff round

    return 0;
}
template
int PressureBasedCoupledSolverAdjoint::m_computeResidualForCell<PressureBasedCoupledSolverAdjoint::FLOW>(
    const int, Vector4f&, Vector4f&, vector<int>&, vector<int>&, vector<int>&, Matrix<numPrec,Dynamic,4>&) const;
template
int PressureBasedCoupledSolverAdjoint::m_computeResidualForCell<PressureBasedCoupledSolverAdjoint::GEOMETRY>(
    const int, Vector4f&, Vector4f&, vector<int>&, vector<int>&, vector<int>&, Matrix<numPrec,Dynamic,4>&) const;
}
