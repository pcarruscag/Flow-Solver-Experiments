//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "adjoint.h"

#include "../../mathUtils/src/matrix.h"
#include <unsupported/Eigen/AutoDiff>

#include <algorithm>
#include <iostream>
#include <iomanip>

#define VECTORSIZE 128

using std::vector;
using std::pair;
using std::cout;
using std::endl;
using namespace mathUtils::matrix;

typedef float diffPrec;
typedef AutoDiffScalar<Matrix<diffPrec,VECTORSIZE,1> > addouble_t;
typedef Matrix<addouble_t,Dynamic,3> MatrixX3add;
typedef Matrix<addouble_t,Dynamic,1> VectorXadd;

namespace adjoint
{
int PressureBasedCoupledSolverAdjoint::m_computeObjective()
{
    m_tObjJac -= omp_get_wtime();
    cout << std::left << std::setw(90) << "|" << "|" << endl;
    cout << std::left << std::setw(90) << "| Computing objective function jacobians..." << "|" << endl;

    // Get faces needed for the calculation
    vector<int> vertices, faces, verticesStart, verticesIndex, cells;
    int Nv, Nf, Nc, numInlets;

    // first pass - inlets
    for(int i=0; i<m_inData_p->boundaryNumber; ++i)
        if(m_inData_p->boundaryType(i) == BoundaryType::VELOCITY ||
           m_inData_p->boundaryType(i) == BoundaryType::TOTALPRESSURE)
            faces.push_back(i);
    numInlets = faces.size();

    // second pass - outlets
    for(int i=0; i<m_inData_p->boundaryNumber; ++i)
        if(m_inData_p->boundaryType(i) == BoundaryType::PRESSURE ||
           m_inData_p->boundaryType(i) == BoundaryType::MASSFLOW)
            faces.push_back(i);
    Nc = Nf = faces.size();

    // get cells and vertices
    int counter=0;
    cells.reserve(Nc);
    for(int Fj : faces)
    {
        cells.push_back(m_inData_p->connectivity[Fj].first);
        for(int i=m_inData_p->verticesStart[Fj];
                i<m_inData_p->verticesStart[Fj+1]; ++i)
        {
            ++counter;
            int vertex = m_inData_p->verticesIndex[i];
            if(!existsInVector(vertices,vertex)) vertices.push_back(vertex);
        }
    }
    Nv = vertices.size();

    cout << std::left << "| Objectives depend on "
         << std::setw(5) << Nc << " cells and "
         << std::setw(5) << Nv << std::setw(46) << " vertices" << "|" << endl;

    // face definition (global to local)
    verticesStart.reserve(Nf+1);
    verticesIndex.reserve(counter);
    for(int Fj : faces)
    {
        verticesStart.push_back(verticesIndex.size());
        for(int i=m_inData_p->verticesStart[Fj];
                i<m_inData_p->verticesStart[Fj+1]; ++i)
        {
            int vertex_g = m_inData_p->verticesIndex[i];
            int vertex_l = findInVector(vertices,vertex_g);
            verticesIndex.push_back(vertex_l);
        }
    }
    verticesStart.push_back(verticesIndex.size());

    // Initialize AD types
    MatrixX3add vertices_coords(Nv,3);
    VectorXadd  flwC_u(Nc), flwC_v(Nc), flwC_w(Nc), flwC_p(Nc);

    #pragma omp parallel for num_threads(m_partNum) schedule(static,1000)
    for(int i=0; i<Nv; ++i)
        for(int j=0; j<3; ++j)
            vertices_coords(i,j) = m_inData_p->verticesCoord(vertices[i],j);

    #pragma omp parallel for num_threads(m_partNum) schedule(static,1000)
    for(int i=0; i<Nc; ++i) {
        int C0 = cells[i];
        flwC_u(i) = m_inData_p->u(C0);
        flwC_v(i) = m_inData_p->v(C0);
        flwC_w(i) = m_inData_p->w(C0);
        flwC_p(i) = m_inData_p->p(C0);
    }
    // objectives
    addouble_t mdotIn, mdotOut,
               blockage, thrust,   torque,
               deltaPtt, deltaPts, deltaPss,
               etatt,    etats,    etass;

    // constants
    diffPrec rpm = m_inData_p->rotationalSpeed,
             rho = m_inData_p->rho;

    // allocate reused variables
    MatrixX3add faces_area(Nf,3), faces_centroid(Nf,3);

    // allocate objective jacobians
    m_objValues.setZero(m_objNum);
    m_objFlowJacobian.setZero(4*m_inData_p->cellNumber,m_objNum);
    m_objGeoJacobian.setZero(3*m_inData_p->vertexNumber,m_objNum);

    // approximate center of the domain, normals must point away from it
    diffPrec centroid[3];
    for(int i=0; i<3; ++i) centroid[i] = m_inData_p->verticesCoord.col(i).mean();

    // *********************** DIFFERENTIATION LOOPS *********************** //
    for(int varType=0; varType<2; ++varType) // 0 for geometry 1 for flow
    {
    int Nderiv = varType*4*Nc + (1-varType)*3*Nv;
    int Nround = Nderiv/VECTORSIZE+(Nderiv%VECTORSIZE!=0);

    for(int derivRound=0; derivRound<Nround; ++derivRound)
    {
    // seed variables, something goes wrong if VECTORSIZE is not multiple of 4
    if(varType==0) {
        for(int i=0; i<VECTORSIZE; ++i) {
            // clear previous seeds
            int outerIdx = (VECTORSIZE*(derivRound-1)+i)/3,
                innerIdx = (VECTORSIZE*(derivRound-1)+i)%3;

            if(derivRound>0)
            vertices_coords(outerIdx,innerIdx).derivatives()(i) = diffPrec(0);

            // seed new variables
            outerIdx = (VECTORSIZE*derivRound+i)/3;
            innerIdx = (VECTORSIZE*derivRound+i)%3;

            if(outerIdx==Nv) break;
            vertices_coords(outerIdx,innerIdx).derivatives()(i) = diffPrec(1);
        }
    } else {
        for(int i=0; i<VECTORSIZE; ) {
            // clear previous seeds
            int outerIdx = (VECTORSIZE*(derivRound-1)+i)/4;

            if(derivRound>0) {
            flwC_u(outerIdx).derivatives()(i  ) = diffPrec(0);
            flwC_v(outerIdx).derivatives()(i+1) = diffPrec(0);
            flwC_w(outerIdx).derivatives()(i+2) = diffPrec(0);
            flwC_p(outerIdx).derivatives()(i+3) = diffPrec(0);
            }
            // seed new variables
            outerIdx = (VECTORSIZE*derivRound+i)/4;

            if(outerIdx==Nc) break;
            flwC_u(outerIdx).derivatives()(i++) = diffPrec(1);
            flwC_v(outerIdx).derivatives()(i++) = diffPrec(1);
            flwC_w(outerIdx).derivatives()(i++) = diffPrec(1);
            flwC_p(outerIdx).derivatives()(i++) = diffPrec(1);
        }
    }

    // geometric properties of faces
    if(varType==0) // only recompute for geometric variables
    {
    counter = 0;
    #pragma omp parallel for num_threads(m_partNum) schedule(dynamic,100) reduction(+:counter)
    for(int i=0; i<Nf; i++)
    {
        for(int j=0; j<3; ++j)
            faces_centroid(i,j) = 0.0;
        int j = verticesStart[i];
        int nFaceVertex = verticesStart[i+1]-j;
        if(nFaceVertex == 3)
            faces_area.row(i) = 0.5*(vertices_coords.row(verticesIndex[j+1])-
                                vertices_coords.row(verticesIndex[ j ])).cross(
                                vertices_coords.row(verticesIndex[j+2])-
                                vertices_coords.row(verticesIndex[j]));
        else if(nFaceVertex == 4)
            faces_area.row(i) = 0.5*(vertices_coords.row(verticesIndex[j+2])-
                                vertices_coords.row(verticesIndex[ j ])).cross(
                                vertices_coords.row(verticesIndex[j+3])-
                                vertices_coords.row(verticesIndex[j+1]));
        else assert(false && "unknown face shape");

        for( ; j<verticesStart[i+1]; ++j)
            faces_centroid.row(i) += vertices_coords.row(verticesIndex[j]);
        faces_centroid.row(i) /= diffPrec(nFaceVertex);

        // check direction is correct
        diffPrec dotprod = 0.0;
        for(j=0; j<3; ++j) dotprod += (centroid[j]-faces_centroid(i,j).value())*faces_area(i,j).value();

        if(dotprod>0.0f) {
            faces_area.row(i) = -1.0*faces_area.row(i);
            ++counter;
        }
    }
    if(derivRound==0)
    cout << std::left << "| " << std::setw(5) << counter << std::setw(83) << " faces were fliped" << "|" << endl;
    } // if(varType==0)

    // intermediate results and their reduction variables
    addouble_t areaIn,  areaOut,  power,
               aintPin, aintPout, mintP0in,  mintP0out,
               aavgPin, aavgPout, mavgP0in,  mavgP0out;
    { // reduction scope
    int NP = m_partNum;
    VectorXadd _areaIn(NP),  _areaOut(NP),  _mdotIn(NP),   _mdotOut(NP),
               _aintPin(NP), _aintPout(NP), _mintP0in(NP), _mintP0out(NP),
               _forceIn(NP), _forceOut(NP), _mintWin(NP),  _mintWout(NP),
               _torque(NP),  _mintVnOut(NP);

    #pragma omp parallel num_threads(m_partNum)
    {
        int i = omp_get_thread_num();
        _areaIn(i) = _mdotIn(i) = _aintPin(i) = _mintP0in(i) = _forceIn(i) = _mintWin(i) = 0.0;
        _areaOut(i)= _mdotOut(i)= _aintPout(i)= _mintP0out(i)= _forceOut(i)= _mintWout(i)= 0.0;
        _torque(i) = _mintVnOut(i) = 0.0;
    }

    // surface integrals
    #pragma omp parallel for num_threads(m_partNum) schedule(dynamic,100)
    for(int i=0; i<numInlets; ++i) // inlets
    {
        int prt = omp_get_thread_num();
        // values at face
        addouble_t uf = m_inData_p->boundaryConditions(faces[i],0),
                   vf = m_inData_p->boundaryConditions(faces[i],1),
                   wf = m_inData_p->boundaryConditions(faces[i],2),
                   pf = flwC_p(i);
        addouble_t darea = faces_area.row(i).norm(),
                   dmdot = rho*(faces_area(i,0)*uf+faces_area(i,1)*vf+faces_area(i,2)*wf),
                   rVt   = vf*faces_centroid(i,0)-uf*faces_centroid(i,1),
                   p0f   = pf+0.5f*rho*(uf*uf+vf*vf+wf*wf);
        // accumulate
        _areaIn(prt)   += darea;
        _mdotIn(prt)   += dmdot;
        _aintPin(prt)  += darea*pf;
        _mintP0in(prt) += dmdot*p0f;
        _forceIn(prt)  += faces_area(i,2)*pf;
        _mintWin(prt)  += dmdot*wf;
        _torque(prt)   += dmdot*rVt;
    }

    #pragma omp parallel for num_threads(m_partNum) schedule(dynamic,100)
    for(int i=numInlets; i<Nf; ++i) // outlets
    {
        int prt = omp_get_thread_num();
        // values at face
        addouble_t uf = flwC_u(i)-rpm*faces_centroid(i,1),
                   vf = flwC_v(i)+rpm*faces_centroid(i,0),
                   wf = flwC_w(i),
                   pf = m_inData_p->boundaryConditions(faces[i],3);

        addouble_t darea = faces_area.row(i).norm(),
                   dmdot = rho*(faces_area(i,0)*uf+faces_area(i,1)*vf+faces_area(i,2)*wf),
                   rVt   = vf*faces_centroid(i,0)-uf*faces_centroid(i,1),
                   p0f   = pf+0.5f*rho*(uf*uf+vf*vf+wf*wf);
        // accumulate
        _areaOut(prt)   += darea;
        _mdotOut(prt)   += dmdot;
        _aintPout(prt)  += darea*pf;
        _mintP0out(prt) += dmdot*p0f;
        _forceOut(prt)  += faces_area(i,2)*pf;
        _mintWout(prt)  += dmdot*wf;
        _torque(prt)    += dmdot*rVt;
        _mintVnOut(prt) += dmdot*dmdot/rho/darea;
    }

    #pragma omp parallel num_threads(m_partNum)
    {
    #pragma omp task
    {
        areaIn  = _areaIn.sum();  aintPin  = _aintPin.sum();
        mdotIn  = _mdotIn.sum();  mintP0in = _mintP0in.sum();
        aavgPin = aintPin/areaIn; mavgP0in = mintP0in/mdotIn;
        thrust  = _forceOut.sum()-_forceIn.sum()-_mintWout.sum()+_mintWin.sum();
    }
    #pragma omp task
    {
        areaOut  = _areaOut.sum();   aintPout  = _aintPout.sum();
        mdotOut  = _mdotOut.sum();   mintP0out = _mintP0out.sum();
        aavgPout = aintPout/areaOut; mavgP0out = mintP0out/mdotOut;
        torque   = _torque.sum();    power     = torque*rpm;
        blockage = mdotOut*mdotOut/(rho*areaOut*_mintVnOut.sum());
    }}
    } // end reduction scope

    // objectives
    deltaPtt = mavgP0out-mavgP0in;
    deltaPts = aavgPout-mavgP0in;
    deltaPss = aavgPout-aavgPin;
    etatt    = (mintP0out+mintP0in)/rho;
    etats    = (aavgPout*mdotOut+mintP0in)/rho;
    etass    = (aavgPout*mdotOut+aavgPin*mdotIn)/rho;
    if(std::abs(rpm)<1e-6f) {etatt = 0.0; etats = 0.0; etass = 0.0;}
    else if(power > etatt) {etatt /= power; etats /= power; etass /= power;}
    else {etatt = power/etatt; etats = power/etats; etass = power/etass;}

    // derivatives
    #pragma omp parallel for num_threads(m_partNum) schedule(static,1)
    for(int k=0; k<m_objNum; ++k)
    {
        const addouble_t* obj;
        switch(m_objectiveList[k]) {
            case MDOT_IN:   obj = &mdotIn;   break;
            case MDOT_OUT:  obj = &mdotOut;  break;
            case BLOCKAGE:  obj = &blockage; break;
            case THRUST:    obj = &thrust;   break;
            case TORQUE:    obj = &torque;   break;
            case DELTAP_TT: obj = &deltaPtt; break;
            case DELTAP_TS: obj = &deltaPts; break;
            case DELTAP_SS: obj = &deltaPss; break;
            case ETA_TT:    obj = &etatt;    break;
            case ETA_TS:    obj = &etats;    break;
            case ETA_SS:    obj = &etass;    break;
            default: assert(false && "unknown objective type");
        }
        m_objValues(k) = obj->value();

        if(varType==0) {
            for(int i=0; (i<VECTORSIZE) && (VECTORSIZE*derivRound+i < 3*Nv); ++i) {
                int vloc = (VECTORSIZE*derivRound+i)/3,
                    cidx = (VECTORSIZE*derivRound+i)%3;
                int vglb = vertices[vloc];
                m_objGeoJacobian(3*vglb+cidx,k) = obj->derivatives()(i)/obj->value();
            }
        } else {
            for(int i=0; (i<VECTORSIZE) && (VECTORSIZE*derivRound+i < 4*Nc); i+=4) {
                int cloc = (VECTORSIZE*derivRound+i)/4;
                for(int j=0; j<4; ++j)
                    m_objFlowJacobian(4*cells[cloc]+j,k) = obj->derivatives()(i+j)/obj->value();
            }
        }
    }
    } // diff round
    // clear geometric properties derivatives before moving to flow variables
    #pragma omp parallel for num_threads(m_partNum) schedule(dynamic,100)
    for(int i=0; i<Nf; ++i)
        for(int j=0; j<3; ++j) {
            faces_area(i,j).derivatives().setZero();
            faces_centroid(i,j).derivatives().setZero();
        }
    } // var type

    cout << std::left << std::setw(90) << "| Objective values:" << "|" << endl;
    cout << std::left << "|  Mass flow in:  " << std::setw(72) << mdotIn.value()   << "|" << endl;
    cout << std::left << "|  Mass flow out: " << std::setw(72) << mdotOut.value()  << "|" << endl;
    cout << std::left << "|  Blockage:      " << std::setw(72) << blockage.value() << "|" << endl;
    cout << std::left << "|  Thrust:        " << std::setw(72) << thrust.value()   << "|" << endl;
    cout << std::left << "|  Torque:        " << std::setw(72) << torque.value()   << "|" << endl;
    cout << std::left << "|  Delta P tt:    " << std::setw(72) << deltaPtt.value() << "|" << endl;
    cout << std::left << "|  Delta P ts:    " << std::setw(72) << deltaPts.value() << "|" << endl;
    cout << std::left << "|  Delta P ss:    " << std::setw(72) << deltaPss.value() << "|" << endl;
    cout << std::left << "|  Efficiency tt: " << std::setw(72) << etatt.value()    << "|" << endl;
    cout << std::left << "|  Efficiency ts: " << std::setw(72) << etats.value()    << "|" << endl;
    cout << std::left << "|  Efficiency ss: " << std::setw(72) << etass.value()    << "|" << endl;
    m_tObjJac += omp_get_wtime();
    return 0;
}
}
