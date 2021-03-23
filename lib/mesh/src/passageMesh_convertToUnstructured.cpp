//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "passageMesh.h"
#include "unstructuredMesh.h"
#include <cmath>
#include <algorithm>
#include <fstream>

namespace mesh
{
class GlobalNumber
{
  public:
    GlobalNumber(const int Ni, const int Nj): m_Ni(Ni), m_Nj(Nj) {}
    int cell(const int i, const int j, const int k) const{
        return i+j*(m_Ni-1)+k*(m_Ni-1)*(m_Nj-1);
    }
    int vertex(const int i, const int j, const int k) const{
        return i+j*m_Ni+k*m_Ni*m_Nj;
    }
  private:
    GlobalNumber();
    int m_Ni;
    int m_Nj;
};

int PassageMesh::convertToUnstructured(UnstructuredMesh& target) const
{
    if(!m_isMeshed)
        return 1;

    int nStream = m_layerMeri[0].cols(),
        nPitch  = m_layerMeri[0].rows(),
        nSpan   = m_nSpanGrdPts;
    GlobalNumber globalNumber(nStream,nPitch);

    // Copy vertices
    target.vertices.coords = m_vertexCoords;
    target.vertices.number = m_vertexCoords.rows();

    // Defines faces
    int cellNumber = (nStream-1)*(nPitch-1)*(nSpan-1),
        faceNumber = cellNumber*3+(nStream-1)*(nPitch-1)+(nPitch-1)*(nSpan-1)+(nStream-1)*(nSpan-1);
    target.cells.number = cellNumber;
    target.faces.number = faceNumber;
    std::vector<int>().swap(target.faces.verticesStart);
    target.faces.verticesStart.reserve(faceNumber+1);
    std::vector<int>().swap(target.faces.verticesIndex);
    target.faces.verticesIndex.reserve(faceNumber*4);
    std::vector<std::pair<int,int> >().swap(target.faces.connectivity);
    target.faces.connectivity.reserve(faceNumber);

    target.faces.groups.clear();
    target.faces.groups.resize(7);
    target.faces.groups[HUB].first     = HUB;
    target.faces.groups[SHROUD].first  = SHROUD;
    target.faces.groups[INFLOW].first  = INFLOW;
    target.faces.groups[OUTFLOW].first = OUTFLOW;
    target.faces.groups[PER_1].first   = PER_1;
    target.faces.groups[PER_2].first   = PER_2;
    target.faces.groups[BLADE].first   = BLADE;
    target.setPeriodicPair(PER_1,PER_2);

    for(int k=0; k<nSpan-1; k++)
    {
        for(int j=0; j<nPitch-1; j++)
        {
            for(int i=0; i<nStream-1; i++)
            {
                int C0 = globalNumber.cell(i,j,k), C1;
                Boundaries group;

                // Constant k faces
                if(k==0){
                    target.faces.verticesStart.push_back(target.faces.verticesIndex.size());
                    target.faces.verticesIndex.push_back(globalNumber.vertex( i , j ,k));
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i+1, j ,k));
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j+1,k));
                    target.faces.verticesIndex.push_back(globalNumber.vertex( i ,j+1,k));
                    target.faces.connectivity.push_back(std::make_pair(C0,-1));
                    target.faces.groups[HUB].second.push_back(target.faces.connectivity.size()-1);
                }
                C1 = (k==nSpan-2)? -1 : globalNumber.cell(i,j,k+1);
                target.faces.verticesStart.push_back(target.faces.verticesIndex.size());
                target.faces.verticesIndex.push_back(globalNumber.vertex( i , j ,k+1));
                target.faces.verticesIndex.push_back(globalNumber.vertex( i ,j+1,k+1));
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j+1,k+1));
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1, j ,k+1));
                target.faces.connectivity.push_back(std::make_pair(C0,C1));
                if(k==nSpan-2)
                    target.faces.groups[SHROUD].second.push_back(target.faces.connectivity.size()-1);

                // Constant j faces
                if(j==0){
                    target.faces.verticesStart.push_back(target.faces.verticesIndex.size());
                    target.faces.verticesIndex.push_back(globalNumber.vertex( i ,j, k ));
                    target.faces.verticesIndex.push_back(globalNumber.vertex( i ,j,k+1));
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j,k+1));
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j, k ));
                    target.faces.connectivity.push_back(std::make_pair(C0,-1));
                    group = (i>=m_jLE && i<m_jTE)? BLADE : PER_1;
                    target.faces.groups[group].second.push_back(target.faces.connectivity.size()-1);
                }
                C1 = (j==nPitch-2)? -1 : globalNumber.cell(i,j+1,k);
                target.faces.verticesStart.push_back(target.faces.verticesIndex.size());
                target.faces.verticesIndex.push_back(globalNumber.vertex( i ,j+1, k ));
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j+1, k ));
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j+1,k+1));
                target.faces.verticesIndex.push_back(globalNumber.vertex( i ,j+1,k+1));
                target.faces.connectivity.push_back(std::make_pair(C0,C1));
                if(j==nPitch-2){
                    group = (i>=m_jLE && i<m_jTE)? BLADE : PER_2;
                    target.faces.groups[group].second.push_back(target.faces.connectivity.size()-1);
                }

                // Constant i faces
                if(i==0){
                    target.faces.verticesStart.push_back(target.faces.verticesIndex.size());
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i, j , k ));
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i,j+1, k ));
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i,j+1,k+1));
                    target.faces.verticesIndex.push_back(globalNumber.vertex(i, j ,k+1));
                    target.faces.connectivity.push_back(std::make_pair(C0,-1));
                    target.faces.groups[INFLOW].second.push_back(target.faces.connectivity.size()-1);
                }
                C1 = (i==nStream-2)? -1 : globalNumber.cell(i+1,j,k);
                target.faces.verticesStart.push_back(target.faces.verticesIndex.size());
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1, j , k ));
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1, j ,k+1));
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j+1,k+1));
                target.faces.verticesIndex.push_back(globalNumber.vertex(i+1,j+1, k ));
                target.faces.connectivity.push_back(std::make_pair(C0,C1));
                if(i==nStream-2)
                    target.faces.groups[OUTFLOW].second.push_back(target.faces.connectivity.size()-1);
            }
        }
    }
    target.faces.verticesStart.push_back(target.faces.verticesIndex.size());
    target.m_meshIsDefined = true;

    return 0;
}

int PassageMesh::saveSU2file(const std::string &filePath) const
{
    if(!m_isMeshed) return 1;

    std::ofstream file;
    file.open(filePath.c_str());
    file << "NDIME= 3" << std::endl;
    if(!file.good()) return 2;

    int nStream = m_layerMeri[0].cols(),
        nPitch  = m_layerMeri[0].rows(),
        nSpan   = m_nSpanGrdPts;
    GlobalNumber globalNumber(nStream,nPitch);

    int nCell = (nStream-1)*(nPitch-1)*(nSpan-1),
        nVert =  nStream*nPitch*nSpan;

    // cells
    file << "NELEM= "  << nCell << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int j=0; j<nPitch-1; ++j)
        for(int i=0; i<nStream-1; ++i)
          file << "12\t"
               << globalNumber.vertex( i , j , k ) << "\t"
               << globalNumber.vertex(i+1, j , k ) << "\t"
               << globalNumber.vertex(i+1,j+1, k ) << "\t"
               << globalNumber.vertex( i ,j+1, k ) << "\t"
               << globalNumber.vertex( i , j ,k+1) << "\t"
               << globalNumber.vertex(i+1, j ,k+1) << "\t"
               << globalNumber.vertex(i+1,j+1,k+1) << "\t"
               << globalNumber.vertex( i ,j+1,k+1) << "\t"
               << globalNumber.cell  ( i , j , k ) << std::endl;

    // vertices
    file << "NPOIN= " << nVert << std::endl;
    for(int k=0; k<nSpan; ++k)
      for(int j=0; j<nPitch; ++j)
        for(int i=0; i<nStream; ++i) {
          int iVert = globalNumber.vertex(i,j,k);
          file << m_vertexCoords.row(iVert) << "\t" << iVert << std::endl;
        }

    // boundaries
    file << "NMARK= 7" << std::endl;

    file << "MARKER_TAG= hub" << std::endl;
    file << "MARKER_ELEMS= " << (nStream-1)*(nPitch-1) << std::endl;
    for(int j=0; j<nPitch-1; ++j)
      for(int i=0; i<nStream-1; ++i)
        file << "9\t"
             << globalNumber.vertex( i , j ,0) << "\t"
             << globalNumber.vertex(i+1, j ,0) << "\t"
             << globalNumber.vertex(i+1,j+1,0) << "\t"
             << globalNumber.vertex( i ,j+1,0) << std::endl;

    file << "MARKER_TAG= shroud" << std::endl;
    file << "MARKER_ELEMS= " << (nStream-1)*(nPitch-1) << std::endl;
    for(int j=0; j<nPitch-1; ++j)
      for(int i=0; i<nStream-1; ++i)
        file << "9\t"
             << globalNumber.vertex( i , j ,nSpan-1) << "\t"
             << globalNumber.vertex(i+1, j ,nSpan-1) << "\t"
             << globalNumber.vertex(i+1,j+1,nSpan-1) << "\t"
             << globalNumber.vertex( i ,j+1,nSpan-1) << std::endl;

    file << "MARKER_TAG= inlet" << std::endl;
    file << "MARKER_ELEMS= " << (nPitch-1)*(nSpan-1) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int j=0; j<nPitch-1; ++j)
        file << "9\t"
             << globalNumber.vertex(0, j , k ) << "\t"
             << globalNumber.vertex(0,j+1, k ) << "\t"
             << globalNumber.vertex(0,j+1,k+1) << "\t"
             << globalNumber.vertex(0, j ,k+1) << std::endl;

    file << "MARKER_TAG= outlet" << std::endl;
    file << "MARKER_ELEMS= " << (nPitch-1)*(nSpan-1) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int j=0; j<nPitch-1; ++j)
        file << "9\t"
             << globalNumber.vertex(nStream-1, j , k ) << "\t"
             << globalNumber.vertex(nStream-1,j+1, k ) << "\t"
             << globalNumber.vertex(nStream-1,j+1,k+1) << "\t"
             << globalNumber.vertex(nStream-1, j ,k+1) << std::endl;

    file << "MARKER_TAG= per1" << std::endl;
    file << "MARKER_ELEMS= " << (nSpan-1)*(nStream-1-m_jTE+m_jLE) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int i=0; i<m_jLE; ++i)
        file << "9\t"
             << globalNumber.vertex( i ,0, k ) << "\t"
             << globalNumber.vertex(i+1,0, k ) << "\t"
             << globalNumber.vertex(i+1,0,k+1) << "\t"
             << globalNumber.vertex( i ,0,k+1) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int i=m_jTE; i<nStream-1; ++i)
        file << "9\t"
             << globalNumber.vertex( i ,0, k ) << "\t"
             << globalNumber.vertex(i+1,0, k ) << "\t"
             << globalNumber.vertex(i+1,0,k+1) << "\t"
             << globalNumber.vertex( i ,0,k+1) << std::endl;

    file << "MARKER_TAG= per2" << std::endl;
    file << "MARKER_ELEMS= " << (nSpan-1)*(nStream-1-m_jTE+m_jLE) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int i=0; i<m_jLE; ++i)
        file << "9\t"
             << globalNumber.vertex( i ,nPitch-1, k ) << "\t"
             << globalNumber.vertex(i+1,nPitch-1, k ) << "\t"
             << globalNumber.vertex(i+1,nPitch-1,k+1) << "\t"
             << globalNumber.vertex( i ,nPitch-1,k+1) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int i=m_jTE; i<nStream-1; ++i)
        file << "9\t"
             << globalNumber.vertex( i ,nPitch-1, k ) << "\t"
             << globalNumber.vertex(i+1,nPitch-1, k ) << "\t"
             << globalNumber.vertex(i+1,nPitch-1,k+1) << "\t"
             << globalNumber.vertex( i ,nPitch-1,k+1) << std::endl;

    file << "MARKER_TAG= blade" << std::endl;
    file << "MARKER_ELEMS= " << 2*(nSpan-1)*(m_jTE-m_jLE) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int i=m_jLE; i<m_jTE; ++i)
        file << "9\t"
             << globalNumber.vertex( i ,0, k ) << "\t"
             << globalNumber.vertex(i+1,0, k ) << "\t"
             << globalNumber.vertex(i+1,0,k+1) << "\t"
             << globalNumber.vertex( i ,0,k+1) << std::endl;
    for(int k=0; k<nSpan-1; ++k)
      for(int i=m_jLE; i<m_jTE; ++i)
        file << "9\t"
             << globalNumber.vertex( i ,nPitch-1, k ) << "\t"
             << globalNumber.vertex(i+1,nPitch-1, k ) << "\t"
             << globalNumber.vertex(i+1,nPitch-1,k+1) << "\t"
             << globalNumber.vertex( i ,nPitch-1,k+1) << std::endl;

    file.close();

    return 0;
}
}
