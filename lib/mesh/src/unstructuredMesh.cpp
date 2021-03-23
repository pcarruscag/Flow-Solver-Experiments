//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "unstructuredMesh.h"

#include "../../mathUtils/src/matrix.h"
#include "geometricProperties.h"

#include <algorithm>
#include <deque>
#include <queue>
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace mathUtils::matrix;

namespace mesh
{

UnstructuredMesh::UnstructuredMesh()
{
    m_meshIsDefined = false;
    m_geoPropsComputed = false;
    m_meshIsOrdered = false;
    m_PER_1 = m_PER_2 = -1;
}

int UnstructuredMesh::status() const
{
    if(!m_meshIsDefined)
        return 1;
    if(!m_geoPropsComputed)
        return 2;
    return 0;
}

int UnstructuredMesh::scale(const float factor)
{
    if(!m_meshIsDefined)
        return 1;
    if(m_geoPropsComputed)
        return 2;
    vertices.coords *= factor;
    return 0;
}

int UnstructuredMesh::computeGeoProps()
{
    if(!m_meshIsDefined)
        return 1;
    if(m_geoPropsComputed)
        return 0;
    m_geoPropsComputed = false;

    faces.area.resize(faces.number,NoChange);
    faces.centroid.resize(faces.number,NoChange);
    faces.alfa0.resize(faces.number);
    faces.alfa1.resize(faces.number);
    faces.r0.resize(faces.number,NoChange);
    faces.r1.resize(faces.number,NoChange);
    faces.d0.resize(faces.number,NoChange);
    faces.d1.resize(faces.number,NoChange);
    faces.wf.resize(faces.number);
    cells.centroid.resize(cells.number,NoChange);
    cells.volume.resize(cells.number);

    GEOMETRICPROPERTIES(float, cells.number, faces.number, faces.connectivity,
      faces.verticesStart, faces.verticesIndex, vertices.coords, faces.centroid, faces.area,
      faces.r0, faces.r1, faces.alfa0, faces.alfa1, faces.wf, cells.centroid, cells.volume)

    // compute d0 and d1, not done in GEOMETRICPROPERTIES because for
    // the adjoint solver it is more efficient to do it on the fly
    for(int i=0; i<faces.number; i++)
    {
        faces.d0.row(i) = faces.r0.row(i)-faces.alfa0(i)*faces.area.row(i);
        faces.d1.row(i) = faces.r1.row(i)-faces.alfa1(i)*faces.area.row(i);
    }
    m_geoPropsComputed = true;
    return 0;
}

int UnstructuredMesh::renumberCells(const VectorXi& perm)
{
    if(!m_meshIsDefined)
        return 1;
    // This test does not guarantee uniqueness of the permutation elements
    if(perm.rows()!=cells.number)
        return 2;

    m_geoPropsComputed = false;
    m_meshIsOrdered = true;

    // Renumber cells
    int C0, C1;
    for(int i=0; i<faces.number; i++){
        C0 = faces.connectivity[i].first;
        C1 = faces.connectivity[i].second;
        faces.connectivity[i].first = perm(C0);
        if(C1!=-1)
            faces.connectivity[i].second = perm(C1);
    }
    // Get new order for faces and renumbering function
    std::vector<int> faceSortIdx, faceNewIdx(faces.number);
    faceSortIdx = sortIndexes(faces.connectivity,&comparePair2);
    for(int i=0; i<faces.number; i++)
        faceNewIdx[faceSortIdx[i]] = i;

    // Sort connectivity
    std::vector<std::pair<int,int> > newConnectivity(faces.number);
    for(int i=0; i<faces.number; i++)
        newConnectivity[i] = faces.connectivity[faceSortIdx[i]];
    faces.connectivity = newConnectivity;

    // Renumber faces in groups
    for(auto & group : faces.groups)
        for(size_t j=0; j<group.second.size(); j++)
            group.second[j] = faceNewIdx[group.second[j]];

    // Update the face vertex information
    std::vector<int> newVerticesStart(faces.number+1), newVerticesIndex;
    newVerticesIndex.reserve(faces.verticesIndex.size());
    for(int i=0; i<faces.number; i++){
        newVerticesStart[i] = newVerticesIndex.size();
        for(int j=faces.verticesStart[faceSortIdx[i]]; j<faces.verticesStart[faceSortIdx[i]+1]; j++)
            newVerticesIndex.push_back(faces.verticesIndex[j]);
    }
    newVerticesStart[faces.number] = newVerticesIndex.size();
    faces.verticesStart = newVerticesStart;
    faces.verticesIndex = newVerticesIndex;

    return 0;
}

int UnstructuredMesh::rcmOrdering(VectorXi& permutation,
                                  const bool includePeriodics,
                                  const bool clusterPeriodics) const
{
    if(!m_meshIsDefined)
        return 1;
    if(includePeriodics && m_meshIsOrdered)
        return 2;

    // Build graph
    std::vector<std::vector<std::pair<int,int> > > connectivity(cells.number);
    // Interior faces
    int C0, C1;
    for(int i=0; i<faces.number; i++)
    {
        C0 = faces.connectivity[i].first;
        C1 = faces.connectivity[i].second;
        if(C1!=-1){
            connectivity[C0].push_back(std::make_pair(0,C1));
            connectivity[C1].push_back(std::make_pair(0,C0));
        }
    }
    std::vector<int> matchingNode(cells.number,-1);
    if(includePeriodics){
        for(size_t i=0; i<faces.groups[m_PER_1].second.size(); i++)
        {
            C0 = faces.connectivity[faces.groups[m_PER_1].second[i]].first;
            C1 = faces.connectivity[faces.groups[m_PER_2].second[i]].first;
            connectivity[C0].push_back(std::make_pair(0,C1));
            connectivity[C1].push_back(std::make_pair(0,C0));
            if(clusterPeriodics) {
                matchingNode[C0] = C1;
                matchingNode[C1] = C0;
            }
        }
    }
    // Calculate degree and sort in ascending order
    std::vector<std::pair<int,int> > degree(cells.number);
    for(int i=0; i<cells.number; i++)
        degree[i] = std::make_pair(connectivity[i].size(),i);

    for(int i=0; i<cells.number; i++){
        for(int j=0; j<degree[i].first; j++)
            connectivity[i][j].first = degree[connectivity[i][j].second].first;
        std::sort(connectivity[i].begin(),connectivity[i].end(),comparePair1<int,int>);
    }
    std::sort(degree.begin(),degree.end(),comparePair1<int,int>);

    // Start ordering
    std::vector<int> R;
    R.reserve(cells.number);
    std::vector<bool> isInserted(cells.number,false);
    std::deque<int> Q;

    while(R.size()<size_t(cells.number))
    {
        int P,C;
        // Get unused node of lowest degree to seed the empty queue
        for(P=0; P<cells.number; ++P)
            if(!isInserted[degree[P].second])
                break;
        Q.push_back(degree[P].second);

        while(!Q.empty()){
            C = Q.front();
            Q.pop_front();
            if(!isInserted[C]){
                R.push_back(C);
                isInserted[C] = true;

                if(matchingNode[C]>-1) { // if C is paired add the match immediately
                    int match = matchingNode[C];
                    R.push_back(match);
                    isInserted[match] = true;
                    // concatenate the neighbours of the pair in P
                    connectivity[C].insert(connectivity[C].end(),
                                           connectivity[match].begin(),
                                           connectivity[match].end());
                    std::sort(connectivity[C].begin(),connectivity[C].end(),comparePair1<int,int>);
                }
                // Add adjacent nodes to queue
                for(size_t i=0; i<connectivity[C].size(); i++)
                    Q.push_back(connectivity[C][i].second);
            }
        }
    }
    // Invert order
    permutation.resize(cells.number);
    for(int i=0; i<cells.number; i++)
        permutation(R[cells.number-1-i]) = i;

    return 0;
}

int UnstructuredMesh::partitioner(const int numParts, std::vector<UnstructuredMesh>& parts, const bool keepOrder)
{
    if(numParts<1) return 1;

    #ifdef MESH_VERBOSE
    std::cout << std::endl << "### Partitioning Unstructured Mesh ###" << std::endl << std::endl;
    #endif
    if(!keepOrder) {
        VectorXi permutation;
        int flag = rcmOrdering(permutation,true,true);
        if(flag!=0) return flag;
        flag = renumberCells(permutation);
        if(flag!=0) return flag;
    }
    // no partitioning needed for only 1 partition
    if(numParts<2) {
        parts.clear();
        parts.push_back(*this);
        m_originFace.resize(1);
        m_originFace[0].reserve(faces.number);
        for(int i=0; i<faces.number; m_originFace[0].push_back(i++)){}
        return 0;
    }

    // Build the cell based connectivity, easier to split
    std::vector<std::vector<std::pair<int,int> > > connectivity(cells.number);
    int bandwidth = 0;
    for(int i=0; i<faces.number; i++)
    {
        int C0 = faces.connectivity[i].first,
            C1 = faces.connectivity[i].second;
        connectivity[C0].push_back(std::make_pair(C1,i));
        if(C1!=-1) {
        connectivity[C1].push_back(std::make_pair(C0,i));
        bandwidth = std::max(bandwidth,std::abs(C1-C0));
        }
    }
    std::cout << "  Graph bandwidth: " << bandwidth << " ("
              << double(bandwidth)/cells.number << "%)" << std::endl << std::endl;

    // Determine the split locations
    ArrayXi splitLocation;
    splitLocation.setLinSpaced(numParts+1,0,cells.number+numParts-1);
    splitLocation(numParts) = cells.number;
    {
    // build the periodic pairs, to know where we can split
    std::vector<int> matchingNode(cells.number,-1);
    for(size_t i=0; i<faces.groups[m_PER_1].second.size(); i++)
    {
        int C0 = faces.connectivity[faces.groups[m_PER_1].second[i]].first,
            C1 = faces.connectivity[faces.groups[m_PER_2].second[i]].first;
        matchingNode[C0] = C1;
        matchingNode[C1] = C0;
    }
    for(int i=1; i<numParts; ++i) {
        int loc0 = splitLocation(i);
        while(matchingNode[splitLocation(i)-1]>-1)
            ++splitLocation(i);
        assert(splitLocation(i)<splitLocation(i+1) && "domain could not be split");
        #ifdef MESH_VERBOSE
        std::cout<<"  Part "<<i<<": "<<splitLocation(i)-splitLocation(i-1)<<" cells; "
                 <<"cut at "<<splitLocation(i)<<" ("<<loc0<<")"<<std::endl;
        #endif
    }
    #ifdef MESH_VERBOSE
    std::cout<<"  Part "<<numParts<<": "
             <<splitLocation(numParts)-splitLocation(numParts-1)<<" cells"<<std::endl;
    #endif
    }
    // Start spliting
    std::vector<UnstructuredMesh>().swap(parts);
    parts.resize(numParts);
    m_originFace.resize(numParts);
    std::vector<int> vertexMap; // this is shared to save memory
    #pragma omp parallel num_threads(numParts)
    {
    int prt       = omp_get_thread_num(),
        startCell = splitLocation(prt),
        endCell   = splitLocation(prt+1)-1,
        numGroups = faces.groups.size()+1; // +1 for ghost faces
    parts[prt].cells.number = endCell-startCell+1;
    parts[prt].faces.groups.resize(numGroups);
    // copy group code directly for first groups
    for(int i=0; i<numGroups-1; ++i)
        parts[prt].faces.groups[i].first = faces.groups[i].first;
    // last group is of special type GHOST for new boundaries that result from partitioning
    parts[prt].faces.groups.back().first = GHOST;
    std::vector<bool> isAdded;

    // connectivity
    std::vector<int>  originFace; // map from local to unpartitioned, to grab the vertices
    int faceIdx = -1;
    isAdded.resize(faces.number,false);
    for(int C0=startCell; C0<=endCell; ++C0)
      for(auto p : connectivity[C0])
        if(!isAdded[p.second]) {
            ++faceIdx;
            originFace.push_back(p.second);
            isAdded[p.second] = true;

            int C1 = p.first;
            if(C1==-1) // this face belongs to a group
            {
                parts[prt].faces.connectivity.push_back(std::make_pair(C0,-1));

                int grp, idx, nGrp=faces.groups.size();
                for(grp=0; grp<nGrp; ++grp) {
                    idx = findInVector(faces.groups[grp].second,p.second);
                    if(idx!=-1) break;
                }
                assert(grp<nGrp && "could not find face in any group");
                parts[prt].faces.groups[grp].second.push_back(faceIdx);

                // if the face is part of a periodic pair we want to keep it as such
                if(grp==m_PER_1 || grp==m_PER_2) {
                    int matchFace = faces.groups[m_PER_1+m_PER_2-grp].second[idx];
                    int matchCell = faces.connectivity[matchFace].first;
                    assert(matchCell >= startCell &&
                           matchCell <= endCell   && "a periodic pair was broken");
                    ++faceIdx;
                    originFace.push_back(matchFace);
                    isAdded[matchFace] = true;
                    parts[prt].faces.groups[m_PER_1+m_PER_2-grp].second.push_back(faceIdx);
                    parts[prt].faces.connectivity.push_back(std::make_pair(matchCell,-1));
                }
            } else if(C1 < startCell || C1 > endCell) // we have created a ghost
            {
                parts[prt].faces.groups.back().second.push_back(faceIdx);
                parts[prt].faces.connectivity.push_back(std::make_pair(C0,-1));
            } else
                parts[prt].faces.connectivity.push_back(std::make_pair(C0,C1));
        }
    isAdded.clear();
    int numFaces = parts[prt].faces.connectivity.size();
    parts[prt].faces.number = numFaces;

    // local numbering of cells, i.e. "start" at 0
    for(int i=0; i<numFaces; ++i) {
        parts[prt].faces.connectivity[i].first -= startCell;
        if(parts[prt].faces.connectivity[i].second != -1)
           parts[prt].faces.connectivity[i].second -= startCell;
    }

    // sort the connectivity and get the face renumbering function
    {
    std::vector<int> faceSortIdx, faceNewIdx(numFaces);
    faceSortIdx = sortIndexes(parts[prt].faces.connectivity,&comparePair2);
    for(int i=0; i<numFaces; i++)
        faceNewIdx[faceSortIdx[i]] = i;

    std::vector<std::pair<int,int> > tmp1;
    tmp1.swap(parts[prt].faces.connectivity);
    parts[prt].faces.connectivity.reserve(numFaces);
    for(auto i : faceSortIdx) parts[prt].faces.connectivity.push_back(tmp1[i]);

    // renumber the faces in the groups
    for(auto & group : parts[prt].faces.groups)
        for(size_t j=0; j<group.second.size(); j++)
            group.second[j] = faceNewIdx[group.second[j]];

    // reorder the face map so we can start grabbing vertices
    std::vector<int> tmp2;
    tmp2.swap(originFace);
    originFace.reserve(numFaces);
    for(auto i : faceSortIdx) originFace.push_back(tmp2[i]);
    }

    // we need a local vertex structure
    // first we mark which ones we need and count
    int numVertRefs = 0;
    isAdded.resize(vertices.number,false);
    for(auto f : originFace)
        for(int v_p=faces.verticesStart[f]; v_p<faces.verticesStart[f+1]; ++v_p) {
            isAdded[faces.verticesIndex[v_p]] = true;
            ++numVertRefs;
        }
    int numVertices = 0;
    for(auto b : isAdded) numVertices += int(b);

    parts[prt].vertices.number = numVertices;
    parts[prt].vertices.coords.resize(numVertices,NoChange);
    parts[prt].faces.verticesStart.reserve(numFaces+1);
    parts[prt].faces.verticesIndex.reserve(numVertRefs);

    // now copy and map, critical to avoid multiple arrays of size "vertices.number"
    #pragma omp critical
    {
    vertexMap.assign(vertices.number,-1);
    int vertexIdx = -1;
    for(int i=0; i<vertices.number; ++i)
        if(isAdded[i]) {
            ++vertexIdx;
            vertexMap[i] = vertexIdx;
            parts[prt].vertices.coords.row(vertexIdx) = vertices.coords.row(i);
        }

    // finally define local faces in terms of local vertices
    for(int f_glb : originFace) {
        parts[prt].faces.verticesStart.push_back(parts[prt].faces.verticesIndex.size());
        for(int v_p=faces.verticesStart[f_glb]; v_p<faces.verticesStart[f_glb+1]; ++v_p)
        {
            int v_glb = faces.verticesIndex[v_p];
            int v_loc = vertexMap[v_glb];
            assert(v_loc!=-1 && "failed to map vertex from global to local");
            parts[prt].faces.verticesIndex.push_back(v_loc);
        }
    }
    parts[prt].faces.verticesStart.push_back(parts[prt].faces.verticesIndex.size());
    }
    m_originFace[prt] = originFace;
    parts[prt].m_meshIsDefined = true;
    }
    return 0;
}

int UnstructuredMesh::nearNeighbourMap(const UnstructuredMesh& donor,
                                       VectorXi& map, const int searchLevels) const
{
    if(status()+donor.status()!=0) return 1;
    if(searchLevels<1) return 2;

    // initialize map to the "unmapped" state
    map.setConstant(cells.number,-1);

    // cell-based connectivity of the donor mesh
    std::vector<int> donor_outerIdxPtr, donor_cellIdx;
    {
        std::vector<std::vector<int> > connecti_tmp(donor.cells.number);

        for(int i=0; i<donor.faces.number; ++i) {
          int C0 = donor.faces.connectivity[i].first,
              C1 = donor.faces.connectivity[i].second;

          if(C1>-1) {
            connecti_tmp[C0].push_back(C1); connecti_tmp[C1].push_back(C0);
          }
        }
        makeCRSfromLIL(connecti_tmp,donor_outerIdxPtr,donor_cellIdx);
    }

    // static work scheduling
    int numThreads = omp_get_max_threads();
    ArrayXi chunks = ArrayXi::LinSpaced(numThreads+1,0,cells.number);

    #pragma omp parallel num_threads(numThreads)
    {
    int thread = omp_get_thread_num();

    // this thread maps the cells in this closed range:
    int startCell = chunks(thread),
        endCell = chunks(thread+1)-1,
        Nc = endCell-startCell+1;

    // cell-based connectivity for this thread, no out of range connections
    std::vector<int> outerIdxPtr, cellIdx;
    {
        std::vector<std::vector<int> > connecti_tmp(Nc);

        for(int i=0; i<faces.number; ++i) {
          int C0 = faces.connectivity[i].first,
              C1 = faces.connectivity[i].second;

          if(C0>=startCell && C0<=endCell && C1>=startCell && C1<=endCell) {
            connecti_tmp[C0-startCell].push_back(C1);
            connecti_tmp[C1-startCell].push_back(C0);
          }
        }
        makeCRSfromLIL(connecti_tmp,outerIdxPtr,cellIdx);
    }

    // queue holding mapped cells whose neighbours we will map next
    std::queue<int> Q;
    // seed the search process
    Q.push(startCell);
    (donor.cells.centroid.rowwise()-
     cells.centroid.row(startCell)).rowwise().norm().minCoeff(&map(startCell));

    while(!Q.empty())
    {
        int cell = Q.front(), idx = cell-startCell;
        Q.pop();

        // fetch the unmapped neighbours of this cell
        std::vector<int> neighbours;

        for(int i=outerIdxPtr[idx]; i<outerIdxPtr[idx+1]; ++i)
          if(map(cellIdx[i])<0)
            neighbours.push_back(cellIdx[i]);

        if(neighbours.empty()) continue;

        // fetch the neighbourhood of "cell"'s donor
        std::vector<int> candidates(1,map(cell)), lvlStart(1,0);

        for(int lvl=0; lvl<searchLevels; ++lvl) {
          lvlStart.push_back(candidates.size());
          for(int i=lvlStart[lvl]; i<lvlStart[lvl+1]; ++i) {
            int C0 = candidates[i];
            for(int j=donor_outerIdxPtr[C0]; j<donor_outerIdxPtr[C0+1]; ++j)
              candidates.push_back(donor_cellIdx[j]);
          }
        }

        MatrixX3f coords(candidates.size(),3);
        int i=-1;
        for(auto ii : candidates) coords.row(++i) = donor.cells.centroid.row(ii);

        // map the neighbours and add them to the queue
        for(std::vector<int>::iterator it=neighbours.begin(); it!=neighbours.end(); ++it)
        {
          int closest;
          (coords.rowwise()-cells.centroid.row(*it)).rowwise().norm().minCoeff(&closest);
          map(*it) = candidates[closest];
          Q.push(*it);
        }
    }
    } //end parallel

    return 0;
}

int UnstructuredMesh::colorEdges(VectorXi& colorStart, VectorXi& cellIdx, const int groupSize) const
{
    if(!m_meshIsDefined) return 1;
    if(groupSize < 1) return 2;

    using std::vector;
    vector<int8_t> edgeColor(faces.number,-1);
    vector<int> colorSize(1,0);
    int color, nColor=1;

    {
    // for each color keep track of the column indices that are in it
    vector<vector<bool> > cellInColor(1,vector<bool>(cells.number,false));

    for(int edge=0; edge<faces.number; edge+=groupSize)
    {
        int localSize = std::min(groupSize,faces.number-edge);

        for(color=0; color<nColor; ++color)
        {
            bool free = true;
            for(int j=0; j<localSize && free; ++j)
            {
                int C0 = faces.connectivity[edge+j].first,
                    C1 = faces.connectivity[edge+j].second;
                if(C1<0) C1=C0;

                free = !cellInColor[color][C0] && !cellInColor[color][C1];
            }
            if(free) break;
        }

        if(color==nColor) // no color was free, make space for a new one
        {
            ++nColor;
            colorSize.push_back(0);
            cellInColor.push_back(vector<bool>(cells.number,false));
        }

        for(int j=0; j<localSize; ++j)
        {
            int C0 = faces.connectivity[edge+j].first,
                C1 = faces.connectivity[edge+j].second;
            if(C1<0) C1=C0;

            edgeColor[edge+j] = color;
            cellInColor[color][C0] = true;
            cellInColor[color][C1] = true;
        }
        colorSize[color] += localSize;
    }
    }

    // convert coloring into a CSR storage system
    colorStart.setZero(nColor+1);
    cellIdx.resize(faces.number);

    int idx=-1;
    for(color=0; color<nColor; ++color)
    {
        colorStart(color+1) = colorStart(color)+colorSize[color];

        for(int edge=0; edge<faces.number; edge+=groupSize)
            if(edgeColor[edge]==color)
                for(int j=0; j<groupSize && (edge+j)<faces.number; ++j)
                    cellIdx(++idx) = edge+j;
    }

    #ifdef MESH_VERBOSE
    #pragma omp critical
    {
        std::cout << "Number of edge colors: " << nColor << '\n';
        for(int i=0; i<nColor; ++i)
            std::cout << "Number of edges: " << colorSize[i] << '\n';
        std::cout << std::flush;
    }
    #endif

    return 0;
}

int UnstructuredMesh::runsChecks()
{
    MatrixX3f norm = MatrixX3f::Zero(cells.number,3);
    MatrixX3f sum = norm;

    for(int i=0; i<faces.number; ++i)
    {
        int C0 = faces.connectivity[i].first,
            C1 = faces.connectivity[i].second;

        sum.row(C0) += faces.area.row(i);
        norm.row(C0)  += faces.area.row(i).cwiseAbs();

        if(C1!=-1)
        {
            sum.row(C1) -= faces.area.row(i);
            norm.row(C1)  += faces.area.row(i).cwiseAbs();
        }
    }

    VectorXf err = sum.rowwise().norm();
    err = err.cwiseQuotient(norm.rowwise().norm());

    std::cout << err.rows() << "  " << err.maxCoeff() << std::endl;

    return 0;
}

}
