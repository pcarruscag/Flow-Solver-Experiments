//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef UNSTRUCTUREDMESH_H
#define UNSTRUCTUREDMESH_H

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

namespace flow
{
    class PressureBasedCoupledSolver;
    class PressureBasedSegregatedSolver;
    void _flowTest(const std::string caseNum, const bool extraCalculations);
    void _adjointInputTest();
}

namespace mesh
{
    class PassageMesh;

    class UnstructuredMesh
    {
        friend class PassageMesh;
        friend void _meshTests(const bool expensiveTests=false);
        friend void flow::_flowTest(const std::string caseNum, const bool extraCalculations);
        friend void flow::_adjointInputTest();
        friend class flow::PressureBasedCoupledSolver;
        friend class flow::PressureBasedSegregatedSolver;
      private:
        struct m_cells
        {
            int number;
            VectorXf  volume;
            MatrixX3f centroid;
        };

        struct m_faces
        {
            // Variables that define the mesh
            int number;
            std::vector<int> verticesStart;
            std::vector<int> verticesIndex;
            std::vector<std::pair<int,int> > connectivity;
            std::vector<std::pair<int,std::vector<int> > > groups;
            // Geometric properties
            MatrixX3f area;  MatrixX3f centroid;
            VectorXf  alfa0; VectorXf  alfa1;
            MatrixX3f r0;    MatrixX3f r1;
            MatrixX3f d0;    MatrixX3f d1;
            VectorXf  wf;
        };

        struct m_vertices
        {
            int number;
            MatrixX3f coords;
        };

        m_cells cells;
        m_faces faces;
        m_vertices vertices;

        // a map connecting the faces of partitions
        // to the faces of the un-partitioned domain
        // used when building the adjoint input
        std::vector<std::vector<int> > m_originFace;

        bool m_meshIsDefined;
        bool m_geoPropsComputed;
        bool m_meshIsOrdered;

        // indices of the periodic groups (not the code!)
        int m_PER_1, m_PER_2;

      public:
        // code given to group holding ghost faces
        static constexpr int GHOST{100};

        UnstructuredMesh();

        inline void setPeriodicPair(const int group1, const int group2) {
            m_PER_1 = group1; m_PER_2 = group2;
        }
        int scale(const float factor);
        int renumberCells(const VectorXi& perm);
        int rcmOrdering(VectorXi& perm,
                        const bool includePeriodics=false,
                        const bool clusterPeriodics=false) const;
        int computeGeoProps();
        int partitioner(const int numParts,
                        std::vector<UnstructuredMesh>& parts,
                        const bool keepOrder = false);
        // map the cells of this to those of the donor based on distance
        int nearNeighbourMap(const UnstructuredMesh& donor,
                             VectorXi& map,
                             const int searchLevels=3) const;
        // greedy edge coloring
        int colorEdges(VectorXi& colorStart,
                       VectorXi& cellIdx,
                       const int groupSize=1) const;
        float volume() {return cells.volume.sum();}
        int runsChecks();

        int status() const;
    };
}


#endif // UNSTRUCTUREDMESH_H
