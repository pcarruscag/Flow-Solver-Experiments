//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#ifndef GEOMETRICPROPERTIES_H
#define GEOMETRICPROPERTIES_H

#include "../../mathUtils/src/matrix.h"
#include <Eigen/Dense>
#include <vector>
#include <utility>

//#define USEACCURATEPROPERTIES

using namespace Eigen;
using namespace mathUtils::matrix;
/*
Compute the geometric properties of an unstructured mesh.
  This is separate code to avoid duplication, it is used by UnstructuredMesh and by the
  adjoint solver. It is a macro because making it a function has a significant negative
  impact on the geometric jacobian compute time. Having the code inlined, by force, must
  allow some optimization not possible otherwise (g++ 7.3).

  Defining USEACCURATEPROPERTIES computes the centroids of non-simplex faces and cells
  as area/volume averages of true centroids, instead of using vertex centroids.
  The difference is only significant if those faces/cells are tapered.
  Computing the geometric jacobian is about 2.4 times more expensive.
*/
#ifndef USEACCURATEPROPERTIES
#define GEOMETRICPROPERTIES(Scalar, Nc, Nf,                                             \
                            connectivity,    verticesStart,  verticesIndex,             \
                            vertices_coords, faces_centroid, faces_area,                \
                            faces_r0, faces_r1, faces_alfa0, faces_alfa1,               \
                            faces_wf,        cells_centroid, cells_volume)              \
/* index by index initialization needed for correct derivative computation */           \
for(int i=0; i<Nc; ++i) {for(int j=0; j<3; ++j) cells_centroid(i,j) = 0.0;}             \
for(int i=0; i<Nc; ++i) cells_volume(i) = 0.0;                                          \
                                                                                        \
/* face centroids and normals */                                                        \
for(int i=0; i<Nf; i++)                                                                 \
{                                                                                       \
    int j = verticesStart[i],                                                           \
        nFaceVertex = verticesStart[i+1]-j;                                             \
    if(nFaceVertex == 3) {                                                              \
      faces_centroid.row(i) = 0.33333333*(vertices_coords.row(verticesIndex[ j ])+      \
                                          vertices_coords.row(verticesIndex[j+1])+      \
                                          vertices_coords.row(verticesIndex[j+2]));     \
      faces_area.row(i) = 0.5*(vertices_coords.row(verticesIndex[j+1])-                 \
                               vertices_coords.row(verticesIndex[ j ])).cross(          \
                               vertices_coords.row(verticesIndex[j+2])-                 \
                               vertices_coords.row(verticesIndex[ j ]));                \
    }                                                                                   \
    else if(nFaceVertex == 4) {                                                         \
      faces_centroid.row(i) = 0.25*(vertices_coords.row(verticesIndex[ j ])+            \
                                    vertices_coords.row(verticesIndex[j+1])+            \
                                    vertices_coords.row(verticesIndex[j+2])+            \
                                    vertices_coords.row(verticesIndex[j+3]));           \
      faces_area.row(i) = 0.5*(vertices_coords.row(verticesIndex[j+2])-                 \
                               vertices_coords.row(verticesIndex[ j ])).cross(          \
                               vertices_coords.row(verticesIndex[j+3])-                 \
                               vertices_coords.row(verticesIndex[j+1]));                \
    } else return 2;                                                                    \
}                                                                                       \
                                                                                        \
/* vertex centroid of the cells */                                                      \
VectorXi faceCount = VectorXi::Zero(Nc);                                                \
for(int i=0; i<Nf; i++)                                                                 \
{                                                                                       \
    int C0 = connectivity[i].first;                                                     \
    cells_centroid.row(C0) += faces_centroid.row(i);                                    \
    faceCount(C0)++;                                                                    \
                                                                                        \
    int C1 = connectivity[i].second;                                                    \
    if(C1!=-1){                                                                         \
    cells_centroid.row(C1) += faces_centroid.row(i);                                    \
    faceCount(C1)++;}                                                                   \
}                                                                                       \
for(int i=0; i<Nc; i++)                                                                 \
    cells_centroid.row(i) /= faceCount(i);                                              \
                                                                                        \
for(int i=0; i<Nf; i++)                                                                 \
{                                                                                       \
 /* normalization required to prevent underflow of the derivatives when computing       \
    "alfa" in single precision, which could happen due to |A|^4 being involved */       \
    double normAf = 0.0;                                                                \
    for(int j=0; j<3; ++j)                                                              \
        normAf += static_cast<double>(getVal(faces_area(i,j))*getVal(faces_area(i,j))); \
    normAf = 1.0/std::sqrt(normAf);                                                     \
    Matrix<Scalar,1,3> face_n = normAf*faces_area.row(i);                               \
    Scalar AdotN = face_n.dot(faces_area.row(i));                                       \
                                                                                        \
    int C0 = connectivity[i].first;                                                     \
    faces_r0.row(i) = faces_centroid.row(i)-cells_centroid.row(C0);                     \
    Scalar volFromFace = faces_r0.row(i).dot(faces_area.row(i))/3.0f;                   \
    if(volFromFace < 0.0f) {                                                            \
        faces_area.row(i) *= -1.0f;                                                     \
        volFromFace *= -1.0f;                                                           \
        face_n *= -1.0f;                                                                \
    }                                                                                   \
    faces_alfa0(i) = face_n.dot(faces_r0.row(i))/AdotN;                                 \
    cells_volume(C0) += volFromFace;                                                    \
                                                                                        \
    int C1 = connectivity[i].second;                                                    \
    if(C1!=-1){                                                                         \
        faces_r1.row(i) = faces_centroid.row(i)-cells_centroid.row(C1);                 \
        volFromFace = -faces_r1.row(i).dot(faces_area.row(i))/3.0f;                     \
        if(volFromFace < 0.0f)                                                          \
            return 3;                                                                   \
        faces_alfa1(i) = face_n.dot(faces_r1.row(i))/AdotN;                             \
        faces_wf(i) = 1.0f/(1.0f+faces_r0.row(i).norm()/faces_r1.row(i).norm());        \
        cells_volume(C1) += volFromFace;                                                \
    }else{                                                                              \
        faces_r1(i,0) = 0.0; faces_r1(i,1) = 0.0; faces_r1(i,2) = 0.0;                  \
        faces_alfa1(i) = 0.0;                                                           \
        faces_wf(i) = 1.0;                                                              \
    }                                                                                   \
}
#else
#define GEOMETRICPROPERTIES(Scalar, Nc, Nf,                                             \
                            connectivity,    verticesStart,  verticesIndex,             \
                            vertices_coords, faces_centroid, faces_area,                \
                            faces_r0, faces_r1, faces_alfa0, faces_alfa1,               \
                            faces_wf,        cells_centroid, cells_volume)              \
/* index by index initialization needed for correct derivative computation */           \
for(int i=0; i<Nc; ++i) cells_volume(i) = 0.0;                                          \
                                                                                        \
/* face centroids and normals */                                                        \
for(int i=0; i<Nf; i++)                                                                 \
{                                                                                       \
    int j = verticesStart[i],                                                           \
        nFaceVertex = verticesStart[i+1]-j;                                             \
                                                                                        \
    Matrix<Scalar,4,3> Pi;                                                              \
    for(int k=0; k<3; ++k) Pi.row(k) = vertices_coords.row(verticesIndex[j+k]);         \
                                                                                        \
    if(nFaceVertex == 3) {                                                              \
      /* this is exact for triangles */                                                 \
      faces_centroid.row(i) = 0.33333333*(Pi.row(0)+Pi.row(1)+Pi.row(2));               \
      faces_area.row(i) = 0.5*(Pi.row(1)-Pi.row(0)).cross(Pi.row(2)-Pi.row(0));         \
    }                                                                                   \
    else if(nFaceVertex == 4) {                                                         \
      /* quadrilaterals are decomposed in 4 triangles, for consistency */               \
      Pi.row(3) = vertices_coords.row(verticesIndex[j+3]);                              \
      Matrix<Scalar,4,3> Ai;                                                            \
      Scalar normAf = 0.0;                                                              \
      /* vertex centroid used for decomposition, non-planar faces are sensitive to it */\
      Matrix<Scalar,1,3> Cv = 0.25*(Pi.row(0)+Pi.row(1)+Pi.row(2)+Pi.row(3)),           \
                         Cf; Cf(0) = 0.0; Cf(1) = 0.0; Cf(2) = 0.0;                     \
      for(int k=0; k<4; ++k) {                                                          \
        Ai.row(k) = 0.5*(Cv-Pi.row(k)).cross(Pi.row((k+1)%4)-Pi.row(k));                \
        Scalar normAi = Ai.row(k).norm(); normAf+=normAi;                               \
        /* Cv is factored out and added later */                                        \
        Cf += normAi*0.33333333*(Pi.row(k)+Pi.row((k+1)%4));                            \
      }                                                                                 \
      faces_area.row(i) = Ai.colwise().sum();                                           \
      faces_centroid.row(i) = Cf/normAf+0.33333333*Cv;                                  \
    }                                                                                   \
    else return 2;                                                                      \
}                                                                                       \
                                                                                        \
/* approximate centroid of the cells, used to decompose them into pyramids, these       \
   do not need to be differentiated as the final result is insensitive to them */       \
MatrixX3f pseudoCentroid = MatrixX3f::Zero(Nc,3);                                       \
VectorXf faceCount = VectorXf::Zero(Nc);                                                \
for(int i=0; i<Nf; i++)                                                                 \
{                                                                                       \
    int C0 = connectivity[i].first;                                                     \
    for(int j=0; j<3; ++j) pseudoCentroid(C0,j) += getVal(faces_centroid(i,j));         \
    faceCount(C0) += 1.0;                                                               \
                                                                                        \
    int C1 = connectivity[i].second;                                                    \
    if(C1!=-1){                                                                         \
    for(int j=0; j<3; ++j) pseudoCentroid(C1,j) += getVal(faces_centroid(i,j));         \
    faceCount(C1) += 1.0;}                                                              \
}                                                                                       \
for(int i=0; i<Nc; i++)                                                                 \
    pseudoCentroid.row(i) /= faceCount(i);                                              \
                                                                                        \
/* each face makes a contribution to the volume and to the centroid */                  \
Matrix<Scalar,Dynamic,3> volIntCentroid(Nc,3);                                          \
for(int i=0; i<Nc; ++i) {for(int j=0; j<3; ++j) volIntCentroid(i,j) = 0.0;}             \
                                                                                        \
for(int i=0; i<Nf; ++i)                                                                 \
{                                                                                       \
    /* volume, height, and centroid of the pyramid(s) associated with face "i" */       \
    Scalar volFromFace;                                                                 \
    Matrix<Scalar,1,3> height, Cpyr;                                                    \
                                                                                        \
    int C0 = connectivity[i].first;                                                     \
    height = faces_centroid.row(i);                                                     \
    for(int j=0; j<3; ++j) height(j) -= pseudoCentroid(C0,j);                           \
    Cpyr = 0.75*faces_centroid.row(i);                                                  \
    for(int j=0; j<3; ++j) Cpyr(j) += 0.25*pseudoCentroid(C0,j);                        \
    volFromFace = 0.33333333*height.dot(faces_area.row(i));                             \
    if(volFromFace < 0.0) {faces_area.row(i)*=-1.0; volFromFace*=-1.0;}                 \
    cells_volume(C0) += volFromFace;                                                    \
    volIntCentroid.row(C0) += volFromFace*Cpyr;                                         \
                                                                                        \
    int C1 = connectivity[i].second;                                                    \
    if (C1!=-1){                                                                        \
    height = faces_centroid.row(i);                                                     \
    for(int j=0; j<3; ++j) height(j) -= pseudoCentroid(C1,j);                           \
    Cpyr = 0.75*faces_centroid.row(i);                                                  \
    for(int j=0; j<3; ++j) Cpyr(j) += 0.25*pseudoCentroid(C1,j);                        \
    volFromFace = -0.33333333*height.dot(faces_area.row(i));                            \
    if(volFromFace < 0.0) return 3;                                                     \
    cells_volume(C1) += volFromFace;                                                    \
    volIntCentroid.row(C1) += volFromFace*Cpyr;                                         \
    }                                                                                   \
}                                                                                       \
for(int i=0; i<Nc; ++i)                                                                 \
    cells_centroid.row(i) = volIntCentroid.row(i)/cells_volume(i);                      \
                                                                                        \
/* compute the vectors used for gradient projection */                                  \
for(int i=0; i<Nf; i++)                                                                 \
{                                                                                       \
 /* normalization required to prevent underflow of the derivatives when computing       \
    "alfa" in single precision, which could happen due to |A|^4 being involved */       \
    double normAf = 0.0;                                                                \
    for(int j=0; j<3; ++j)                                                              \
        normAf += static_cast<double>(getVal(faces_area(i,j))*getVal(faces_area(i,j))); \
    normAf = 1.0/std::sqrt(normAf);                                                     \
    Matrix<Scalar,1,3> face_n = normAf*faces_area.row(i);                               \
    Scalar AdotN = face_n.dot(faces_area.row(i));                                       \
                                                                                        \
    int C0 = connectivity[i].first;                                                     \
      faces_r0.row(i) = faces_centroid.row(i)-cells_centroid.row(C0);                   \
      faces_alfa0(i) = face_n.dot(faces_r0.row(i))/AdotN;                               \
                                                                                        \
    int C1 = connectivity[i].second;                                                    \
    if (C1!=-1){                                                                        \
      faces_r1.row(i) = faces_centroid.row(i)-cells_centroid.row(C1);                   \
      faces_alfa1(i) = face_n.dot(faces_r1.row(i))/AdotN;                               \
      faces_wf(i) = 1.0/(1.0+faces_r0.row(i).norm()/faces_r1.row(i).norm());            \
    }else{                                                                              \
      faces_r1(i,0) = 0.0; faces_r1(i,1) = 0.0; faces_r1(i,2) = 0.0;                    \
      faces_alfa1(i) = 0.0;                                                             \
      faces_wf(i) = 1.0;                                                                \
    }                                                                                   \
} // GEOMETRICPROPERTIES
#endif // USEACCURATEPROPERTIES

#endif // GEOMETRICPROPERTIES_H
