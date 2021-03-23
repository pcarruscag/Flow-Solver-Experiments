//  Copyright (C) 2018-2021  Pedro Gomes
//  See full notice in NOTICE.md

#include "meshTests.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

namespace mesh
{
void testSuite(const bool expensiveTests)
{
    std::cout << std::endl;
    _meshTests(expensiveTests);
}

void _meshTests(const bool expensiveTests)
{
    std::cout << "### MESH GENERATION TESTS ###" << std::endl;

    fileManagement::GeometryParamManager geoParams;
    fileManagement::MeshParamManager mshParams;
    geometryGeneration::BladeGeometry blade;
    PassageMesh mesh;
    PassageMeshInspector meshInspector(&mesh);

    std::ifstream baseline, current;

    std::string geoParFile = "../../lib/mesh/tests/geoParFile.txt";
    std::string geoParFile2 = "../../lib/mesh/tests/geoParFile2.txt";
    std::string mshParFile = "../../lib/mesh/tests/mshParFile.txt";

    std::string msh2Dfile = "../../lib/mesh/tests/msh2D.txt";
    std::string msh2DderivFile = "../../lib/mesh/tests/msh2Dderiv.dat";
    std::string msh3Dfile = "../../lib/mesh/tests/msh3D.txt";
    std::string msh3Dfile2 = "../../lib/mesh/tests/msh3D2.txt";
    std::string msh3DderivFile = "../../lib/mesh/tests/msh3Dderiv.dat";

    int errorCode;
    errorCode = mshParams.readFile(mshParFile);
    assert(errorCode == 0);

    // Meshing
    errorCode = geoParams.readFile(geoParFile);
    assert(errorCode == 0);
    errorCode = blade.buildGeometry(geoParams,mshParams);
    assert(errorCode == 0);

    errorCode = mesh.mesh(blade,mshParams);
    assert(errorCode == 0);

    assert(meshInspector.save2Dlayers("./tmp.txt") == 0);
    baseline.open(msh2Dfile);
    current.open("./tmp.txt");
    double diff = 0.0;
    double refVal = 0.0;
    for(int span=0; span<3; ++span) {
        // consume span section
        std::string line;
        getline(baseline,line); getline(current,line);
        for(int coord=0; coord<2; ++coord) {
            // consume coordinate section
            getline(baseline,line); getline(current,line);
            for(int i=0; i<30; ++i) {
                // read values
                getline(baseline,line); std::stringstream ss1(line);
                getline(current ,line); std::stringstream ss2(line);
                double v1, v2;
                while(ss1 >> v1 && ss2 >> v2) {
                    diff = std::max(diff,std::abs(v2-v1));
                    refVal = std::max(refVal,std::abs(v1));
                }
            }
        }
    }
    assert(diff/refVal < 1e-5);
    baseline.close(); current.close();

//    assert(meshInspector.save2DlayerJacobian(msh2DderivFile) == 0);
    assert(meshInspector.comp2DlayerJacobian(msh2DderivFile,1e-5) == 0);

    assert(meshInspector.save3Dgrid("./tmp.txt") == 0);
    baseline.open(msh3Dfile);
    current.open("./tmp.txt");
    diff = 0.0;
    refVal = 0.0;
    for(int i=0; i<27000; ++i) {
        std::string line;
        getline(baseline,line); std::stringstream ss1(line); double v1;
        getline(current ,line); std::stringstream ss2(line); double v2;
        while(ss1 >> v1 && ss2 >> v2) {
            diff = std::max(diff,std::abs(v2-v1));
            refVal = std::max(refVal,std::abs(v1));
        }
    }
    assert(diff/refVal < 1e-5);
    baseline.close(); current.close();

//    assert(meshInspector.save3DgridJacobian(msh3DderivFile) == 0);
    assert(meshInspector.comp3DgridJacobian(msh3DderivFile,1e-5) == 0);

    if(expensiveTests)
    {
        meshInspector.finDiffVerification(geoParams,mshParams,0.001);
        meshInspector.finDiffVerifBladeGeo(geoParams,mshParams,0.1,0.001);
    }

    // Morphing
    errorCode = geoParams.readFile(geoParFile2);
    assert(errorCode == 0);
    errorCode = blade.buildGeometry(geoParams,mshParams,true);
    assert(errorCode == 0);

    errorCode = mesh.morph(blade);
    assert(errorCode == 0);

    assert(meshInspector.save3Dgrid("./tmp.txt") == 0);
    baseline.open(msh3Dfile2);
    current.open("./tmp.txt");
    diff = 0.0;
    refVal = 0.0;
    for(int i=0; i<27000; ++i) {
        std::string line;
        getline(baseline,line); std::stringstream ss1(line); double v1;
        getline(current ,line); std::stringstream ss2(line); double v2;
        while(ss1 >> v1 && ss2 >> v2) {
            diff = std::max(diff,std::abs(v2-v1));
            refVal = std::max(refVal,std::abs(v1));
        }
    }
    assert(diff/refVal < 1e-5);
    baseline.close(); current.close();

    // Unstructured conversion
    UnstructuredMesh master;
    std::vector<UnstructuredMesh> domains;
    assert(mesh.convertToUnstructured(master) == 0);
    assert(master.partitioner(4,domains) == 0);

    // Geometric parameter calculation
    #pragma omp parallel for reduction(+:errorCode)
    for(int i=0; i<4; i++)
        errorCode += domains[i].computeGeoProps();
    assert(errorCode == 0);

    // rcm ordering and cell renumbering
    VectorXf vol1(domains[0].cells.number);
    vol1 = domains[0].cells.volume;
    VectorXi perm;
    domains[0].rcmOrdering(perm,true);
    domains[0].renumberCells(perm);
    assert(domains[0].computeGeoProps() == 0);
    for(int i=0; i<perm.rows(); i++)
        vol1(i) = std::abs(vol1(i)-domains[0].cells.volume(perm(i)))/vol1(i);
    assert(vol1.maxCoeff()<1e-6f);

    std::string renumFile = "../../lib/mesh/tests/renumbering.txt";
    std::ofstream file;  file.open(renumFile);  file << perm;  file.close();

    std::remove("./tmp.txt");
}

int PassageMeshInspector::finDiffVerifBladeGeo(fileManagement::GeometryParamManager& geoParams,
                                               const fileManagement::MeshParamManager& mshParams,
                                               const double dx1, const double dx2)
{
    std::cout << "\n### Finite Difference Verification of Blade Geometry Generation ###" << std::endl;

    geometryGeneration::BladeGeometry blade;
    assert(blade.buildGeometry(geoParams,mshParams) == 0);

    int Nvars = blade.m_nIndepVars, Nspan = geoParams.m_nSpanSec, currentVar = 0;

    std::vector<MatrixXd> dX, dY;
    dX.resize(Nspan);
    dY.resize(Nspan);
    for(int i=0; i<Nspan; i++)
    {
        dX[i].resize(2*blade.m_nPtMeri,Nvars);
        dY[i].resize(2*blade.m_nPtMeri,Nvars);
        for(int j=0; j<Nvars; j++)
        {
            dX[i].col(j) = -blade.m_meriCoord[i];
            dY[i].col(j) = -blade.m_pitchCoord[i];
        }
    }
    for(int i=0; i<geoParams.m_nPtHub; i++)
    {
        // x coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_xHubPt[i] += dx1;
        assert(blade.buildGeometry(geoParams,mshParams) == 0);
        geoParams.m_xHubPt[i] -= dx1;
        for(int j=0; j<Nspan; j++) {
            dX[j].col(currentVar) += blade.m_meriCoord[j];  dX[j].col(currentVar) /= dx1;
            dY[j].col(currentVar) += blade.m_pitchCoord[j]; dY[j].col(currentVar) /= dx1;
        }
        currentVar++;
        // y coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yHubPt[i] += dx1;
        assert(blade.buildGeometry(geoParams,mshParams) == 0);
        geoParams.m_yHubPt[i] -= dx1;
        for(int j=0; j<Nspan; j++) {
            dX[j].col(currentVar) += blade.m_meriCoord[j];  dX[j].col(currentVar) /= dx1;
            dY[j].col(currentVar) += blade.m_pitchCoord[j]; dY[j].col(currentVar) /= dx1;
        }
        currentVar++;
    }
    for(int i=0; i<geoParams.m_nPtShr; i++)
    {
        // x coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_xShrPt[i] += dx1;
        assert(blade.buildGeometry(geoParams,mshParams) == 0);
        geoParams.m_xShrPt[i] -= dx1;
        for(int j=0; j<Nspan; j++) {
            dX[j].col(currentVar) += blade.m_meriCoord[j];  dX[j].col(currentVar) /= dx1;
            dY[j].col(currentVar) += blade.m_pitchCoord[j]; dY[j].col(currentVar) /= dx1;
        }
        currentVar++;
        // y coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yShrPt[i] += dx1;
        assert(blade.buildGeometry(geoParams,mshParams) == 0);
        geoParams.m_yShrPt[i] -= dx1;
        for(int j=0; j<Nspan; j++) {
            dX[j].col(currentVar) += blade.m_meriCoord[j];  dX[j].col(currentVar) /= dx1;
            dY[j].col(currentVar) += blade.m_pitchCoord[j]; dY[j].col(currentVar) /= dx1;
        }
        currentVar++;
    }
    for(int i=0; i<Nspan; i++)
    {
        // le
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yLePt[i] += dx2;
        assert(blade.buildGeometry(geoParams,mshParams) == 0);
        geoParams.m_yLePt[i] -= dx2;
        for(int j=0; j<Nspan; j++) {
            dX[j].col(currentVar) += blade.m_meriCoord[j];  dX[j].col(currentVar) /= dx2;
            dY[j].col(currentVar) += blade.m_pitchCoord[j]; dY[j].col(currentVar) /= dx2;
        }
        currentVar++;
        // te
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yTePt[i] += dx2;
        assert(blade.buildGeometry(geoParams,mshParams) == 0);
        geoParams.m_yTePt[i] -= dx2;
        for(int j=0; j<Nspan; j++) {
            dX[j].col(currentVar) += blade.m_meriCoord[j];  dX[j].col(currentVar) /= dx2;
            dY[j].col(currentVar) += blade.m_pitchCoord[j]; dY[j].col(currentVar) /= dx2;
        }
        currentVar++;
    }
    for(int i=0; i<Nspan; i++)
    {
        for(int j=0; j<geoParams.m_nPtTheta; j++)
        {
            std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
            geoParams.m_yThetaPt[i][j] += dx2;
            assert(blade.buildGeometry(geoParams,mshParams) == 0);
            geoParams.m_yThetaPt[i][j] -= dx2;
            for(int k=0; k<Nspan; k++) {
                dX[k].col(currentVar) += blade.m_meriCoord[k];  dX[k].col(currentVar) /= dx2;
                dY[k].col(currentVar) += blade.m_pitchCoord[k]; dY[k].col(currentVar) /= dx2;
        }
            currentVar++;
        }
    }
    std::cout << "\nComparison with AD Jacobian" << std::endl;

    assert(blade.buildGeometry(geoParams,mshParams) == 0);
    for(int i=0; i<Nspan; i++) {
        dX[i] -= blade.m_meriCoordDeriv[i];
        dY[i] -= blade.m_pitchCoordDeriv[i];
        for(int j=0; j<dX[i].rows(); j++){
            for(int k=0; k<dX[i].cols(); k++){
                dX[i](j,k) = std::abs(dX[i](j,k));
                dY[i](j,k) = std::abs(dY[i](j,k));
            }
        }
    }
    int row, col;
    std::cout << "Max difference, AD derivative at point, Max AD derivative for variable" << std::endl;
    for(int j=0; j<Nspan; j++) {
        std::cout << "Layer " << j << std::endl;
        for(int i=0; i<Nvars; i++) {
            std::cout << "Variable " << i << ": Meri. coord.: ";
            std::cout << dX[j].col(i).maxCoeff(&row,&col) << "\t-> ";
            std::cout << blade.m_meriCoordDeriv[j](row,i) << "\t-> ";
            std::cout << std::max(std::abs(blade.m_meriCoordDeriv[j].col(i).minCoeff()),
                                  std::abs(blade.m_meriCoordDeriv[j].col(i).maxCoeff()));
            std::cout << "\tPitch coord.: ";
            std::cout << dY[j].col(i).maxCoeff(&row,&col) << "\t-> ";
            std::cout << blade.m_pitchCoordDeriv[j](row,i) << "\t-> ";
            std::cout << std::max(std::abs(blade.m_pitchCoordDeriv[j].col(i).minCoeff()),
                                  std::abs(blade.m_pitchCoordDeriv[j].col(i).maxCoeff()));
            std::cout << std::endl;
        }
    }
    std::cout << "\n### Finished ###" << std::endl;
    return 0;
}

int PassageMeshInspector::finDiffVerification(fileManagement::GeometryParamManager& geoParams,
                                              const fileManagement::MeshParamManager& mshParams,
                                              const double dx)
{
    std::cout << "\n### Finite Difference Verification of Grid Jacobian ###" << std::endl;

    int Nvertex = m_obj->m_vertexCoords.rows(),
        Nvars   = m_obj->m_xCoordJacobian.cols();
    MatrixXf X0(Nvertex,3), dX0(3*Nvertex,Nvars);
    X0 = m_obj->m_vertexCoords;
    X0.resize(3*Nvertex,1);
    for(int i=0; i<Nvars; i++)
        dX0.col(i) = -X0;

    geometryGeneration::BladeGeometry blade;
    int currentVar = 0;

    for(int i=0; i<geoParams.m_nPtHub; i++)
    {
        // x coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_xHubPt[i] += dx;
        blade.buildGeometry(geoParams,mshParams);
        geoParams.m_xHubPt[i] -= dx;
        assert(m_obj->morph(blade) == 0);
        X0.resize(Nvertex,3);
        X0 = m_obj->m_vertexCoords;
        X0.resize(3*Nvertex,1);
        dX0.col(currentVar) += X0;
        currentVar++;
        // y coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yHubPt[i] += dx;
        blade.buildGeometry(geoParams,mshParams);
        geoParams.m_yHubPt[i] -= dx;
        assert(m_obj->morph(blade) == 0);
        X0.resize(Nvertex,3);
        X0 = m_obj->m_vertexCoords;
        X0.resize(3*Nvertex,1);
        dX0.col(currentVar) += X0;
        currentVar++;
    }
    for(int i=0; i<geoParams.m_nPtShr; i++)
    {
        // x coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_xShrPt[i] += dx;
        blade.buildGeometry(geoParams,mshParams);
        geoParams.m_xShrPt[i] -= dx;
        assert(m_obj->morph(blade) == 0);
        X0.resize(Nvertex,3);
        X0 = m_obj->m_vertexCoords;
        X0.resize(3*Nvertex,1);
        dX0.col(currentVar) += X0;
        currentVar++;
        // y coordinate
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yShrPt[i] += dx;
        blade.buildGeometry(geoParams,mshParams);
        geoParams.m_yShrPt[i] -= dx;
        assert(m_obj->morph(blade) == 0);
        X0.resize(Nvertex,3);
        X0 = m_obj->m_vertexCoords;
        X0.resize(3*Nvertex,1);
        dX0.col(currentVar) += X0;
        currentVar++;
    }
    for(int i=0; i<geoParams.m_nSpanSec; i++)
    {
        // le
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yLePt[i] += dx;
        blade.buildGeometry(geoParams,mshParams);
        geoParams.m_yLePt[i] -= dx;
        assert(m_obj->morph(blade) == 0);
        X0.resize(Nvertex,3);
        X0 = m_obj->m_vertexCoords;
        X0.resize(3*Nvertex,1);
        dX0.col(currentVar) += X0;
        currentVar++;
        // te
        std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
        geoParams.m_yTePt[i] += dx;
        blade.buildGeometry(geoParams,mshParams);
        geoParams.m_yTePt[i] -= dx;
        assert(m_obj->morph(blade) == 0);
        X0.resize(Nvertex,3);
        X0 = m_obj->m_vertexCoords;
        X0.resize(3*Nvertex,1);
        dX0.col(currentVar) += X0;
        currentVar++;
    }
    for(int i=0; i<geoParams.m_nSpanSec; i++)
    {
        for(int j=0; j<geoParams.m_nPtTheta; j++)
        {
            std::cout << "\nVariable: " << currentVar+1 << "\\" << Nvars << std::endl;
            geoParams.m_yThetaPt[i][j] += dx;
            blade.buildGeometry(geoParams,mshParams);
            geoParams.m_yThetaPt[i][j] -= dx;
            assert(m_obj->morph(blade) == 0);
            X0.resize(Nvertex,3);
            X0 = m_obj->m_vertexCoords;
            X0.resize(3*Nvertex,1);
            dX0.col(currentVar) += X0;
            currentVar++;
        }
    }
    std::cout << "\nComparison with AD Jacobian" << std::endl;
    std::cout << "Max difference, AD derivative at point, ||difference|| over ||AD derivatives||:" << std::endl;
    dX0 /= float(dx);
    MatrixXf dX0norms(1,Nvars);
    for(int i=0; i<Nvars; i++)
        dX0norms(0,i) = dX0.col(i).norm();
    dX0.block(    0    ,0,Nvertex,Nvars) -= m_obj->m_xCoordJacobian;
    dX0.block( Nvertex ,0,Nvertex,Nvars) -= m_obj->m_yCoordJacobian;
    dX0.block(2*Nvertex,0,Nvertex,Nvars) -= m_obj->m_zCoordJacobian;
    for(int j=0; j<dX0.cols(); j++)
        for(int i=0; i<dX0.rows(); i++)
            dX0(i,j) = std::abs(dX0(i,j));

    int row, col;
    for(int i=0; i<Nvars; i++)
    {
        std::cout << "Variable " << i << ":\t-> " <<
        dX0.col(i).maxCoeff(&row,&col) << "\t-> ";
        if(row >= 2*Nvertex)
            std::cout << m_obj->m_zCoordJacobian(row-2*Nvertex,i);
        else if(row >= Nvertex)
            std::cout << m_obj->m_yCoordJacobian(row-Nvertex,i);
        else
            std::cout << m_obj->m_yCoordJacobian(row,i);
        std::cout << "\t-> " << dX0.col(i).norm()/dX0norms(0,i) << std::endl;
    }
    std::cout << "\n### Finished ###" << std::endl;
    return 0;
}

int PassageMeshInspector::save2Dlayers(const std::string& filePath) const
{
    std::ofstream file;
    file.open(filePath.c_str());
    if(file.is_open()) {
        for(size_t i=0; i<m_obj->m_layerMeri.size(); i++) {
            file << "### Layer " << i << " ###\n";
            file << " ## Meridional Coordinate ##\n";
            file << m_obj->m_layerMeri[i];
            file << "\n";
            file << " ## Pitch Coordinate ##\n";
            file << m_obj->m_layerPitch[i];
            file << "\n";
        }
        file.close();
        return 0;
    } else {
        return 1;
    }
}

int PassageMeshInspector::save2DlayerJacobian(const std::string& filePath) const
{
    std::ofstream file;
    file.open(filePath.c_str(), std::ios::binary | std::ios::out);
    float value;
    int sizes[4];
    if(file.is_open()) {
        sizes[0] = m_obj->m_layerMeri[0].cols();
        sizes[1] = m_obj->m_layerMeri[0].rows();
        sizes[2] = m_obj->m_layerJacobian.size();
        file.write(reinterpret_cast<char *>(sizes),sizeof(sizes));
        for(int i=0; i<sizes[2]; i++) {
            for(int k=0; k<m_obj->m_layerJacobian[0].cols(); k++) {
                for(int j=0; j<m_obj->m_layerJacobian[0].rows(); j++) {
                    value = float(m_obj->m_layerJacobian[i](j,k));
                    file.write(reinterpret_cast<char *>(&value),sizeof(value));
                }
            }
        }
        file.close();
        return 0;
    } else {
        return 1;
    }
}

int PassageMeshInspector::comp2DlayerJacobian(const std::string& filePath, const double tol) const
{
    std::ifstream file;
    file.open(filePath.c_str(), std::ios::binary | std::ios::in);
    assert(file.good());

    float value, diff = 0.0;
    char data[4];

    for(int i=0; i<4; ++i) file.read(data,4);

    for(size_t i=0; i<m_obj->m_layerJacobian.size(); ++i)
        for(int k=0; k<m_obj->m_layerJacobian[0].cols(); ++k)
            for(int j=0; j<m_obj->m_layerJacobian[0].rows(); ++j) {
                file.read(data,4);
                value = *reinterpret_cast<float*>(data);
                diff = std::max(diff,std::abs(value-float(m_obj->m_layerJacobian[i](j,k))));
            }
    file.close();
    std::cout << std::endl << "Compared 2D layer jacobian: " << diff << std::endl;
    if(diff > float(tol))
        return 1;
    else
        return 0;
}

int PassageMeshInspector::save3Dgrid(const std::string& filePath) const
{
    std::ofstream file;
    file.open(filePath.c_str());
    if(file.is_open()) {
        file << m_obj->m_vertexCoords;
        file.close();
        return 0;
    } else {
        return 1;
    }
}

int PassageMeshInspector::save3DgridJacobian(const std::string& filePath) const
{
    std::ofstream file;
    float value;
    int sizes[4];
    file.open(filePath.c_str(), std::ios::binary | std::ios::out);
    if(file.is_open()) {
        sizes[0] = m_obj->m_layerMeri[0].cols();
        sizes[1] = m_obj->m_layerMeri[0].rows();
        sizes[2] = m_obj->m_nSpanGrdPts;
        sizes[3] = m_obj->m_xCoordJacobian.cols();
        file.write(reinterpret_cast<char *>(sizes),sizeof(sizes));
        for(int j=0; j<m_obj->m_xCoordJacobian.cols(); j++) {
            for(int i=0; i<m_obj->m_xCoordJacobian.rows(); i++) {
                value = sqrt(pow(m_obj->m_xCoordJacobian(i,j),2.0)+
                             pow(m_obj->m_yCoordJacobian(i,j),2.0)+
                             pow(m_obj->m_zCoordJacobian(i,j),2.0));
                file.write(reinterpret_cast<char *>(&value),sizeof(value));
            }
        }
        file.close();
        return 0;
    } else {
        return 1;
    }
}

int PassageMeshInspector::comp3DgridJacobian(const std::string& filePath, const double tol) const
{
    std::ifstream file;
    file.open(filePath.c_str(), std::ios::binary | std::ios::in);
    assert(file.good());

    float value1, diff = 0.0, refVal = 0.0;
    char data[4];

    for(int i=0; i<4; ++i) file.read(data,4);

    for(int j=0; j<m_obj->m_xCoordJacobian.cols(); j++)
        for(int i=0; i<m_obj->m_xCoordJacobian.rows(); i++) {
            float value2 = sqrt(pow(m_obj->m_xCoordJacobian(i,j),2.0)+
                                pow(m_obj->m_yCoordJacobian(i,j),2.0)+
                                pow(m_obj->m_zCoordJacobian(i,j),2.0));
            file.read(data,4);
            value1 = *reinterpret_cast<float*>(data);
            diff = std::max(diff,std::abs(value1-value2));
            refVal = std::max(refVal,std::abs(value1));
        }
    file.close();
    diff /= refVal;
    std::cout << std::endl << "Compared 3D grid jacobian: " << diff << std::endl;
    if(diff > float(tol))
        return 1;
    else
        return 0;
}

}
