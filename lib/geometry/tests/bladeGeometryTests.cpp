#include "bladeGeometryTests.h"
#include "../src/bladeGeometry.h"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>

namespace geometryGeneration
{
void testSuite()
{
    std::cout << std::endl;
    _bladeGeometryTests();
}

void _bladeGeometryTests()
{
    std::cout << "### BLADE GEOMETRY GENERATION TESTS ###" << std::endl;

    fileManagement::GeometryParamManager geoParams;
    fileManagement::MeshParamManager mshParams;
    BladeGeometry blade;

    std::ifstream baseline, current;

    std::string geoParFile = "../../lib/geometry/tests/geoParFile.txt";
    std::string mshParFile = "../../lib/geometry/tests/mshParFile.txt";
    std::string geoOutFile = "../../lib/geometry/tests/geoOutFile.txt";
    std::string derOutFile = "../../lib/geometry/tests/derOutFile.txt";

    int errorCode;

    errorCode = geoParams.readFile(geoParFile);
    assert(errorCode == 0);

    errorCode = mshParams.readFile(mshParFile);
    assert(errorCode == 0);

    errorCode = blade.buildGeometry(geoParams,mshParams);
    assert(errorCode == 0);

    blade.saveGeometry("./tmp.txt");
    baseline.open(geoOutFile);
    current.open("./tmp.txt");
    double diff = 0.0;
    for(int span=0; span<2; ++span) {
        // consume span section
        std::string line;
        getline(baseline,line); getline(current,line);
        for(int coord=0; coord<2; ++coord) {
            // consume coordinate section
            getline(baseline,line); getline(current,line);
            for(int i=0; i<2*47; ++i) {
                // read values
                getline(baseline,line);
                std::stringstream ss1(line); double v1; ss1 >> v1;
                getline(current,line);
                std::stringstream ss2(line); double v2; ss2 >> v2;
                diff = std::max(diff,std::abs(v2-v1));
            }
        }
    }
    baseline.close(); current.close();
    assert(diff < 1e-6);

    blade.saveJacobian("./tmp.txt");
    baseline.open(derOutFile);
    current.open("./tmp.txt");
    diff = 0.0;
    for(int span=0; span<2; ++span) {
        // consume span section
        std::string line;
        getline(baseline,line); getline(current,line);
        for(int coord=0; coord<2; ++coord) {
            // consume coordinate section
            getline(baseline,line); getline(current,line);
            for(int i=0; i<2*47; ++i) {
                getline(baseline,line); std::stringstream ss1(line);
                getline(current,line);  std::stringstream ss2(line);
                for(int var=0; var<26; ++var) {
                    // read values
                    double v1; ss1 >> v1; ss1.ignore();
                    double v2; ss2 >> v2; ss2.ignore();
                    diff = std::max(diff,std::abs(v2-v1));
                }
            }
        }
    }
    baseline.close(); current.close();
    assert(diff < 1e-6);

    std::remove("./tmp.txt");
}

}
