#include <igl/readOFF.h>
#include <igl/writeOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/readSTL.h>
#include <igl/writeSTL.h>

#include <igl/remove_unreferenced.h>

#include "tutorial_shared_path.h"

#include <string>
#include <vector>
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[])
{
    using namespace Eigen;
    namespace fs = std::filesystem;

    const std::vector<std::string> data_set_ids = { "D1", "D2", "D3" };
    const std::vector<size_t> decimations = { 1000, 5000, 10000 };

    for (const auto& data_set_id : data_set_ids)
    {
        for (const auto& decimation : decimations)
        {
            const std::string data_path{ "E:/learning/data/progressive_parameterizations/" + data_set_id + "/decimated_" + std::to_string(decimation) + "/" };

            for (const auto& entry : fs::directory_iterator(data_path)) {
                std::cout << entry.path().filename().string() << std::endl;

                // load file
                const auto& filename = entry.path().string();
                Eigen::MatrixXd V, NV;
                Eigen::MatrixXi F, NF;
                Eigen::VectorXi I;
                if (!igl::readOBJ(filename, V, F)) {
                    std::cout << "could not read: " << filename << std::endl;
                    continue;
                }

                // cleanup input
                igl::remove_unreferenced(V, F, NV, NF, I);
                //std::cout << V.rows() << " - " << F.rows() << std::endl;
                //std::cout << NV.rows() << " - " << NF.rows() << std::endl;

                // export
                igl::writeOBJ(filename, NV, NF);
            }
        }
    }

    return 0;
}
