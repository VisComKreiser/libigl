#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/readSTL.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/collapse_edge.h>
#include <igl/edge_flaps.h>
#include <igl/decimate.h>
#include <igl/remove_unreferenced.h>
#include <igl/shortest_edge_and_midpoint.h>
#include <igl/parallel_for.h>
#include <igl/opengl/glfw/Viewer.h>

#include "tutorial_shared_path.h"

#include <string>
#include <filesystem>

Eigen::MatrixXd V, NV;
Eigen::MatrixXi F, NF;
Eigen::MatrixXd V_uv;
Eigen::VectorXi I;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    if (key == '1')
    {
      // Plot the 3D mesh
        viewer.data().set_mesh(V, F);
        viewer.core().align_camera_center(V, F);
    }
    else if (key == '2')
    {
      // Plot the mesh in 2D using the UV coordinates as vertex coordinates
        viewer.data().set_mesh(V_uv, F);
        viewer.core().align_camera_center(V_uv, F);
    }

    viewer.data().compute_normals();

    return false;
}

void collapse_mesh(const Eigen::MatrixXd& V_in, const Eigen::MatrixXi& F_in, Eigen::MatrixXd& V_out, Eigen::MatrixXi& F_out, size_t target_size)
{
    using namespace Eigen;

    // Prepare array-based edge data structures and priority queue
    VectorXi EMAP;
    MatrixXi E, EF, EI;
    igl::min_heap< std::tuple<double, int, int> > Q;
    VectorXi EQ;
    // If an edge were collapsed, we'd collapse it to these points:
    MatrixXd C;

    // initialize
    V_out = V_in;
    F_out = F_in;

    // calc. edge info
    igl::edge_flaps(F_out, E, EMAP, EF, EI);
    C.resize(E.rows(), V_out.cols());
    VectorXd costs(E.rows());
    // https://stackoverflow.com/questions/2852140/priority-queue-clear-method
    // Q.clear();
    Q = {};
    EQ = VectorXi::Zero(E.rows());
    {
        VectorXd costs(E.rows());
        igl::parallel_for(E.rows(), [&](const int e)
            {
                double cost = e;
                RowVectorXd p(1, 3);
                igl::shortest_edge_and_midpoint(e, V_out, F_out, E, EMAP, EF, EI, cost, p);
                C.row(e) = p;
                costs(e) = cost;
            }, 10000);
        for (int e = 0; e < E.rows(); e++)
        {
            Q.emplace(costs(e), e, 0);
        }
    }

    // collapse edge
    const auto max_iter = static_cast<size_t>(F.rows()) - target_size;
    size_t num_collapsed = 0;
    for (size_t idx = 0; idx < max_iter; idx++)
    {
        if (!igl::collapse_edge(igl::shortest_edge_and_midpoint, V_out, F_out, E, EMAP, EF, EI, Q, EQ, C))
        {
            break;
        }
        num_collapsed++;
    }

    std::cout << "num_collapsed: " << num_collapsed << std::endl;
}

int main(int argc, char* argv[])
{
    using namespace Eigen;
    namespace fs = std::filesystem;

    const std::string data_path{ "D:/tmp/decimated/1000_faces/" };

    size_t too_many_boundaries{ 0 }, no_boundary{ 0 };
    for (const auto& entry : fs::directory_iterator(data_path)) {
        std::cout << entry.path().filename().string() << std::endl;

        // load file
        const auto& filename = entry.path().string();
        if (!igl::readOBJ(filename, V, F)) {
            std::cout << "could not read: " << filename << std::endl;
            continue;
        }

        // cleanup input
        igl::remove_unreferenced(V, F, NV, NF, I);
        V = NV;
        F = NF;

        // get boundary loops
        std::vector<std::vector<int>> b;
        igl::boundary_loop(F, b);

        // check if only one loop exists
        if (b.size() > 1) {
            std::cout << "too many boundaries: " << filename << std::endl;
            too_many_boundaries++;
            continue;
        }
        else if (b.empty()) {
            std::cout << "no boundaries:       " << filename << std::endl;
            no_boundary++;
            continue;
        }

        // collapse to target size
        //collapse_mesh(V, F, V, F, 1000);

        // write to Eigen object
        const auto& b_tmp = b[0];
        const auto num_bnd_verts = b_tmp.size();
        VectorXi bnd;
        bnd.resize(num_bnd_verts, 1);
        for (size_t i{ 0 }; i < num_bnd_verts; ++i)
            bnd(i) = b_tmp[i];

        // map boundary to circle
        MatrixXd bnd_uv;
        igl::map_vertices_to_circle(V, bnd, bnd_uv);

        // flatten
        igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);

        // pad
        auto V_uv_padded = V_uv;
        V_uv_padded.conservativeResize(V_uv_padded.rows(), V_uv_padded.cols() + 1);
        MatrixXd padding_data(V_uv_padded.rows(), 1);
        padding_data.setZero();
        V_uv_padded.col(V_uv_padded.cols() - 1) = padding_data;

        // export
        igl::writeOBJ(std::string("D:/tmp/decimated/1000_faces/flattened/") + entry.path().filename().string(), V_uv_padded, F);
    }

    // draw statistics
    std::cout << "meshes with too many open boundaries: " << too_many_boundaries << std::endl;
    std::cout << "meshes with no open boundary:         " << no_boundary << std::endl;

    return 0;
}
