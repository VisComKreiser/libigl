#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/PI.h>

// ToDo: is this really necessary?
#include <Eigen/Geometry>

#include "tutorial_shared_path.h"

#include <iostream>

template <typename T>
T clamp(T value, T min_value, T max_value)
{
    return std::min(std::max(value, min_value), max_value);
}

void triangleArea2D_CCW(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXd& area)
{
    /*
    # half the area of a planar parallelogram
    # |v x w| = |v| * |w| * |sin(theta)| with theta = angle between v and w and v = vector(p0, p1) and w = vector(p0, p2)
    # returned area is signed
    v0 = p1 - p0
    v1 = p2 - p0
    # simplified cross product in 2D
    */
    area.resize(F.rows());

    for (size_t idx{ 0 }; idx < area.size(); ++idx)
    {
        const Eigen::Vector2d& p0(V.row(F(idx, 0)));
        const Eigen::Vector2d& p1(V.row(F(idx, 1)));
        const Eigen::Vector2d& p2(V.row(F(idx, 2)));

        const Eigen::Vector2d v0(p1 - p0);
        const Eigen::Vector2d v1(p2 - p0);

        area(idx) = 0.5 * (v0.x() * v1.y() - v1.x() * v0.y());
    }
}

void triangleArea3D(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::VectorXd& area)
{
    // returns the area for each triangle
    area.resize(F.rows());

    for (size_t idx{ 0 }; idx < area.size(); ++idx)
    {
        const Eigen::Vector3d& p0(V.row(F(idx, 0)));
        const Eigen::Vector3d& p1(V.row(F(idx, 1)));
        const Eigen::Vector3d& p2(V.row(F(idx, 2)));

        const Eigen::Vector3d v0(p1 - p0);
        const Eigen::Vector3d v1(p2 - p0);

        area(idx) = 0.5 * v0.cross(v1).norm();
    }
}

void triangleAngles(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& angles, const double acos_clamp = 1.0 - 1e-5)
{
    /*
    # calculate angles at all vertices of the triangle using the cosine-similarity law
    # pi minus the two first angles is the third one
    */
    angles.resize(F.rows(), F.cols());

    for (size_t idx{ 0 }; idx < angles.rows(); ++idx)
    {
        const Eigen::Vector3d& p0(V.row(F(idx, 0)));
        const Eigen::Vector3d& p1(V.row(F(idx, 1)));
        const Eigen::Vector3d& p2(V.row(F(idx, 2)));

        const Eigen::Vector3d v01 = (p1 - p0).normalized();
        const Eigen::Vector3d v02 = (p2 - p0).normalized();
        const Eigen::Vector3d v21 = (p1 - p2).normalized();

        const double phi0 = std::acos(clamp(v01.dot(v02), -acos_clamp, acos_clamp));
        const double phi1 = std::acos(clamp(v01.dot(v21), -acos_clamp, acos_clamp));

        angles.row(idx) = Eigen::Vector3d(phi0, phi1, igl::PI - phi0 - phi1);
    }
}

int main(int argc, char* argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    // Load a mesh
    igl::readOBJ("D:/tmp/cube.obj", V, F);

    // calculate triangle area
    Eigen::VectorXd area;
    triangleArea3D(V, F, area);
    Eigen::MatrixXd angles;
    triangleAngles(V, F, angles);

    // debug print
    std::cout << "area:" << std::endl;
    for (size_t idx{ 0 }; idx < area.size(); ++idx)
    {
        std::cout << area(idx) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "angles:" << std::endl;
    for (size_t idx{ 0 }; idx < angles.rows(); ++idx)
    {
        std::cout << angles.row(idx) << std::endl;
    }

    return 0;
}
