get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME} parameter_calculation.cpp)
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl igl::opengl_glfw OpenMP::OpenMP_CXX tutorials)
