cmake_minimum_required(VERSION 3.19)
project("Vulkan Tutorial" CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include("cmake/CMakeTemplates/shader.cmake")

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED)

add_executable(tutorial "main.cpp")
target_link_libraries(tutorial PRIVATE Vulkan::Vulkan SDL2::SDL2)
add_shader(VertShader "shader.vert")
add_shader(FragShader "shader.frag")
add_dependencies(tutorial VertShader FragShader)
