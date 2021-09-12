cmake_minimum_required(VERSION 3.7)
project("Vulkan Tutorial" C)

include("cmake/CMakeTemplates/CFlags.cmake")
include("cmake/CMakeTemplates/shader.cmake")

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(tutorial "main.c")
target_link_libraries(tutorial PRIVATE Vulkan::Vulkan glfw)
add_shader(VertShader "shader.vert")
add_shader(FragShader "shader.frag")
add_dependencies(tutorial VertShader FragShader)
