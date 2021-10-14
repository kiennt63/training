cmake_minimum_required(VERSION 3.5)

project(trt)
enable_language(CUDA)
find_package(OpenCV REQUIRED)

add_executable(infer infer.cpp)

target_link_libraries(infer PRIVATE
                        nvinfer
                        nvonnxparser
                        cuda
                        cudart
                        ${OpenCV_LIBS}
                        )

target_include_directories(infer PRIVATE
                        /usr/local/cuda-10.2/targets/x86_64-linux/include/
                        )
