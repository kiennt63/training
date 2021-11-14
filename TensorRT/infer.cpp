#include <iostream>
#include <fstream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <numeric>

size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: cannot read file with class names\n";
        return classes;
    }
    std::string class_name;
    while( std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

void preprocessImage(const std::string& img_path, float* gpu_input, const nvinfer1::Dims& dims)
{
    cv::Mat frame = cv::imread(img_path);
    if (frame.empty())
    {
        std::cerr << "Input img loading failed!" << std::endl;
    }
    std::cout << "Input image loaded successfully!" << std::endl;

    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);
    auto input_width = dims.d[2];
    auto input_height = dims.d[3];
    auto channels = dims.d[1];
    // std::cout << dims.d[0] << std::endl;
    std::cout << "SIZE: " << dims.d[0] << " x " << dims.d[1] << " x " << dims.d[2] << " x " << dims.d[3] << std::endl;
    auto input_size = cv::Size(input_width, input_height);
    // Resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    cv::Mat host_debug;
    resized.download(host_debug);

    // Normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    std::cout << host_debug.size() << std::endl;
    
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    
    // ToTensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

void postprocessResults(float* gpu_output, const nvinfer1::Dims& dims, int batch_size, const std::string& class_names_file)
{
    auto classes = getClassNames(class_names_file);
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector< int > indices(getSizeByDim(dims) * batch_size);
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    // while (cpu_output[indices[i]] / sum > 0.005)
    // {
    //     if (classes.size() > indices[i])
    //     {
    //         std::cout << "class: " << classes[indices[i]] << " | ";
    //     }
    //     std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
    //     ++i;
    // }
}

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override
    {
        // Only log if level == error
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
        {
            std::cout << msg << std::endl;
        }
    }

} gLogger; // -> create instance gLogger

// TRTUniquePtr definition
struct TRTDestroy
{
    template<class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template<class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// Initialize model
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine, TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    nvinfer1::NetworkDefinitionCreationFlags flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flags)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};

    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model" << std::endl;
        return;
    }

    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // Memory for tactic selection
    config->setMaxWorkspaceSize(1ULL << 30);
    // Use FP16 if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // set batch size
    builder->setMaxBatchSize(1);
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}

void serializedEngine(const std::string& output_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine)
{
    auto serializedEngine = engine->serialize();
    std::ofstream f(output_path.c_str());
    f.write((const char*)serializedEngine->data(),serializedEngine->size());
    f.close(); 
}

void deserializeEngine(const std::string& engine_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine, TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    std::vector<char> trt_model_stream;
    size_t size{0};

    std::ifstream file(engine_path.c_str(), std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trt_model_stream.resize(size);
        file.read(trt_model_stream.data(), size);
        file.close();
    }
    else
    {
        std::cerr << "Could not load the engine file" << std::endl;
    }
    TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
    assert(runtime != nullptr);

    engine.reset(runtime->deserializeCudaEngine(trt_model_stream.data(), size, nullptr));
    context.reset(engine->createExecutionContext());
}

int main(int argc, char * argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_file> <class_names_file>" << std::endl;
        return -1;
    }
    int batch_size = 1;
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    std::string image_path(argv[2]);

    // // Parse onnx and create engine
    // std::string model_path(argv[1]);
    // parseOnnxModel(model_path, engine, context);

    // // Serialize engine
    // std::string output_path(argv[1]);
    // serializedEngine(output_path.c_str(), engine);

    // Deserialize engine
    std::string engine_path(argv[1]);
    deserializeEngine(engine_path, engine, context);

    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    std::vector<void*> buffers(engine->getNbBindings());

    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
            std::cout << "Input dimensions: " << input_dims[0].nbDims << std::endl;
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
            std::cout << "Output dimensions: " << output_dims[0].nbDims << std::endl;
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "ERROR: expected at least one input and one output for network\n";
        return -1;
    }
    // auto start = std::chrono::high_resolution_clock::now();
    // preprocess input data
    preprocessImage(image_path, (float*) buffers[0], input_dims[0]);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);
    for (size_t i = 0; i < 10000; ++i)
    {
        // inference
        context->enqueue(batch_size, &buffers[0], stream, nullptr);
    }
    cudaEventRecord(end, stream);

    cudaEventSynchronize(end);

    float timeTaken;
    cudaEventElapsedTime(&timeTaken, start, end);
    // auto stop = std::chrono::high_resolution_clock::now();
    // postprocess results
    postprocessResults((float*) buffers[1], output_dims[0], batch_size, argv[3]);
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken: " << timeTaken << " seconds" << std::endl;
    

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }

    return 0;

}

