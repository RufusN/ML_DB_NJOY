#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>

int main() {
    // Create a TensorRT runtime instance
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    
    // Deserialize your engine from file (engineFilePath)
    std::ifstream file("path/to/engine.trt", std::ios::binary);
    std::string engineData((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);
    if (!engine) {
        std::cerr << "Error loading engine!" << std::endl;
        return -1;
    }

    // Create execution context and perform inference (details omitted)
    // ...

    // Clean up
    engine->destroy();
    runtime->destroy();
    return 0;
}
