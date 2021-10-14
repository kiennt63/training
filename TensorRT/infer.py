import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import model_loading
import torch
import time

ONNX_FILE_PATH = 'resnet50.onnx'
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path, output_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # parse ONNX
    # with open(onnx_file_path, 'rb') as model:
    #     print('Beginning ONNX file parsing')
    #     success = parser.parse(model.read())
    # print('Completed parsing of ONNX file')
    success = parser.parse_from_file(onnx_file_path)
    if not success:
        return False

    print('[DEBUG MESSAGE] {}:{}'.format('', success))


    # Config
    builder.max_workspace_size = 1 << 30 # GPU mem for tactic selection
    builder.max_batch_size = 1
    if builder.platform_has_fast_fp16: # Use FP16 mode if possible
        builder.fp16_mode = True

    print('Building engine...')
    engine = builder.build_cuda_engine(network)
    with open(output_file_path, 'wb') as f:
        f.write(engine.serialize())
    print('Completed creating Engine')

    return True

def main():
    # sadge = build_engine(ONNX_FILE_PATH, 'engine.trt')
    engine = None
    with open('engine.trt', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    print('[DEBUG MESSAGE] {}:{}'.format('BINDING NUMS: ', len(engine)))
    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
            device_input = cuda.mem_alloc(input_size)
        else:
            output_shape = engine.get_binding_shape(binding)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    
    stream = cuda.Stream()
    start_time = time.time()
    for i in range(10000):
        host_input = np.array(model_loading.preprocess_img('turkish_coffee.webp').numpy(), dtype=np.float32, order='C')
        cuda.memcpy_htod_async(device_input, host_input, stream)

        context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()
        
        output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])
        model_loading.postprocess(output_data)
    end_time = time.time()

    print('Time taken: {} seconds'.format(end_time - start_time))

if __name__ == '__main__':
    main()