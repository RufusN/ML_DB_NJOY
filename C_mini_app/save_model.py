import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

saved_model_dir = "/Users/ru/FFT/ML_DB_sliced_pipeline/MLP/3x170_950_1050_model.keras"
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,  # or INT8 if calibrated
    max_workspace_size_bytes=1<<30  # Adjust based on your GPU memory
)
converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir,
                                     conversion_params=conversion_params)
converter.convert()
converter.save("/Users/ru/FFT/ML_DB_sliced_pipeline/C_mini_app/3x170_950_1050_model_converted.keras")