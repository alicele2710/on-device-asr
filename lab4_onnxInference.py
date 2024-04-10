# import onnxruntime as ort
# import numpy as np
# import time
#
# def measure_inference_latency(onnx_model_path, sample_input, iterations=10):
#     # Load the ONNX model
#     session = ort.InferenceSession(onnx_model_path)
#
#     # Prepare the input data. ONNX Runtime expects numpy arrays
#     input_name = session.get_inputs()[0].name
#     input_data = {input_name: sample_input}  # Directly use the NumPy array
#
#     # Warm-up run (for more accurate measurement, especially on GPU)
#     _ = session.run(None, input_data)
#
#     # Measure the inference latency
#     start_time = time.time()
#     for _ in range(iterations):
#         _ = session.run(None, input_data)
#     end_time = time.time()
#
#     # Calculate the average latency
#     avg_latency = (end_time - start_time) / iterations
#     return avg_latency
#
#
# # Sample input for the model (adjust the shape as necessary)
# batch_size = 1
# sample_input = np.random.rand(batch_size, 118793).astype(np.float32)
#
# # Test the original model
# original_model_path = "original_sew_model.onnx"
# original_latency = measure_inference_latency(original_model_path, sample_input)
# print(f"Average Inference Latency of Original Model: {original_latency:.6f} seconds")
#
# # Test the pruned model (replace with your actual pruned model path)
# pruned_model_path = "pruned_sew_model.onnx"  # Example for the first iteration
# pruned_latency = measure_inference_latency(pruned_model_path, sample_input)
# print(f"Average Inference Latency of Pruned Model: {pruned_latency:.6f} seconds")

import onnxruntime as ort
import numpy as np
import time

def measure_inference_latency(onnx_model_path, sample_input, iterations=10):
    # Load the ONNX model
    start_time = time.time()
    session = ort.InferenceSession(onnx_model_path)

    # Prepare the input data. ONNX Runtime expects numpy arrays
    input_name = session.get_inputs()[0].name
    input_data = {input_name: sample_input}

    # Warm-up run (for more accurate measurement, especially on GPU)
    _ = session.run(None, input_data)

    end_time = time.time()

    print(f"load the model latency = {end_time - start_time:.6f} seconds")

    # Measure the inference latency
    latencies = []
    for _ in range(iterations):
        start_time = time.time()
        _ = session.run(None, input_data)
        end_time = time.time()
        latencies.append(end_time - start_time)

    # Calculate the average latency and standard deviation
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return avg_latency, std_latency

# Sample input for the model (adjust the shape as necessary)
batch_size = 1
sample_input = np.random.rand(batch_size, 118793).astype(np.float32)

# Test the original model
original_model_path = "original_sew_model.onnx"
original_latency, original_std = measure_inference_latency(original_model_path, sample_input)
print(f"Original Model: Average Latency = {original_latency:.6f} seconds, Std Dev = {original_std:.6f} seconds")

# # Test the pruned model (replace with your actual pruned model path)
# pruned_model_path = "pruned_model_iter_1.onnx"  # Example for the first iteration
# pruned_latency, pruned_std = measure_inference_latency(pruned_model_path, sample_input)
# print(f"Pruned Model: Average Latency = {pruned_latency:.6f} seconds, Std Dev = {pruned_std:.6f} seconds")


for i in range(1, 2):
    # Test the pruned model (replace with your actual pruned model path)
    pruned_model_path = f"pruned_sew_model_iter_{i}.onnx"  # Example for the first iteration
    pruned_latency, pruned_std = measure_inference_latency(pruned_model_path, sample_input)
    print(f"Pruned Model: Average Latency = {pruned_latency:.6f} seconds, Std Dev = {pruned_std:.6f} seconds")