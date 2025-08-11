
import os
import onnxruntime as rt
import numpy as np

def check_available_providers():
    """
    Check and print the available providers for ONNX Runtime.
    """
    available_providers = rt.get_available_providers()
    print("\nAvailable Execution Providers:", available_providers)
    return available_providers

def load_facenet_model(onnx_model_path="weights/facenet128.onnx", mode="gpu_optimized"):
    """
    Load the ONNX FaceNet model from the specified path with selected execution mode.

    Args:
        onnx_model_path (str): the path to the ONNX model file.
        mode (str): The execution mode for ONNX Runtime.
                    Options: 'gpu', 'gpu_optimized', 'cpu', 'cpu_optimized', 'npu', 'npu_optimized'.

    Returns:
        session (InferenceSession): ONNX Runtime session for the loaded model.
    """
    # Check providers before loading the model
    available_providers = check_available_providers()

    # Get the directory of the current script file
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path relative to the script's directory
    full_onnx_path = os.path.join(base_dir, onnx_model_path)

    # Verify the ONNX model file exists
    if not os.path.exists(full_onnx_path):
        raise FileNotFoundError(f"ONNX model not found at {full_onnx_path}. Please ensure the file exists.")

    # Create session options
    session_options = rt.SessionOptions()

    # Choose the execution mode using match-case for extensibility
    match mode:
        case "gpu":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        case "gpu_optimized":
            session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        case "cpu":
            providers = ['CPUExecutionProvider']
        case "cpu_optimized":
            session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CPUExecutionProvider']
        case "npu":
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        case "npu_optimized":
            session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        case _:
            raise ValueError(f"Invalid mode selected: {mode}. Choose from 'gpu', 'gpu_optimized', 'cpu', 'cpu_optimized', 'npu', 'npu_optimized'.")

    # Validate if the desired provider is available
    if not any(provider in available_providers for provider in providers):
        raise ValueError(
            f"None of the desired providers {providers} are available on this system. "
            f"Available providers are: {available_providers}"
        )

    # Load the ONNX model into an inference session
    try:
        print(f"\nLoading ONNX model from {full_onnx_path}...\n")
        session = rt.InferenceSession(full_onnx_path, sess_options=session_options, providers=providers)
        print("\nONNX model successfully loaded.\n")
        print("Using Execution Providers:", session.get_providers())
    except Exception as err:
        raise ValueError(
            f"An error occurred while loading the ONNX model from {full_onnx_path}. "
            "Please ensure the file is correct and not corrupted."
        ) from err

    return session


class FaceNetClient:
    """
    FaceNet ONNX model client class.
    """

    def __init__(self, model_type="128", mode="gpu"):
        """
        Initialize the FaceNet model with the selected model type and execution mode.

        Args:
            model_type (str): The type of FaceNet model to load. Options: '128' or '512'.
            mode (str): The execution mode for ONNX Runtime.
                        Options: 'gpu', 'gpu_optimized', 'cpu', 'cpu_optimized', 'npu', 'npu_optimized'.
        """
        # Set the model path based on the selected type
        if model_type == "512":
            onnx_model_path = "weights/facenet512.onnx"
            self.output_shape = 512
            self.model_name = "FaceNet-512d"
        else:
            onnx_model_path = "weights/facenet128.onnx"
            self.output_shape = 128
            self.model_name = "FaceNet-128d"

        # Load the ONNX model with the selected mode and model path
        self.model = load_facenet_model(onnx_model_path=onnx_model_path, mode=mode)
        self.input_shape = (160, 160)

        # Get the model input and output names for inference
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Perform inference using the ONNX model.

        Args:
            img (np.ndarray): The input image (as a numpy array) to the model.

        Returns:
            embedding (np.ndarray): The output embedding from the model as a numpy array.
        """
        # Run inference on the ONNX model
        result = self.model.run([self.output_name], {self.input_name: img})[0]

        return result  # Return the embedding as a 1D array


def scaling(x, scale):
    """
    Scale the input array x by the specified scale factor.

    Args:
        x (np.ndarray): Input array to be scaled.
        scale (float): Scaling factor.

    Returns:
        np.ndarray: Scaled array.
    """
    return x * scale


