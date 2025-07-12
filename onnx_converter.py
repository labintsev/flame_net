import torch
from main import ConvNet, NUM_CLASSES

def export_to_onnx():
    # Load model weights
    model = ConvNet(NUM_CLASSES)
    model.load_state_dict(torch.load("weights/convnet_0.pth", map_location="cpu"))
    model.eval()

    # Example input: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 1024, 1024)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "weights/convnet_0.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )

    print("Exported convnet_0.pth to weights/convnet_0.onnx")


def test_onnx():
    import onnx
    import onnxruntime as ort

    # Load the ONNX model
    model = onnx.load("weights/convnet_0.onnx")
    onnx.checker.check_model(model)
    print("ONNX model is valid.")

    # Create an inference session
    session = ort.InferenceSession("weights/convnet_0.onnx")

    # Example input: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 1024, 1024).numpy()
    
    # Run inference
    outputs = session.run(None, {"input": dummy_input})
    print(f"Output shape: {outputs[0].shape}")


if __name__ == "__main__":
    export_to_onnx()
    test_onnx()
