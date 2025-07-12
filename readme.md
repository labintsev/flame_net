# FlameNet

A simple PyTorch pipeline for video frame classification using convolutional neural networks.

## Project Structure

```
flame_net/
├── data/
│   ├── rotated/         # Directory with input videos (rotated and preprocessed)
│   └── labels.csv       # CSV file: file_name,label
├── main.py              # Main training/testing script
├── .gitignore
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- opencv-python
- onnx
- onnxruntime

Install dependencies:
```bash
pip install torch torchvision pandas opencv-python onnx onnxruntime
```

## Data Preparation

- Place your videos in rotated.
- Create a labels.csv file with the following format:

  ```
  file_name,label
  NMS_0316.MOV,0
  NMS_0317.MOV,1
  ...
  ```

## Usage

### Training

Run the training script:

```bash
python main.py
```

- All frames from all videos are loaded into memory and shuffled.
- Each frame is classified independently.
- Model checkpoints are saved as simple_convnet.pth.

### Testing

Uncomment the `test()` call in main.py to run a simple test loader.

## Model

- The default model is a convolutional neural network (ConvNet) with several layers.
- You can easily modify the architecture in main.py.

## Customization

- Change `DATA_DIR`, `LABELS_PATH`, `NUM_CLASSES`, `BATCH_SIZE`, `EPOCHS`, and other parameters at the top of main.py.
- Adjust frame size in the `VideoDataset` class if needed.

## ONNX Export and Test

You can export your trained PyTorch model to ONNX format and test it using the provided `onnx_converter.py` script.

### Export to ONNX

```bash
python onnx_converter.py
```

This will:
- Export the model weights (e.g., `weights/convnet_0.pth`) to ONNX format (`weights/convnet_0.onnx`).
- Print confirmation after export.

### Test ONNX Model

The script will also:
- Validate the exported ONNX model.
- Run a test inference using ONNX Runtime and print the output shape.

Make sure you have installed the `onnx` and `onnxruntime` packages.

## Notes

- All video frames are loaded into RAM at startup. For large datasets, consider a streaming approach.
- Each frame is treated as an independent sample for classification.

## License

MIT License

---

**Author:** Andrej  
**Project:** FlameNet  
