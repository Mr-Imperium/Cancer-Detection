# Cancer Detection Web Application 🔬

A deep learning-powered web application for detecting cancer in medical images. Built with Streamlit and PyTorch, this application uses a fine-tuned ResNet34 model to provide real-time cancer detection from medical imaging data.

## Features

- Real-time cancer detection from medical images
- Interactive web interface built with Streamlit
- Probability distribution visualization with confidence scores
- Support for TIF format medical images
- System information monitoring
- Data augmentation during training
- Transfer learning using pretrained ResNet34

## Model Architecture

The application uses a modified ResNet34 architecture:
- Base: Pretrained ResNet34 from torchvision
- Modifications:
  - Frozen feature extraction layers
  - Custom fully connected layer (num_features → 2 classes)
  - Optimized for binary classification

### Training Details

- **Dataset Split**:
  - Training: 2608 images
  - Validation: 100 images
  - Test: Remaining images
- **Data Augmentation**:
  - Random horizontal flips
  - Random rotation (±5 degrees)
  - Resize to 224x224
  - Normalization (ImageNet statistics)
- **Training Parameters**:
  - Optimizer: Adam (lr=3e-4)
  - Loss Function: CrossEntropyLoss
  - Batch Size: 32
  - Workers: 2
- **Hardware**: Supports both CPU and GPU training

## Requirements

```
python >= 3.6
streamlit
torch
torchvision
Pillow
huggingface_hub
pandas
numpy
tqdm
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cancer-detection-webapp.git
cd cancer-detection-webapp
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Configure Hugging Face credentials:
   - Create an account on [Hugging Face](https://huggingface.co)
   - Replace `YOUR_HUGGING_FACE_USERNAME` with your username in `app.py`
   - For private repositories, set your HF token:
     ```bash
     export HF_TOKEN=your_token_here  # On Windows: set HF_TOKEN=your_token_here
     ```

## Usage

### Training the Model

1. Prepare your dataset:
   ```
   data/
   ├── data_sample/
   │   ├── image1.tif
   │   ├── image2.tif
   │   └── ...
   └── labels.csv
   ```

2. Run the training script:
```bash
python train_model.py
```

### Running the Web Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Use the application:
   - Upload a medical image (`.tif` format)
   - Click "Analyze Image"
   - View results and probability distribution

## Model Performance

The model is evaluated on a held-out test set with metrics including:
- Binary classification accuracy
- Loss convergence during training
- Real-time performance metrics available in the web interface

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development Roadmap

Future improvements planned:
- [ ] Support for additional image formats
- [ ] Model explainability visualizations
- [ ] Batch processing capabilities
- [ ] GPU acceleration support
- [ ] REST API endpoint
- [ ] Docker containerization

## License

[Add your chosen license here]

## Acknowledgments

- ResNet34 architecture from [torchvision](https://pytorch.org/vision/)
- Web interface built with [Streamlit](https://streamlit.io/)
- Model hosting on [Hugging Face Hub](https://huggingface.co/)

## Screenshots

[Add screenshots of your application here]

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{cancer_detection_webapp,
  author = {[Your Name]},
  title = {Cancer Detection Web Application},
  year = {2024},
  url = {https://github.com/yourusername/cancer-detection-webapp}
}
```
