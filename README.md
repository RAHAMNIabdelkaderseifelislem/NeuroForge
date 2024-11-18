# NeuroForge (Beta) 🚀

> Build neural networks with ease! 🧠✨

NeuroForge is an intuitive tool for data preprocessing, analysis, visualization, and neural network creation using a drag-and-drop interface. This beta version provides core functionality while maintaining room for expansion.

## ✨ Features (Beta)

- **Data Processing** 📊
  - CSV and Excel file upload support
  - Basic data preprocessing capabilities
  - Data preview and basic statistics

- **Data Visualization** 📈
  - Automated histogram generation for numerical columns
  - Interactive plots using Plotly
  - Basic data insights

- **Neural Network Builder** 🧠
  - Drag-and-drop interface for network creation
  - Support for basic PyTorch layers:
    - Linear layers
    - Convolutional layers (Conv2d)
  - CUDA support for GPU acceleration (when available) ⚡

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/RAHAMNIabdelkaderseifelislem//neuroforge.git
cd neuroforge

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

1. Start the application:
```bash
streamlit run src/ui/app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload your dataset and follow the intuitive UI to:
   - Process and visualize your data 📊
   - Create neural network architectures 🧠
   - Configure and train your models ⚡

## 📁 Project Structure

```
neuroforge/
├── src/
│   ├── core/
│   │   ├── data_processor.py
│   │   ├── model_builder.py
│   │   └── model_trainer.py
│   └── ui/
│       └── app.py
├── requirements.txt
└── README.md
```

## 🏗️ Architecture

NeuroForge follows clean architecture principles and design patterns:

- **Factory Pattern** 🏭: Used for layer creation
- **Builder Pattern** 🔨: Implements neural network construction
- **Strategy Pattern** 🎯: Handles different data processing approaches
- **Dependency Injection** 💉: Manages component dependencies

## ⚠️ Limitations (Beta)

- Limited layer types available
- Basic visualization options
- Simple data preprocessing capabilities
- Training functionality is limited

## 🗺️ Roadmap

- [ ] Add more PyTorch layer types
- [ ] Enhance visualization capabilities
- [ ] Implement advanced data preprocessing
- [ ] Add model export in various formats
- [ ] Improve UI/UX
- [ ] Add comprehensive testing suite

## 🤝 Contributing

This is a beta version and contributions are welcome! Please feel free to submit issues and pull requests.

## 📜 License

MIT License - see LICENSE file for details

---
Built by [AbdEl Kader Seif El Islem RAHMANI](https://github.com/RAHAMNIabdelkaderseifelislem/)