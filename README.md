# SwellSight Wave Analysis Model

SwellSight is a multi-task deep learning system that extracts objective physical wave parameters from single beach camera images using computer vision and synthetic data generation.

## Dataset Access

**[Download SwellSight Dataset](https://drive.google.com/drive/folders/11gDkj5GhGVXM9uMlJaVp0b_jIxK_OaoZ?usp=drive_link)**

The complete SwellSight dataset including labeled beach camera images and annotations is available on Google Drive. This dataset contains:
- 729 labeled beach camera images with wave parameter annotations
- Ground truth labels for wave height, type, and direction
- Image metadata and confidence scores
- Pre-processed splits for training, validation, and testing

**Note**: Download the dataset and place it in the `data/real/` directory before running the notebooks or scripts.

## Overview

SwellSight predicts three key wave characteristics from a single RGB image using a shared feature extractor with task-specific heads:

- **Wave Height**: Continuous value in meters (regression)
- **Wave Type**: `beach_break`, `reef_break`, `point_break`, `closeout`, `a_frame` (classification)
- **Wave Direction**: `left`, `right`, `both` (classification)

### Data Sources

The model is trained on two complementary datasets:

- **Real Dataset**: 729 labeled beach-camera images with comprehensive wave parameter annotations ([Download from Google Drive](https://drive.google.com/drive/folders/11gDkj5GhGVXM9uMlJaVp0b_jIxK_OaoZ?usp=drive_link))
- **Synthetic Dataset**: Photorealistic images generated using **SDXL + ControlNet Depth**, guided by procedural depth maps with controllable wave parameters

## Key Features

- **Multi-task Learning**: Shared backbone with specialized heads for efficient parameter extraction
- **Synthetic Data Generation**: SDXL and ControlNet Depth for realistic wave image synthesis
- **Procedural Depth Maps**: Controllable wave parameter generation for training data
- **Reproducible Pipeline**: JSONL-based dataset indexing and split generation
- **End-to-End Workflow**: Complete scripts for data preparation, generation, and training
- **CLI Inference**: Single-image wave analysis from command line

## Repository Structure

```
SwellSight_Project/
├── swellsight/                    # Core package
│   ├── controlnet/               # Synthetic data generation
│   │   ├── depth_utils.py
│   │   ├── param_depth_generator.py
│   │   └── sdxl_depth_controlnet.py
│   ├── data.py                   # Dataset handling
│   ├── model.py                  # Neural network architecture
│   ├── losses.py                 # Loss functions
│   ├── metrics.py                # Evaluation metrics
│   ├── transforms.py             # Data augmentation
│   └── utils.py                  # Utility functions
├── scripts/                      # Data pipeline scripts
│   ├── 00_build_real_index.py    # Index real dataset
│   ├── 01_split_real.py          # Create train/val/test splits
│   ├── 02_generate_real_aug.py   # Data augmentation
│   ├── 03_generate_param_synth.py # Generate synthetic data
│   └── 04_build_mix_splits.py    # Merge real + synthetic
├── data/                         # Dataset storage
│   ├── real/                     # Original labeled images
│   ├── processed/                # Processed datasets
│   └── synthetic/                # Generated synthetic data
├── runs/                         # Training outputs
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── inference.py                  # Single image inference
└── requirements.txt              # Dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for training and synthetic generation)

### Setup

1. **Clone and navigate to the repository:**
   ```bash
   cd SwellSight_Project  # Navigate to project root
   ```

2. **Download the dataset:**
   - Visit the [SwellSight Dataset on Google Drive](https://drive.google.com/drive/folders/11gDkj5GhGVXM9uMlJaVp0b_jIxK_OaoZ?usp=drive_link)
   - Download all files and folders
   - Extract to `data/real/` directory in your project root
   - Ensure the structure matches: `data/real/images/` and `data/real/labels.json`

3. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Python path:**
   ```bash
   export PYTHONPATH="."
   python -c "import swellsight; print('SwellSight imports successfully')"
   ```

   > **Note**: For one-time commands, you can use: `PYTHONPATH=. python <script>`

## Data Pipeline

**⚠️ Important**: Before running the data pipeline, make sure you have downloaded the dataset from [Google Drive](https://drive.google.com/drive/folders/11gDkj5GhGVXM9uMlJaVp0b_jIxK_OaoZ?usp=drive_link) and placed it in the `data/real/` directory.

Follow these steps to prepare your dataset for training:

### 1. Build Real Dataset Index

Create an index file for the real dataset:

```bash
PYTHONPATH=. python scripts/00_build_real_index.py \
  --images_dir data/real/images \
  --labels_json data/real/labels.json \
  --out_index data/processed/real_index.jsonl
```

**Inputs:**
- `data/real/images/` - Directory containing beach camera images
- `data/real/labels.json` - Wave parameter annotations

**Output:**
- `data/processed/real_index.jsonl` - Indexed dataset

### 2. Create Train/Validation/Test Splits

Split the real dataset for training:

```bash
PYTHONPATH=. python scripts/01_split_real.py \
  --real_index data/processed/real_index.jsonl \
  --out_dir data/processed/splits
```

**Outputs:**
- `data/processed/splits/train.jsonl`
- `data/processed/splits/val.jsonl`
- `data/processed/splits/test.jsonl`

### 3. Generate Synthetic Images

Create synthetic training data using SDXL + ControlNet:

```bash
OUT_DIR="data/synthetic/param_synth_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

PYTHONPATH=. python scripts/03_generate_param_synth.py \
  --out_dir "$OUT_DIR" \
  --n 300 \
  --seed 42 \
  --steps 28 \
  --guidance 6.0 \
  --control_scale 0.55
```

**Outputs:**
- `$OUT_DIR/images/*.png` - Generated RGB images
- `$OUT_DIR/depth/*.png` - Depth maps used for conditioning
- `$OUT_DIR/index.jsonl` - Ground truth labels for synthetic samples

### 4. Create Mixed Training Splits

Combine real and synthetic data:

```bash
PYTHONPATH=. python scripts/04_build_mix_splits.py \
  --real_splits_dir data/processed/splits \
  --param_synth_index "$OUT_DIR/index.jsonl" \
  --out_dir data/synthetic/mix
```

**Outputs:**
- `data/synthetic/mix/train.jsonl` - Combined training data
- `data/synthetic/mix/val.jsonl` - Validation split
- `data/synthetic/mix/test.jsonl` - Test split
## Running Jupyter Notebooks

### Prerequisites for Notebooks

1. **Install Jupyter Lab or Jupyter Notebook:**
   ```bash
   # Activate your virtual environment first
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install Jupyter (choose one)
   pip install jupyterlab          # Recommended - modern interface
   # OR
   pip install notebook            # Classic interface
   ```

2. **Install additional notebook dependencies:**
   ```bash
   pip install ipywidgets tqdm matplotlib seaborn pandas
   ```

### Method 1: Using Jupyter Lab (Recommended)

1. **Start Jupyter Lab:**
   ```bash
   # Make sure you're in the project root directory
   cd SwellSight_Project
   
   # Set Python path and start Jupyter Lab
   export PYTHONPATH="."
   jupyter lab
   ```

2. **Your browser will open automatically** showing the Jupyter Lab interface

3. **Navigate and run notebooks:**
   - Click on any `.ipynb` file in the file browser (left panel)
   - Run cells by pressing `Shift + Enter` or clicking the ▶️ button
   - Run all cells with `Run → Run All Cells` from the menu

### Method 2: Using Jupyter Notebook (Classic)

1. **Start Jupyter Notebook:**
   ```bash
   # Make sure you're in the project root directory
   cd SwellSight_Project
   
   # Set Python path and start Jupyter Notebook
   export PYTHONPATH="."
   jupyter notebook
   ```

2. **Your browser will open** showing the file browser

3. **Click on any `.ipynb` file** to open it

4. **Run cells** using `Shift + Enter` or the toolbar buttons

### Method 3: Using VS Code (If you have it installed)

1. **Install Python and Jupyter extensions** in VS Code

2. **Open the project folder** in VS Code

3. **Click on any `.ipynb` file** - VS Code will open it in notebook mode

4. **Select your Python interpreter** (the one from your `.venv` folder)

5. **Run cells** by clicking the ▶️ button next to each cell

### Recommended Notebook Execution Order

**For first-time users:**

1. **Start here:** `01_model_architecture.ipynb`
   - Understand the neural network structure
   - No data required, works immediately

2. **Data setup:** `04_build_real_index.ipynb`
   - Process your downloaded dataset
   - Creates the index files needed for training

3. **Create splits:** `05_create_splits.ipynb`
   - Split data into train/validation/test sets
   - Analyzes data distribution

4. **Explore depth generation:** `06_depth_map_generation.ipynb`
   - Interactive parameter exploration
   - Understand synthetic data generation

5. **Train model:** `07_train_model.ipynb`
   - Full training pipeline with live visualization
   - Requires GPU for best performance

6. **Run inference:** `08_inference.ipynb`
   - Test trained models on new images
   - Analyze model performance

### Troubleshooting Notebook Issues

**Problem: "ModuleNotFoundError: No module named 'swellsight'"**
```bash
# Solution: Set Python path before starting Jupyter
export PYTHONPATH="."
jupyter lab
```

**Problem: "Kernel not found" or Python environment issues**
```bash
# Solution: Install ipykernel in your virtual environment
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name swellsight --display-name "SwellSight"

# Then select "SwellSight" kernel in Jupyter
```

**Problem: Widgets not displaying properly**
```bash
# Solution: Install and enable widget extensions
pip install ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager  # For Jupyter Lab
```

**Problem: Plots not showing**
```bash
# Solution: Install matplotlib backend
pip install matplotlib
# Add this to the first cell of notebooks:
%matplotlib inline
```

### Quick Start Commands

**For Standalone Notebooks (Easiest):**
```bash
# Just start Jupyter - no setup needed!
jupyter lab
# Then open any *_standalone.ipynb file and run all cells
```

**For Interconnected Notebooks (Full Project):**
```bash
cd SwellSight_Project && source .venv/bin/activate && export PYTHONPATH="." && jupyter lab
```

**For Windows users:**
```cmd
cd SwellSight_Project && .venv\Scripts\activate && set PYTHONPATH=. && jupyter lab
```

## Jupyter Notebooks

This project includes comprehensive Jupyter notebooks for interactive development and analysis. **Two versions are available:**

### 🔗 **Interconnected Notebooks** (Original - Removed)
The original interconnected notebooks have been replaced with standalone versions for easier use. All functionality is now available in the standalone notebooks above.

### 🚀 **Standalone Notebooks** (Recommended for Easy Use)
These notebooks include all necessary code and work independently:
- `01_model_architecture_standalone.ipynb` - Complete model definition and testing
- `02_data_loading_standalone.ipynb` - Full data pipeline with dummy data support
- `03_loss_and_metrics_standalone.ipynb` - Loss functions and evaluation metrics
- `04_build_real_index_standalone.ipynb` - Dataset indexing and processing
- `05_create_splits_standalone.ipynb` - Data splitting and analysis
- `06_depth_map_generation_standalone.ipynb` - AI-powered wave image generation
- `07_train_model_standalone.ipynb` - Complete training pipeline
- `08_inference_standalone.ipynb` - Model inference and analysis

### **Which Version Should You Use?**

**Use Standalone Notebooks if:**
- ✅ You want to run notebooks without setting up the full project structure
- ✅ You're exploring individual components
- ✅ You don't have the dataset downloaded yet
- ✅ You want everything to "just work" immediately
- ✅ You want to generate AI-powered wave images
- ✅ You're learning about the SwellSight system

**Use Python Scripts if:**
- ✅ You have the full project setup with dataset
- ✅ You want to run the complete pipeline for production
- ✅ You're doing batch processing or automated workflows
- ✅ You need maximum performance and efficiency

### **Getting Started with Standalone Notebooks:**

1. **No setup required** - just open and run!
2. **Auto-installs dependencies** - notebooks install missing packages
3. **Works with dummy data** - creates sample data if real data is missing
4. **AI image generation** - `06_depth_map_generation_standalone.ipynb` creates realistic wave images
5. **Start with**: `01_model_architecture_standalone.ipynb`

### **Getting Started with Python Scripts:**
1. Ensure you have downloaded the [dataset](https://drive.google.com/drive/folders/11gDkj5GhGVXM9uMlJaVp0b_jIxK_OaoZ?usp=drive_link)
2. Set up the project structure and Python path
3. Follow the Data Pipeline section above for the complete workflow

## Training

Train the SwellSight model on your prepared dataset:

```bash
PYTHONPATH=. python train.py \
  --train_jsonl data/synthetic/mix/train.jsonl \
  --val_jsonl data/synthetic/mix/val.jsonl \
  --out_dir runs/swell_mix_run_01 \
  --epochs 10 \
  --batch_size 8
```

**Training artifacts saved to `runs/swell_mix_run_01/`:**
- `best.pt` - Best model checkpoint
- `last.pt` - Final model checkpoint  
- `history.json` - Training metrics and loss curves
- `vocabs.json` - Label vocabularies for classification tasks

## Evaluation

Evaluate model performance on the test set:

```bash
PYTHONPATH=. python evaluate.py \
  --ckpt runs/swell_mix_run_01/best.pt \
  --test_jsonl data/synthetic/mix/test.jsonl
```

## Inference

Run wave analysis on a single image:

```bash
PYTHONPATH=. python inference.py \
  --ckpt runs/swell_mix_run_01/best.pt \
  --image_path path/to/wave_image.jpg
```

**Example output:**
```
Wave Height: 1.8m
Wave Type: reef_break
Direction: left
```
## Data Formats

### Real Dataset Labels

The `data/real/labels.json` file contains wave parameters keyed by filename:

```json
{
  "image_001.jpg": {
    "height_meters": 1.2,
    "wave_type": "beach_break",
    "direction": "left",
    "confidence": "high",
    "notes": "example note",
    "data_key": 1
  }
}
```

### JSONL Index Format

Each line in `.jsonl` files represents one training sample:

```json
{
  "image_path": "data/synthetic/param_synth_run_*/images/rgb_000001.png",
  "depth_path": "data/synthetic/param_synth_run_*/depth/depth_000001.png",
  "height_meters": 1.2,
  "wave_type": "reef_break",
  "direction": "right",
  "source": "param_synth"
}
```
## Troubleshooting

### Common Issues

#### `ModuleNotFoundError: No module named 'swellsight'`

Ensure you're in the project root directory and have set the Python path:

```bash
source .venv/bin/activate
export PYTHONPATH="."
python -c "import swellsight; print('Import successful')"
```

#### Slow Synthetic Generation

For quick testing, reduce the number of images and diffusion steps:

```bash
OUT_DIR="data/synthetic/quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

PYTHONPATH=. python scripts/03_generate_param_synth.py \
  --out_dir "$OUT_DIR" \
  --n 3 \
  --seed 42 \
  --steps 12 \
  --guidance 6.0 \
  --control_scale 0.55
```

#### Circular Artifacts in Generated Images

If you notice circular artifacts in synthetic images, try:

- Disabling circular occlusions in the depth generator
- Lowering `--control_scale` (e.g., from 0.55 to 0.45)
- Strengthening negative prompts against "circles, rings, bokeh, lens flare"

### Performance Tips

- **GPU Memory**: Reduce batch size if you encounter CUDA out-of-memory errors
- **Training Speed**: Use mixed precision training for faster convergence
- **Data Loading**: Increase `num_workers` in data loaders for faster I/O

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SwellSight in your research or academic work, please cite:

```bibtex
@software{swellsight2026,
  title = {SwellSight: Multi-Task Deep Learning for Wave Analysis from Beach Camera Images},
  author = {SwellSight Project Team},
  year = {2026},
  url = {https://github.com/your-username/SwellSight}
}
```

---

<div align="center">
  <strong>Ride the wave of AI-powered surf analysis!</strong>
</div>