# Project Run Instructions: Water Body Segmentation

This guide outlines the necessary steps to set up and run the water body segmentation pipeline from data ingestion to model testing.

## Prerequisites

*   Python 3.x installed.
*   The project repository cloned to your local machine.
*   All required Python dependencies installed (see "Installation" section below).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your GitHub Repository URL]
    cd [Your Project's Main Directory Name]
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate     # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: You'll need a `requirements.txt` file in your project root containing all project dependencies, which can be generated using `pip freeze > requirements.txt`.)

## Usage Instructions

### 1. Prepare Your Raw Data

Ensure your raw satellite images and their corresponding water masks are organized in the following directory structure within your project:

    water-bodies/
    └── data/
        └── raw/
            ├── images/  (Contains image files)
            └── masks/   (Contains corresponding mask files with same filenames)


### 2. Run Data Ingestion & Preprocessing (`ingest_data.py`)

This script loads raw images and masks, applies resizing, normalization, binarization, data augmentation, and tiles them into smaller patches suitable for model training.

*   **Execute:**
    ```bash
    python ingest_data.py
    ```
*   **Output:** Processed images and masks will be saved to `data/processed/images/` and `data/processed/masks/`.

### 3. Run Model Training & Experiment Tracking (`model_trainer1.py`)

This script initiates the training process for multiple segmentation models (U-Net, DeepLabV3, FPN) and rigorously logs all training parameters, metrics, and model artifacts using MLflow.

*   **Execute:**
    ```bash
    python model_trainer1.py
    ```
*   **Output:** Training progress will be displayed in the console. MLflow run data (including best model checkpoints) will be stored in the `mlruns/` directory in your project root.
*   **View MLflow UI:** To visualize training runs and compare models in a web interface:
    ```bash
    mlflow ui
    ```
    Then, open your web browser and navigate to `http://localhost:5000`.

### 4. Run Model Testing (`tester.py`)

This script evaluates the performance of your trained models on a held-out test dataset, providing detailed evaluation metrics and saving visual predictions. You will need the MLflow Run IDs from your training step (which can be found in the MLflow UI).

*   **Execute (Example - Replace placeholders with your actual Run IDs and model names):**
    ```bash
    python tester.py --run_ids <run_id_for_unet_resnet34> <run_id_for_fpn_resnet34> <run_id_for_unet_resnet50> <run_id_for_deeplabv3_resnet34> --models unet_resnet34 fpn_resnet34 unet_resnet50 deeplabv3_resnet34
    ```
    *To test a single model from a local path:*
    ```bash
    python tester.py --single_model unet_resnet34 --model_path path/to/your/best_unet_resnet34_model.pth
    ```
*   **Output:** Test results will be printed to the console. Reports and sample prediction images will be saved to the `test_results/` directory.

### 5. Run Inference Service & Deployment

This section describes how to build and run the Dockerized inference service for real-time water body segmentation.

*   **Build Docker Image:**
    # docker build -t water: .

*   **Run Docker Container:**
        docker run --rm -v ${PWD}:/app water:latest --input input --output output

        You can also run python app.py to inference the best model