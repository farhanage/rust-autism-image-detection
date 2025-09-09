# Rust Deep Learning Project

A Rust-based deep learning project using the Candle framework for neural network experimentation and learning.

## Project Structure

```
rust_dl/
├── src/
│   ├── config.rs       # Configuration management
│   └── main.rs         # Main application code
├── data/               # Dataset directory (auto-created)
│   ├── raw/            # Original datasets
│   └── processed/      # Preprocessed datasets
├── models/             # Saved model weights (auto-created)
├── Cargo.toml          # Dependencies
└── README.md           # This file
```

## Dataset Configuration

### Default Setup
The application automatically creates and uses:
- `./data/` for dataset storage
- `./models/` for saved model weights

### Custom Paths
You can override the default paths using environment variables:

```bash
export DATA_DIR="/path/to/your/datasets"
export MODELS_DIR="/path/to/your/models"
cargo run
```

### Dataset Cache
Datasets will be automatically downloaded and cached in the data directory on first run.

## Usage

```bash
# Run with default settings
cargo run

# Use custom data directory
DATA_DIR="/opt/datasets" cargo run

# Use custom model directory
MODELS_DIR="/opt/models" cargo run
```

## Models
- **Linear**: Simple linear classifier
- **MLP**: Multi-layer perceptron with ReLU activation
- **CNN**: Convolutional neural network (default)

Change the `model_type` variable in `main.rs` to switch between models.

## Dependencies
- `candle-core`: Core tensor operations
- `candle-nn`: Neural network layers
- `candle-datasets`: Dataset loading utilities
- `anyhow`: Error handling
