# Active Learning Project

## Description
The **Active Learning Project** aims to explore and evaluate multiple active learning strategies to enhance the training process of machine learning models. By selecting the most informative data points to be labeled, the project seeks to improve model performance while minimizing the labeling budget. This repository provides an implementation of different active learning techniques using a Python pipeline, including Random Sampling, Uncertainty Sampling, and other innovative approaches.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/Active-Learning-Project.git
   ```

2. Change into the project directory:
   ```bash
   cd Active-Learning-Project
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the active learning pipeline, make sure you have a dataset in CSV format. The pipeline expects a file path for the dataset and configurations for data splits and iterations.

To execute the main pipeline:

```bash
python active_learning_pipeline.py
```

You can modify the dataset, features, and sampling methods by changing the parameters in the script or passing them as arguments.

## Features

### 1. Active Learning Strategies
The **ActiveLearningPipeline** class provides several strategies for data sampling:
- **Random Sampling**: Randomly selects samples without any specific heuristic.
- **Uncertainty Sampling**: Selects samples the model is least certain about to maximize learning efficiency.
- **Diversity Sampling**: Selects diverse samples based on pairwise distance to ensure varied data.
- **Density-Weighted Uncertainty Sampling**: A combination of density-based selection and uncertainty.
- **Query by Committee (QBC)**: Uses multiple models to select samples with the highest disagreement.
- **Risk-Based Sampling**: Weights features by correlation to label interest to enhance sampling efficiency.
- **Multi-Armed Bandit (MAB) Strategy**: Uses a multi-armed bandit approach to dynamically select the best sampling method.

### 2. Custom Visualization Tools
The **visualizations.py** script provides functions to visualize the performance of the active learning strategies:
- **Accuracy Plots**: Track accuracy improvement across iterations.
- **Comparison Charts**: Compare scores of different sampling methods on various datasets.

You can generate plots using functions like `generate_plot()` and `plot_all_datasets_results()` to visualize and compare different sampling methods.

## Files
- **active_learning_pipeline.py**: Implements the core active learning pipeline, offering multiple sampling methods and training strategies for different datasets.
- **visualizations.py**: Provides utility functions to generate plots comparing the effectiveness of different active learning methods.
- **process_data.ipynb**: Jupyter notebook for data preprocessing, exploration, and preparation prior to running the main active learning pipeline.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeatureName
   ```
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
