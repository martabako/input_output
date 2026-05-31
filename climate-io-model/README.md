# Climate Input-Output Model

This project focuses on developing an input-output model to analyze climate impacts and decarbonization strategies. The model aims to provide insights into how different sectors contribute to greenhouse gas emissions and how various decarbonization strategies can mitigate these impacts.

## Project Structure

- **notebooks/**: Contains Jupyter Notebooks for data exploration, model development, and scenario analysis.
  - `01-data-exploration.ipynb`: Explore the dataset and visualize data distributions.
  - `02-input-output-model.ipynb`: Develop the input-output model and analyze results.
  - `03-scenario-analysis.ipynb`: Conduct scenario analysis on decarbonization strategies.

- **src/**: Contains the source code for data loading, processing, modeling, analysis, and visualization.
  - **data/**: Functions for loading and processing data.
    - `loaders.py`: Load raw data from various sources.
    - `processing.py`: Process and clean the loaded data.
  - **model/**: Implementation of the input-output model.
    - `io_model.py`: Define the Input-Output Model class.
    - `calibration.py`: Calibrate model parameters.
  - **analysis/**: Functions for calculating decarbonization metrics.
    - `decarbonization_metrics.py`: Calculate key metrics related to decarbonization.
  - **viz/**: Functions for visualizing analysis results.
    - `plotting.py`: Visualize emissions trajectories and model outputs.

- **data/**: Contains raw and processed datasets.
  - **raw/**: Store raw datasets used for analysis.
  - **processed/**: Store cleaned and transformed datasets.

- **tests/**: Contains unit tests for the project.
  - `test_loaders.py`: Unit tests for data loading functions.
  - `test_io_model.py`: Unit tests for the input-output model functions.

- **environment.yml**: Conda environment configuration file.

- **requirements.txt**: List of required Python packages.

- **.gitignore**: Specifies files and directories to ignore in version control.

## Installation

To set up the project environment, you can use either conda or pip.

### Using Conda

```bash
conda env create -f environment.yml
conda activate climate-io-model
```

### Using Pip

```bash
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open the notebooks in the `notebooks/` directory to explore the data, develop the model, and conduct scenario analyses.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.