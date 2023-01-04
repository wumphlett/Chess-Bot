# Chess Engine Exploration

## Prerequisites
- Environment properly setup with for use of cuda with tensorflow is preferred
- May have to install 7z executable depending on the platform
- Conda

## Usage
- `conda env create -f environment.yml`
- `conda activate chess`
- `jupyter notebook`
- Training and benchmarking performed in `project.ipynb`

## Notebook
- Setup: Configures notebook
- Environment: Downloads, unpacks, processes, and splits the dataset
- Pos2Vec: Implements greedy layer-wise pretraining in prep for;
- DeepChess: Uses a siamese network to compare two given positions and determine which is best
- Evaluation: Evaluates the implementation of DeepChess
- Play: Play chess against the completed engine
- Sandbox: Delete upon completion
