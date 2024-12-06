## About This Repository
This repository contains resources and materials for the course "Big Data for Finance Government." It includes lectures, assignments, and projects that demonstrate the application of big data technologies in the finance government sector. 

## Repository Structure
- **Dataset/**: Contains datasets used for analysis.
- **model/**: Stores model data and outputs.
- **src/**: Contains all the code files of this project.
## Dataset

The **Dataset/** folder contains the S&P 500 dataset, all data files are saved as `.csv` files in the `individual_stocks_5yr/` folder under this directory. There are 506 csv files in total.

## model

The **model/** folder stores the trained model data. The naming convention for the files is `transformer_model_epoch(i).pth`, where `i` represents the number of epochs the model was trained for. This allows for easy identification of the model's training duration and performance.

## src
The **src/** folder contains essential Python scripts for the project:

- **DataLoader.py**: Defines the dataset used for training and testing the model.
- **TransformerModel.py**: Contains the architecture of the constructed model.
- **utils.py**: Includes a collection of utility functions used during model construction, training, and testing.
- **model_train.py**: Implements the training process for the model.
- **model_test.py**: Handles the testing process to evaluate model performance.

## Usage Instructions

To utilize the resources in this repository, follow these steps:

1. **Navigate to the Directory**:
    ```bash
    cd src/
    ```

2. **Train the Model**:
    To train the model, run the `model_train.py` script:
    ```bash
    python3 model_train.py
    ```

3. **Test the Model**:
    After training, evaluate the model's performance using:
    ```bash
    python3 model_test.py
    ```

4. **View Presentations**:
    After successfully running the test, the image files will be saved in the `output/` folder. You can view them by going into the folder.
