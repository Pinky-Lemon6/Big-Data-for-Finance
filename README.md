# Big-Data-for-Finance
## About This Repository
This repository contains resources and materials for the course "Big Data for Finance Government." It includes lectures, assignments, and projects that demonstrate the application of big data technologies in the finance government sector. 

## Repository Structure
- **Dataset/**: Contains datasets used for analysis.
- **Model/**: Stores model data and outputs.
- **PPT/**: Includes course presentation materials.
## Dataset

The **Dataset/** folder contains the S&P 500 dataset, which includes the following files:

- **all_stocks_5yr.csv**: The original dataset.
- **file_list.csv**: A list of CSV sub-files that need to be read.
- **all_stocks_5yr_clean_9T_train.csv**: The preprocessed training set.
- **all_stocks_5yr_clean_9T_test.csv**: The preprocessed testing set.

## Model

The **Model/** folder stores the trained model data. The naming convention for the files is `transformer_model_epoch(i).pth`, where `i` represents the number of epochs the model was trained for. This allows for easy identification of the model's training duration and performance.

## PPT

The **PPT/** folder includes materials for classroom presentations. Notable files are:

- **ESWA2022_2.pdf**: A PDF document containing references for the course.
- **pre.pptx**: The PowerPoint presentation used for classroom demonstrations.

## Src

The **src/** folder contains essential Python scripts for the project:

- **StockDataset.py**: Defines the dataset used for training and testing the model.
- **TransformerModel.py**: Contains the architecture of the constructed model.
- **utils.py**: Includes a collection of utility functions used during model construction, training, and testing.
- **model_train.py**: Implements the training process for the model.
- **model_test.py**: Handles the testing process to evaluate model performance.
## Usage Instructions

To utilize the resources in this repository, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Pinky-Lemon6/Big-Data-for-Finance.git
    ```

2. **Navigate to the Directory**:
    ```bash
    cd Big-Data-for-Finance
    ```

3. **Install Required Packages**:
    Ensure you have the necessary Python packages installed. You can use pip to install them:
    ```bash
    pip install -r requirements.txt
    ```


4. **Train the Model**:
    To train the model, run the `model_train.py` script:
    ```bash
    python3 model_train.py
    ```

5. **Test the Model**:
    After training, evaluate the model's performance using:
    ```bash
    python3 model_test.py
    ```

6. **View Presentations**:
    Open the PowerPoint files in the `PPT/` folder for course materials.

Make sure to adjust file paths as necessary based on your local setup.




