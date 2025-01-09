# Data Details: FER2013 Dataset

The *FER2013 (Facial Expression Recognition)* dataset is a widely-used dataset for emotion detection tasks in deep learning and machine learning. This dataset contains grayscale images of faces, each labeled with one of seven emotion categories.

---

## Source

The dataset was sourced from Kaggle:

*[Facial Expression Recognition on Kaggle](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)*

You can download the dataset from the above link.

---

## Dataset Overview

•⁠  ⁠*Number of Samples*: 35,887
•⁠  ⁠*Image Dimensions*: 48x48 pixels
•⁠  ⁠*Number of Channels*: 1 (Grayscale)
•⁠  ⁠*Number of Classes*: 7 (Emotion categories)
•⁠  ⁠*File Format*: ⁠ .csv ⁠

---

## Emotion Categories

The dataset contains the following emotion labels:

| Label | Emotion   |
|-------|-----------|
| 0     | Angry     |
| 1     | Disgust   |
| 2     | Fear      |
| 3     | Happy     |
| 4     | Sad       |
| 5     | Surprise  |
| 6     | Neutral   |

---

## Data Structure

The dataset is provided as a CSV file with the following columns:

1.⁠ ⁠*emotion*:
   - The target label for each image, represented as an integer (0-6).
   - Maps to one of the seven emotion categories.

2.⁠ ⁠*pixels*:
   - A string of space-separated pixel values (2304 values representing a 48x48 grayscale image).

3.⁠ ⁠*Usage*:
   - Specifies the purpose of the image:
     - ⁠ Training ⁠: Used for training the model.
     - ⁠ PublicTest ⁠: Used for validation during model training.
     - ⁠ PrivateTest ⁠: Used for testing the model after training.

---

## Dataset Splits

The dataset is split into three subsets based on the ⁠ Usage ⁠ column:

| Usage       | Number of Samples | Percentage |
|-------------|-------------------|------------|
| Training    | 28,709            | ~80%       |
| PublicTest  | 3,589             | ~10%       |
| PrivateTest | 3,589             | ~10%       |

---

## Preprocessing

Before using the dataset, the following preprocessing steps are applied:
1.⁠ ⁠*Normalization*:
   - Pixel values are normalized to the range ⁠ [0, 1] ⁠ by dividing each value by 255.
2.⁠ ⁠*Reshaping*:
   - The 1D pixel values are reshaped into a 2D array with dimensions ⁠ 48x48 ⁠.
3.⁠ ⁠*Tensor Conversion*:
   - Images and labels are converted into PyTorch tensors for model training and evaluation.

---

## How to Load the Dataset

The dataset is loaded and split in the project using the ⁠ data_loader.py ⁠ script. The ⁠ Usage ⁠ column is used to split the data into training, validation, and testing sets.

### Example Code:
```python
from src.data_loader import load_data
from src.config import Config

# Path to FER2013 CSV file
csv_file = "data/fer2013.csv"

# Load DataLoaders
train_loader, val_loader, test_loader = load_data(csv_file, Config.batch_size)

print(f"Training data size: {len(train_loader.dataset)}")
print(f"Validation data size: {len(val_loader.dataset)}")
print(f"Testing data size: {len(test_loader.dataset)}")