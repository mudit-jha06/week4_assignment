# Week 4 Assignment: Data Cleaning and training a model for Outdoor Scene Recognition

## IMPORTANT

For us to verify your submission, you need to create a pull request and the action will be triggered automatically. However, sometimes actions arent triggered automatically. In that case, you can close the PR and create a new PR by selecting the main and submission (where you pushed code) branch. Even merging the PR will trigger the action. Happy learning!

## Overview

In this assignment, you will work with a **corrupted** Outdoor Scene Recognition dataset that contains two common real-world data quality issues:

1. **Mislabeled images** - Some images have incorrect class labels
2. **Class imbalance** - Some classes have significantly fewer samples than others

Your task is to:
1. Perform **Exploratory Data Analysis (EDA)** to understand the dataset
2. **Identify and remove mislabeled images** using loss-based detection
3. **Handle class imbalance** through data augmentation
4. Train and compare **baseline vs improved models**

Your models will be automatically evaluated via **GitHub Actions** on a held-out validation set.

---

## Dataset

**Outdoor Scene Classification Dataset**
- **6 Classes**: buildings, forest, glacier, mountain, sea, street
- **~14,034 training images** (corrupted version)
- **~3,000 validation images** (clean version for local testing)
- **Image size**: 150x150 pixels

The dataset has been intentionally corrupted - your job is to fix it!

### Download Dataset

📥 **Download the dataset from Google Drive:**

[**Download Week 4 Dataset**](https://drive.google.com/drive/folders/1vcEiOn3fu4l4kTXncMA4Pi1Sn8E22OSP)

The downloaded zip file contains:
```
dataset/
├── train/              # Corrupted training data (~14,034 images)
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── validation_data/    # Clean validation data (~3,000 images)
    ├── buildings/
    ├── forest/
    ├── glacier/
    ├── mountain/
    ├── sea/
    └── street/
```

**After downloading:**
1. Extract the zip file
2. Place the `dataset/` folder in your project directory
3. Use `validation_data/` to test your models locally

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: EDA                                                        │
│  - Count images per class                                           │
│  - Visualize class distribution                                     │
│  - Apply PCA to understand feature space                            │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Train Baseline Model                                       │
│  - Train on corrupted dataset                                       │
│  - Save model as baseline_model.pth                                 │
│  - Export to baseline_model.onnx                                    │
│  - Generate confusion matrix                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Identify Mislabels                                         │
│  - Calculate per-sample loss                                        │
│  - High-loss samples = likely mislabeled                            │
│  - Visualize with FiftyOne (Voxel51)                                │
│  - Tag and remove confirmed mislabels                               │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: Handle Class Imbalance                                     │
│  - Analyze class distribution after cleaning                        │
│  - Augment underrepresented classes                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: Train Improved Model                                       │
│  - Train on cleaned + augmented dataset                             │
│  - Save model as improved_model.pth                                 │
│  - Export to improved_model.onnx                                    │
│  - Generate confusion matrix                                        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌───────────────────────────────────────────────────────────────────────┐
│  Step 6: Local Evaluation                                             │
│  - Test both models on validation_data using the test_models.py script│
│  - Verify improved model beats baseline                               │
│  - Gain confidence before submission                                  │
└───────────────────────────────────────────────────────────────────────┘
                                    ↓
┌───────────────────────────────────────────────────────────────────────┐
│  Step 7: Export & Submit to GitHub                                    │
│  - Convert models to ONNX format using convert_to_onnx.py             │
│  - Push .onnx files to your repository and create a PR                │
│  - GitHub Actions will evaluate on validation set                     │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Exploratory Data Analysis (EDA) 

Before fixing the data, understand what you're working with.

### 1.1 Count Images Per Class

### 1.2 Visualize Sample Images

### 1.3 PCA Visualization 

Use PCA to visualize the feature space and identify potential outliers:

**What to look for:**
- Clusters that overlap heavily (confusable classes)
- Outliers far from their cluster (potential mislabels)
- Class separation quality

---

## Step 2: Train Baseline Model

Train a model on the **corrupted dataset** to establish a baseline.

This will:
1. Train a model on the corrupted data
2. Save the model as baseline_model.pth
3. Calculate per-sample losses for mislabel detection

### Generate Baseline Confusion Matrix

**Save these outputs** - you'll compare them with your improved model later.

---

## Step 3: Identify and Remove Mislabels

### 3.1 Why This Works

When training on mislabeled data:
- **Correctly labeled images**: Model learns consistent patterns → **low loss**
- **Mislabeled images**: Model sees contradictory patterns → **high loss**

By sorting samples by loss, mislabeled images naturally surface at the top.

### 3.2 Visualize with FiftyOne (Voxel51)

This opens an interactive web interface where you can:
1. **Filter** to suspicious (high-loss) samples
2. **Sort** by loss value
3. **Inspect** images and compare labels vs predictions
4. **Tag** confirmed mislabels

### 3.3 How to Use FiftyOne

1. **Filter suspicious samples**: Click "Add filter" → "tags" → select "suspicious"
2. **Sort by loss**: Click on "loss" column header (descending)
3. **Review images**: Click on each image to see details
4. **Tag mislabels**: Select images → Click "Tag" → Enter tag name (e.g., "mislabeled")
5. **Create a saved view**: Save your filtered view for export

### 3.4 Export Cleaned Dataset

After tagging mislabels in FiftyOne, remove all tagged mislabels and exports a clean dataset.

---

## Step 4: Handle Class Imbalance

After removing mislabels, check class distribution:

### 4.1 Identify Underrepresented Classes

Compare class sizes. Classes with significantly fewer samples need augmentation.

### 4.2 Apply Data Augmentation

**Augmentation techniques examples:**
- Horizontal flips
- Rotation (±10°)
- Brightness adjustment (±20%)
- Contrast adjustment (±30%)
- Saturation changes
- Gaussian blur

### 4.3 Why Augmentation Helps

Class imbalance causes models to:
- Bias predictions toward majority classes
- Underperform on minority classes
- Overfit to limited samples

Augmentation creates synthetic variations, helping the model generalize better.

---

## Step 5: Train Improved Model

Train on the cleaned and augmented dataset:

### Generate Improved Confusion Matrix

### Compare Results

| Metric | Baseline | Improved |
|--------|----------|----------|
| Accuracy | ??% | ??% |
| Macro F1 | ?.???? | ?.???? |
| buildings F1 | ?.???? | ?.???? |
| forest F1 | ?.???? | ?.???? |
| glacier F1 | ?.???? | ?.???? |
| mountain F1 | ?.???? | ?.???? |
| sea F1 | ?.???? | ?.???? |
| street F1 | ?.???? | ?.???? |

---

## Step 6: Local Evaluation

**Before submitting to GitHub, test your models locally!**

### 6.1 Test Your Models

Use the provided validation data to verify your improved model is better:

```bash
# Test your models on the validation_data folder
python test_models.py \
    --baseline ./models/baseline_model.pth \
    --improved ./models/improved_model.pth \
    --test-data ./dataset/validation_data
```

This will:
- Evaluate both models on the clean validation data
- Show accuracy and F1 scores for each model
- Verify that your improved model is better than baseline
- Give you confidence before submitting

**Expected Output:**
```
✅ PASS: Improved model F1 score is better than baseline 
```

### 6.2 Why Test Locally?

- **Catch issues early**: Verify models work before submission
- **Debug problems**: Fix any model loading or evaluation issues
- **Build confidence**: Know your submission will pass before creating PR
- **Save time**: Avoid GitHub Actions failures and resubmissions

**Note**: The local validation data is different from the private validation set used by GitHub Actions. This prevents overfitting while still allowing you to test locally.

---

## Step 7: Export & Submit to GitHub

### 7.1 Convert Models to ONNX Format

Before submitting, you **must** convert your PyTorch models to ONNX format. This allows us to evaluate any model architecture.

```bash
# Convert your .pth models to .onnx format
python convert_to_onnx.py
```

This will create:
- `models/baseline_model.onnx` + `models/baseline_model.onnx.data`
- `models/improved_model.onnx` + `models/improved_model.onnx.data`

### 7.2 Required Directory Structure

Your repository **must** follow this structure:

```
your-repo/
├── models/                              # ⚠️ REQUIRED
│   ├── baseline_model.onnx             # ⚠️ REQUIRED - Baseline model (ONNX format)
│   ├── baseline_model.onnx.data        # ⚠️ REQUIRED - Baseline model weights
│   ├── improved_model.onnx             # ⚠️ REQUIRED - Improved model (ONNX format)
│   ├── improved_model.onnx.data        # ⚠️ REQUIRED - Improved model weights
│   ├── baseline_model.pth              # Optional - PyTorch checkpoint
│   └── improved_model.pth              # Optional - PyTorch checkpoint
├── evaluation/
│   ├── baseline_confusion_matrix.png
│   ├── improved_confusion_matrix.png
│   └── comparison_table.md
├── analysis/
│   ├── class_distribution.png
│   ├── pca_visualization.png 
│   └── mislabel_analysis.json
├── convert_to_onnx.py                     # Script to convert models
├── dataset/                               # dataset (from Google Drive)
└── README.md                           
```

**Critical Requirements:**
- ✅ `models/baseline_model.onnx` + `.onnx.data` - Baseline model in ONNX format
- ✅ `models/improved_model.onnx` + `.onnx.data` - Improved model in ONNX format
- ✅ Models must accept input shape: `(batch, 3, 150, 150)`
- ✅ Models must output logits for 6 classes 

**Note:** Do NOT include the `train/` or `validation_data/` folders in your GitHub repository (add them to `.gitignore`). They are too large and should only be stored locally.

### 7.3 GitHub Actions Evaluation

When you create a **pull request**, **GitHub Actions will automatically**:
1. Clone a private validation dataset (different from your local validation_data)
2. Load your `baseline_model.onnx` and `improved_model.onnx`
3. Evaluate both models on the private validation set
4. Post results as a comment on your pull request

**Important Notes:**
- The private validation dataset is **different** from your local `validation_data/`
- This ensures you're not overfitting to the validation set
- Your improved model **must** have a higher F1 score than baseline to pass
- **You can use any model architecture** - ONNX format allows architecture-agnostic evaluation

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PROJECT_NAME.git
cd PROJECT_NAME
```

### 2. Download Dataset

1. Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1vcEiOn3fu4l4kTXncMA4Pi1Sn8E22OSP)
2. Extract the zip file
3. Move `dataset/` folder to your project root

### 3. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Verify Setup

Your directory should look like:
```
week4_assignment/                 
├── dataset/        # From Google Drive
├── models/                 # Create this folder
├── requirements.txt
└── README.md
```

---

## Tips for Success

1. **Do EDA first**: Understand the data before fixing it
2. **Train baseline before cleaning**: You need a reference point
3. **Be conservative with mislabel removal**: When in doubt, don't remove
4. **Focus on high-loss samples**: They're most likely mislabeled
5. **Don't over-augment**: 2-3x augmentation is usually enough
6. **Document everything**: Screenshots, statistics, observations
7. **Compare before/after**: Quantify your improvements

---

## FAQ

**Q: How do I know if an image is mislabeled?**
A: Look at the image content and compare to the assigned label. If a "glacier" image clearly shows a mountain with no ice, it's likely mislabeled.

**Q: What if I'm unsure about an image?**
A: When in doubt, don't tag it. Some images are ambiguous.

**Q: How much augmentation is enough?**
A: Aim to balance classes to within 20% of each other.

**Q: Why can't I see the validation dataset?**
A: To ensure fair evaluation. Your model should generalize, not memorize.

**Q: What if GitHub Actions fails?**
A: Check that:
   - Your models are in `models/baseline_model.onnx` and `models/improved_model.onnx`
   - Both `.onnx` and `.onnx.data` files are committed
   - You ran `python convert_to_onnx.py` to convert your models
   - Your improved model actually performs better than baseline

**Q: Why ONNX format instead of .pth?**
A: ONNX (Open Neural Network Exchange) includes both the model architecture and weights, so we can evaluate any model you create without knowing the exact architecture you used. This gives you freedom to experiment with different model designs.

**Q: How do I convert my model to ONNX?**
A: Run `python convert_to_onnx.py` after training. Make sure your `.pth` files are in the `models/` folder first.

**Q: Can I test my models before submitting?**
A: Yes! Use the provided `validation_data/` folder to test locally. This gives you confidence before creating a pull request.

**Q: Why are there two validation datasets?**
A: The local `validation_data/` is for your testing. GitHub Actions uses a different private validation set to prevent overfitting.

---

## Resources

- [FiftyOne Documentation](https://docs.voxel51.com/)
- [PyTorch Data Augmentation](https://pytorch.org/vision/stable/transforms.html)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Confident Learning Paper](https://arxiv.org/abs/1911.00068)

---

## Getting Help

- Consult FiftyOne documentation for visualization questions
- Ask your instructor/TAs during office hours

Good luck! 🚀
