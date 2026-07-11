# Skin Lesion Classifier: ANN + CNN Tutorial

## What We're Doing

Train **two different models** on **two different datasets**, then compare them to see which approach works better for skin lesion classification.

| Model | Dataset | Input Type | Task |
|-------|---------|------------|------|
| **ANN** | HAM10000 metadata | Tabular (age, sex, location) | Predict 7 lesion types |
| **CNN** | DDI images | Pixels (656 PNGs) | Predict malignant vs benign |

**Why two models?** The ANN tests whether structured patient data alone can classify lesions. The CNN tests whether visual patterns in images can. Comparing them tells you if images add value beyond basic demographics.

---

## Part 0: Setup

Run this first in your notebook.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
```

> **Edge case to watch**: Different machines have different TensorFlow versions. Use `tf.config.list_physical_devices()` to check if a GPU is available — CNNs are painfully slow on CPU (like 10x+ slower).

---

## Part 1: ANN — Predict Lesion Type from Tabular Data

### 1.1 Load HAM10000 metadata

```python
url = "https://huggingface.co/datasets/ShiroOnigami23/skin-cancer-ham10000-dataset/resolve/main/HAM10000_metadata.csv"
df = pd.read_csv(url)
```

Columns: `lesion_id, image_id, dx, dx_type, age, sex, localization`

**Target**: `dx` — the diagnosis (7 classes: nv, mel, bkl, bcc, akiec, vasc, df)

### 1.2 Clean the data

```python
# Drop rows with missing age or sex
df = df.dropna(subset=['age', 'sex']).reset_index(drop=True)

# Check class balance — this is CRITICAL
df['dx'].value_counts()
```

> **Edge case #1 — Class imbalance**: Look at the counts. `nv` (melanocytic nevi) will dominate (~67%). If you train on this raw, the ANN will learn to predict "nv" for everything and get 67% accuracy without learning anything. This is the #1 trap.

### 1.3 Encode features for the ANN

ANNs only understand numbers, not text. You need to convert:

- **`sex`** → male=0, female=1 (LabelEncoder)
- **`localization`** → one-hot encode (17+ body locations)
- **`age`** → already numeric, but should be scaled

```python
# Encode sex
df['sex_enc'] = LabelEncoder().fit_transform(df['sex'])

# One-hot encode localization
loc_dummies = pd.get_dummies(df['localization'], prefix='loc')

# Prepare features
X = pd.concat([
    df[['age', 'sex_enc']],
    loc_dummies
], axis=1)

# Handle missing ages (fill with median)
X['age'] = X['age'].fillna(X['age'].median())

# Encode target (dx)
y = LabelEncoder().fit_transform(df['dx'])  # 0-6
num_classes = len(np.unique(y))
```

> **Edge case #2 — Missing ages**: Your EDA showed some rows have missing age. If you don't handle this, sklearn will silently drop them or crash. Fill with median, not mean (median is robust to outliers).

### 1.4 Scale features

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Why scale? ANNs use gradient descent. Features on different scales (age=80 vs one-hot=0/1) cause unstable gradients.

### 1.5 Train/validation/test split

```python
# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
)

# Note: 0.1 of 0.8 = 0.08 of total → 72/8/20 split
```

> **Edge case #3 — `stratify=y`**: Always use `stratify` for classification. Without it, the test set might randomly get 0% of the rare classes (like `vasc` or `df`), making evaluation meaningless.

### 1.6 Build the ANN

```python
model_ann = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model_ann.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_ann.summary()
```

**Architecture explained:**
- **Input layer**: one neuron per feature (age + sex + one-hot locations)
- **Dense(128, relu)**: learns feature combinations (e.g., "older males on ears → likely BCC")
- **Dropout(0.3)**: randomly turns off 30% of neurons during training to prevent overfitting
- **Dense(64, relu)**: deeper abstraction
- **Dense(7, softmax)**: outputs probabilities for each class, sums to 1

> **Edge case #4 — How many neurons?** There's no magic formula. Rule of thumb: start with powers of 2 (128, 64, 32). Too few → underfits. Too many → overfits. Dropout helps with the latter.

### 1.7 Train the ANN

```python
history = model_ann.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)
```

> **Edge case #5 — EarlyStopping**: Without it, the model will keep training past the point where validation loss starts rising (overfitting). `patience=5` means "stop if val_loss hasn't improved for 5 epochs." `restore_best_weights` rolls back to the best epoch.

### 1.8 Evaluate the ANN

```python
y_pred = np.argmax(model_ann.predict(X_test), axis=1)

print(classification_report(y_test, y_pred, target_names=df['dx'].unique()))

# Recall per class — critical for your fairness audit
class_recall = recall_score(y_test, y_pred, average=None)
for i, cls in enumerate(np.unique(df['dx'])):
    print(f"{cls}: recall = {class_recall[i]:.3f}")
```

> **Why recall matters**: Your proposal says recall is the key metric. A model with 90% accuracy but 0% recall on melanoma is **dangerous**. The ANN will likely have terrible recall on rare classes.

---

## Part 2: CNN — Classify Lesions from Images (DDI Dataset)

### 2.1 Load DDI metadata

```python
import os

# Edit this path to match your setup
DDI_DIR = "ddidiversedermatologyimages/ddidiversedermatologyimages"
ddi_meta = pd.read_csv(os.path.join(DDI_DIR, "ddi_metadata.csv"))
```

Columns: `DDI_ID, DDI_file, skin_tone, malignant, disease`

### 2.2 Explore DDI

```python
print(ddi_meta['malignant'].value_counts())
print(ddi_meta['skin_tone'].value_counts())
```

**What you'll see**: Most images are malignant=True (many cancers), and skin_tone values are 12, 34, 56 — these are **combined Fitzpatrick scores** (the first digit = skin type 1-6, the second = how well it tans).

### 2.3 Load images

Images are 450x600 PNGs. You can't feed raw large images into a CNN efficiently. Resize them.

```python
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128  # resize to 128x128

def load_ddi_images(meta_df, img_dir, img_size=IMG_SIZE):
    images = []
    labels = []
    for _, row in meta_df.iterrows():
        img_path = os.path.join(img_dir, row['DDI_file'])
        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = image.img_to_array(img) / 255.0  # normalize pixel values to [0,1]
        images.append(img)
        labels.append(1 if row['malignant'] else 0)  # True=1, False=0
    return np.array(images), np.array(labels)

X_img, y_img = load_ddi_images(ddi_meta, DDI_DIR)
print(f"Loaded {X_img.shape[0]} images, shape: {X_img.shape}")
```

> **Edge case #6 — Normalization**: Dividing by 255 scales pixels to [0,1]. If you skip this, the CNN will struggle to converge (big pixel values = big gradients = unstable training). Some people use [−1,1] normalization instead — pick one and be consistent.

> **Edge case #7 — `target_size`**: Resizing to 128x128 loses detail but makes training feasible. A 450x600 image = 810,000 pixels per channel. At 128x128 = 49,152 pixels. That's 16x less data to process. You can try 224x224 if you have a GPU.

### 2.4 Split DDI data

```python
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img, y_img, test_size=0.2, random_state=42, stratify=y_img
)

X_train_img, X_val_img, y_train_img, y_val_img = train_test_split(
    X_train_img, y_train_img, test_size=0.1, random_state=42, stratify=y_train_img
)
```

> **Edge case #8 — Small dataset**: 656 images is tiny for a CNN. After split, you'll have ~470 training images. The model will overfit badly without data augmentation. Your proposal mentions augmentation — now's the time.

### 2.5 Data augmentation (CRITICAL)

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
```

This generates slightly different versions of each image every epoch — flipping, rotating, zooming. It effectively multiplies your dataset size.

> **Edge case #9 — What NOT to augment**: Don't use extreme rotations (>15°) or vertical flips on medical images. A skin lesion rotated 90° is still a lesion, but 180° might look unnatural. Medical images have orientation conventions.

### 2.6 Build the CNN

```python
model_cnn = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    # Classifier head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model_cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)

model_cnn.summary()
```

**Architecture explained:**
- **Conv2D(32, 3x3)**: 32 filters slide across the image, each detecting a pattern (edges, colors, textures). Output = 32 "feature maps" of same size.
- **MaxPooling2D(2,2)**: Downsamples 2x, keeps only the strongest activation in each 2x2 block. Makes the model focus on what matters, not where.
- **Depth increases** (32→64→128) as spatial size shrinks — classic pattern: more filters on smaller images.
- **Flatten**: Converts 2D feature maps into 1D vector for the dense layers.
- **Dropout(0.5)**: Stronger dropout (50%) because the dataset is tiny.
- **Sigmoid**: Outputs a probability between 0 and 1 (benign vs malignant).

> **Edge case #10 — Binary vs multi-class**: Use `sigmoid` + `binary_crossentropy` for 2 classes. Use `softmax` + `categorical_crossentropy` for 7 classes. Mixing them up will silently give wrong results.

### 2.7 Train the CNN

```python
history_cnn = model_cnn.fit(
    X_train_img, y_train_img,
    validation_data=(X_val_img, y_val_img),
    epochs=100,
    batch_size=16,  # small batch for small dataset
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)  # lower LR when stuck
    ],
    verbose=1
)
```

> **Edge case #11 — Batch size**: Batch size 16 is small. Small batches = noisy gradients (helps generalization) but slower per epoch. Large batches = stable gradients but can overfit. For 470 training images, batch 16 gives ~30 steps per epoch.

> **Edge case #12 — ReduceLROnPlateau**: When validation loss plateaus, cutting the learning rate helps the model escape local minima. Without it, training often stalls at a mediocre result.

### 2.8 Evaluate the CNN

```python
y_pred_img = (model_cnn.predict(X_test_img) > 0.5).astype(int)

print(classification_report(y_test_img, y_pred_img, target_names=['Benign', 'Malignant']))
```

---

## Part 3: Edge Case Roundup (Read This Before Your Meeting)

| # | Edge Case | What Happens If You Ignore It |
|---|-----------|-------------------------------|
| 1 | **Class imbalance** (HAM10000) | ANN predicts "nv" for everything → 67% accuracy, 0% recall on melanoma |
| 2 | **Missing ages** | sklearn crashes or silently drops rows |
| 3 | **No stratification** on split | Rare classes randomly missing from test set → can't evaluate |
| 4 | **Unscaled features** (ANN) | Training diverges or converges very slowly |
| 5 | **No EarlyStopping** | Model trains past optimal point → overfits |
| 6 | **No pixel normalization** (CNN) | CNN can't converge (exploding gradients) |
| 7 | **Image resizing** too large | Training takes hours on CPU |
| 8 | **Tiny dataset** (DDI = 656 images) | CNN overfits immediately → memorize training set |
| 9 | **Bad augmentations** | Unrealistic images confuse the model |
| 10 | **Wrong loss function** | Model trains but outputs nonsense |
| 11 | **Too-small batch size** | Slow, noisy training |
| 12 | **No LR scheduling** | Training plateaus early |

---

## Part 4: Discussion Questions for Your Meeting

1. **Fairness**: How does recall differ between skin_tone groups (12 vs 34 vs 56)?
2. **ANN vs CNN**: The ANN uses metadata but no images. The CNN uses images but no metadata. Which does better? Why?
3. **Data augmentation**: Did it help the CNN? How would you know? (Compare with/without.)
4. **Failure modes**: Look at misclassified images. Are there patterns? (Dark skin? Unusual angles?)
5. **Next step**: Would combining metadata + images (multi-modal model) beat both?

---

## Quick Reference: ANN vs CNN

```
ANN:
  Input: feature vector (age, sex, location dummies)
  Layers: Dense → Dense → Softmax
  Parameters: ~10,000 – 100,000
  Data needed: ~1,000+ rows
  Best for: structured/tabular data

CNN:
  Input: image tensor (height × width × channels)
  Layers: Conv2D → Pool → Conv2D → Pool → Flatten → Dense → Sigmoid
  Parameters: ~100,000 – 10,000,000+
  Data needed: ~1,000+ images per class (or augmentation)
  Best for: images, spatial data
```
