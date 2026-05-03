import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import pandas as pd

#PATH
train_dir = "fruits-360/Training"
test_dir  = "fruits-360/Test"

#LOAD DATA
IMG_SIZE = (100, 100)
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Số lớp:", num_classes)

#OPTIMIZED PIPELINE
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

#DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

#MODEL
model = models.Sequential([
    layers.Input(shape=(100, 100, 3)),

    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.35),

    layers.Dense(num_classes, activation='softmax')
])

#COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

#
#CALLBACKS
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor='val_loss',
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )
]

#TRAIN
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=25,
    callbacks=callbacks,
    verbose=1
)

#SAVE HISTORY TO EXCEL
hist_df = pd.DataFrame(history.history)
hist_df.to_excel("training_log.xlsx", index=False)
print("Đã lưu file training_log.xlsx")

#SAVE MODEL
model.save("fruits_model.h5")
np.save("class_names.npy", class_names)
print("Đã lưu model!")

#EVALUATE
loss, acc = model.evaluate(test_ds)
print("Accuracy:", acc)
