import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.xception_cbam import build_xception_cbam
from training.utils import check_dataset

check_dataset()

IMAGE_SIZE = (256,256)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    "dataset/real_vs_fake/real-vs-fake/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

valid = datagen.flow_from_directory(
    "dataset/real_vs_fake/real-vs-fake/valid",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = build_xception_cbam()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True)
]

model.fit(train, validation_data=valid, epochs=10, callbacks=callbacks)
