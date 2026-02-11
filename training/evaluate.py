import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("best_model.keras")

datagen = ImageDataGenerator(rescale=1./255)

test = datagen.flow_from_directory(
    "dataset/real_vs_fake/real-vs-fake/test",
    target_size=(256,256),
    class_mode='binary',
    shuffle=False
)

pred = model.predict(test)
pred_labels = (pred > 0.5).astype(int)

print(classification_report(test.classes, pred_labels))
