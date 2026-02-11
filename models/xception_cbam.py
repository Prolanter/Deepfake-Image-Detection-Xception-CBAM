import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from .attention import cbam_block
from .classifier_head import build_classifier

def build_xception_cbam(input_shape=(256,256,3)):
    base_model = Xception(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = cbam_block(x)
    output = build_classifier(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
