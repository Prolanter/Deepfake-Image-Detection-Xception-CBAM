from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

def build_classifier(x):
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    return output
