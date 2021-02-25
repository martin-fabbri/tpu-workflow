from tensorflow.keras import Sequential
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Lambda


def create_xception_ft_model(image_size, num_classes):
    img_adjust_layer = Lambda(preprocess_input, input_shape=[image_size, image_size, 3])
    pretrained_model = Xception(include_top=False)
    pretrained_model.trainable = True
    # fmt: off
    model = Sequential([
        img_adjust_layer, 
        pretrained_model, 
        GlobalAveragePooling2D(), 
        Dense(num_classes, activation="softmax")
    ])
    # fmt: on
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
