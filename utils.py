import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMG_SIZE = (224, 224)

CLASSES = [
    "positive_IDC",
    "negative_IDC",
    "Unknown"
]


def preprocess_image(image):

    """
    Same preprocessing used during training.
    """

    img = np.array(image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        IMG_SIZE
    )

    img = img.astype(
        np.float32
    )

    img = preprocess_input(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    return img



def make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name="Conv_1"
):

    """
    Generate Grad-CAM heatmap.
    """

    if last_conv_layer_name is None:
        last_conv_layer_name = ""

    if last_conv_layer_name in {layer.name for layer in model.layers}:
        target_layer = model.get_layer(last_conv_layer_name)
    else:
        target_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break

        if target_layer is None:
            raise ValueError(
                "No convolutional layer was found in the model."
            )


    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array,
            training=False
        )


        predicted_class = tf.argmax(
            predictions[0]
        )


        loss = predictions[:, predicted_class]


    grads = tape.gradient(
        loss,
        conv_outputs
    )

    if grads is None:
        raise ValueError(
            "Unable to compute gradients for Grad-CAM."
        )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]


    heatmap = tf.squeeze(
        heatmap
    )


    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )


    return heatmap.numpy()



def overlay_heatmap(
        image,
        heatmap,
        alpha=0.4
):

    """
    Overlay Grad-CAM heatmap on image.
    """


    heatmap = cv2.resize(
        heatmap,
        (
            image.shape[1],
            image.shape[0]
        )
    )


    heatmap = np.uint8(
        255 * heatmap
    )


    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    overlay = cv2.addWeighted(
        image,
        1-alpha,
        heatmap,
        alpha,
        0
    )


    return overlay