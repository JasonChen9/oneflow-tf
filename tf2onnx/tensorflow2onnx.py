import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tf2onnx
import onnxruntime as rt


if __name__ == '__main__':
    img_path = '../img/cat.jpg'
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = ResNet50(weights='imagenet')

    preds = model.predict(x)
    print('Keras Predicted:', decode_predictions(preds, top=3)[0])

    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    output_path = model.name + ".onnx"

    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]

    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(output_path, providers=providers)
    onnx_pred = m.run(output_names, {"input": x})

    print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])

    # make sure ONNX and keras have the same results
    np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-3)
