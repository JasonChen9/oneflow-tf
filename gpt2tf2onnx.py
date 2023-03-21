from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import tf2onnx
import torch
from onnx2torch import convert
import onnxruntime as rt
text = "a dog"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id
)

input = tokenizer.encode(text, return_tensors = 'tf')
tf.random.set_seed(0)

sample_output = model.generate(
    input, 
    do_sample=True, 
    max_length=97, 
    top_p=0.78, 
    top_k=0)

print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


input_ids = tf.ones(
    [1, 5], dtype=tf.int64
)


model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(model)



# model.save_pretrained("./tfgpt2model", saved_model=True)


# torch_model = convert("./tf2flow_gpt2_model.onnx")

# input = tokenizer.encode(text, return_tensors = 'pt')
# shape = input.shape

# output = torch_model(
#     input,
#     torch.ones(shape, dtype=torch.int32)
# )
# print(output)

# input = tokenizer.encode(text, return_tensors = 'tf')
# print(input.shape)
# tf.random.set_seed(0)

# sample_output = model.generate(
#     input, 
#     do_sample=True, 
#     max_length=97, 
#     top_p=0.78, 
#     top_k=0)

# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
# input_ids = tf.ones(
#     [1, 5], dtype=tf.int64
# )
# model_proto, _ = tf2onnx.convert.from_function (
#     model, input_signature=input_ids ,opset=11, output_path="model/tf2flow_gpt2.onnx")
# output_names = [n.name for n in model_proto.graph.output]

# providers = ['CPUExecutionProvider']
# m = rt.InferenceSession(onnx_model_path, providers=providers)
# onnx_pred = m.run(output_names, {"input": x})

# print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])