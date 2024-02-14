import T5Translation as model
import numpy as np
'''
get_encoding(promt to translate to german STRING) - returns encoder outputs as numpy array, inputs as tokens
get_decode(inputs, encoder_outputs)

'''

f = open("venv/myfile.txt", "a")

prompts = ["Today is wednesday"]
norms = []
vectors = []
x = 0
for prompt in prompts:
    encoderOUT, inputs = model.get_encoding(prompt)
    prompt1Vec = model.get_vector(encoderOUT)
    prompt1Norm = model.get_norm(prompt1Vec)
    prompt1Attention = model.get_attention_vector(encoderOUT)
    print("ENCODER VECTOR: \n")
    print(prompt1Vec)
    print("\nENCODER NORM: \n")
    print(prompt1Norm)
    print("\nATTENTION VECTOR: \n")
    print(prompt1Attention)
    print("\nATTENTION NORM: \n")
    print(model.get_norm(prompt1Attention))
    print(prompt1Vec.shape)
    print(prompt1Attention.shape)
    vectors.append(prompt1Vec)
    norms.append(prompt1Norm)
    outPrompt = model.get_decode(inputs, encoderOUT)
    x = x + 1

#prompt1Vec.tofile("venv/myfile.txt", format='%s')
dot = np.matmul(vectors[0], vectors[1].transpose())

print(dot)