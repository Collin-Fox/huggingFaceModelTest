import numpy as np
import summarizationT5 as model2
import torch

'''
get_encoding(promt to translate to german STRING) - returns encoder outputs as numpy array, inputs as tokens
get_decode(inputs, encoder_outputs)

'''

prompts = ("In the first time step t1, the previous hidden state h0 is considered zero or randomly selected. So the "
           "first RNN cell updates the current hidden state with the first input and h0. Each level outputs two "
           "things - the updated hidden state and the output for each level. The outputs at each level are rejected "
           "and only the hidden states are passed to the next level.")

prompt2 = ("At the first timestep t1, the previous hidden state h0 will be considered as zero or randomly chosen. So "
           "the first RNN cell will update the current hidden state with the first input and h0. Each layer outputs "
           "two things — updated hidden state and the output for each stage. The outputs at each stage are rejected "
           "and only the hidden states will be propagated to the next layer.")


def get_cos_distance(last_state1, other_last_state):
    print("")
    dot = torch.FloatTensor.dot(last_state1, other_last_state)
    norm_product = torch.FloatTensor.norm(last_state1) * torch.FloatTensor.norm(other_last_state)
    computed = 1.0 - (dot / norm_product)
    return computed


last_state, attentions, summary = model2.get_all_attributes(prompts)

print("PROMPT1: \n")
print(last_state)
print("\n")
print(summary)

last_state2, attentions2, summary2 = model2.get_all_attributes(prompt2)
print("\nPROMPT2: \n")
print(last_state2)
print("\n")
print(summary2)

distance = get_cos_distance(last_state, last_state2)
print("\nCOS DISTANCE: \n")
print(distance)
