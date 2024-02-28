import numpy as np
import summarizationT5 as model2
import torch
import threading
import concurrent.futures
from sklearn.metrics.pairwise import cosine_distances

'''
get_encoding(promt to translate to german STRING) - returns encoder outputs as numpy array, inputs as tokens
get_decode(inputs, encoder_outputs)

'''

prompts = ("In the first time step t1, the previous hidden state h0 is considered zero or randomly selected. So the "
           "first RNN cell updates the current hidden state with the first input and h0. Each level outputs two "
           "things - the updated hidden state and the output for each level. The outputs at each level are rejected "
           "and only the hidden states are passed to the next level.")

prompt3 = ("At the first timestep t1, the previous hidden state h0 will be considered as zero or randomly chosen. So "
           "the first RNN cell will update the current hidden state with the first input and h0. Each layer outputs "
           "two things — updated hidden state and the output for each stage. The outputs at each stage are rejected "
           "and only the hidden states will be propagated to the next layer.")


prompt2 = ("Before long the cat was seized by another fit of longing. She said to the mouse, You must do me a favour, "
           "and once more manage the house for a day alone. I am again asked to be godmother, and, as the child has a "
           "white ring round its neck, I cannot refuse. The good mouse consented, but the cat crept behind the town "
           "walls to the church, and devoured half the pot of fat. Nothing ever seems so good as what one keeps to "
           "oneself, said she, and was quite satisfied with her day's work. When she went home the mouse inquired, "
           "And what was this child christened?" "Half-done, answered the cat. Half-done! What are you saying? I "
           "never heard the name in my life, I'll wager anything it is not in the calendar!")


def get_cos_distance(last_state1, other_last_state):
    dot = torch.FloatTensor.dot(last_state1, other_last_state)
    norm_product = torch.FloatTensor.norm(last_state1) * torch.FloatTensor.norm(other_last_state)
    computed = 1.0 - (dot / norm_product)
    return computed


def get_prompt_inp(prompt):
    last_state, attentions, summary, inp = model2.get_all_attributes(prompt)

    print("PROMPT1: \n")
    print("Embedded Vector: \n")
    print(inp)
    print(last_state)
    print("\n")
    print(summary)

