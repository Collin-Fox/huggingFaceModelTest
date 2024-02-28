import numpy as np
import torch
import transformers.modeling_outputs
from transformers import T5Tokenizer, T5ForConditionalGeneration

# set up tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large', output_hidden_states=True)
def get_all_attributes(prompt):
    #Getting the prompt
    one_piece_sequence = (prompt)
    inputs = tokenizer.encode("summarize: " + one_piece_sequence,
                              return_tensors='pt',
                              max_length=512,
                              truncation=True)

    summarization_ids = model.generate(inputs, max_length=80, min_length=40, length_penalty=5., num_beams=2)

    data = model(decoder_input_ids=summarization_ids, input_ids=inputs, output_hidden_states=True,
                 output_attentions=True)

    last_state = data.encoder_last_hidden_state
    encoder_attentions = data.encoder_attentions

    summarization = tokenizer.decode(summarization_ids[0])

    return last_state[0][0], encoder_attentions, summarization, inputs
