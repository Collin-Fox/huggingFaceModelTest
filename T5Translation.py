import numpy
from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import jax.numpy as jnp
import numpy as np
import torch
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_nf4 = AutoModelForCausalLM.from_pretrained("t5-large", output_attentions='True', output_hidden_states='True', quantization_config=nf4_config)

#translate = "Translate from English to German: "
translate = "Summarize: "
tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = FlaxT5ForConditionalGeneration.from_pretrained("t5-large", output_attentions='True', output_hidden_states='True')

model.generation_config.max_new_tokens = 500
model.generation_config.output_attentions
model.generation_config.output_hidden_states


def get_encoding(prompt):
    final_prompt = translate + prompt
    inputs = tokenizer(final_prompt, return_tensors="np")
    encoder_outputs = model.encode(**inputs)
    return encoder_outputs, inputs


def get_decode(inputs, encoder_outputs):
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
    outputs = model.decode(decoder_input_ids, encoder_outputs)
    summary_ids = model.generate(inputs["input_ids"]).sequences

    return print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False,
                                  max_new_tokens=45))


def get_logits(outputs):
    logits = outputs.logits


def get_summary(inputs):
    summary_ids = model.generate(inputs["input_ids"]).sequences


def get_vector(encoded):
    return np.array(encoded[0])


def get_attention_vector(encoded):
    return np.array(encoded[1])

def get_test(encoded):
    return np.array(encoded[2])


def get_norm(npArray):
    return np.linalg.norm(npArray)


'''



tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = FlaxT5ForConditionalGeneration.from_pretrained("t5-large")
text = "translate from English to German: Ill see you in five minutes"
inputs = tokenizer(text, return_tensors="np")
encoder_outputs = model.encode(**inputs)
arr = np.array(encoder_outputs[0])
print("ENCODER: \n")
print(np.linalg.norm(arr))
print(arr.shape)

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

outputs = model.decode(decoder_input_ids, encoder_outputs)
print(outputs)

summary_ids = model.generate(inputs["input_ids"]).sequences
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False, max_new_tokens=45))
logits = outputs.logits
print(logits)

'''
