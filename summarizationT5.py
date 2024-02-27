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

    return last_state[0][0], encoder_attentions, summarization

"""

one_piece_sequence = (
    "Kyle Shanahan's 49ers come up shy once again. Once again, Shanahan watched Mahomes eat up a 10-point deficit in "
    "the Super Bowl. Credit the coach for not puttering out, going for it on a big fourth-and-3 early in the fourth "
    "quarter when a chip-shot field goal would have tied the game. The decision underscored that Shanny knew they "
    "needed touchdowns, not field goals, against Mahomes. The coach pulled out the stops, including a big trick play "
    "TD pass from receiver Jauan Jennings to Christian McCaffrey in the first half. Alas, an OT FG from inside the 10 "
    "opened the door for Mahomes' heroics. The Niners will beat themselves up all offseason for not putting the game "
    "away early. San Francisco gashed the Chiefs D in the first quarter, including gains of 18 yards, 11 yards, "
    "and 11 yards on the opening possession, but a McCaffrey fumble (just his third loss all season) snuffed out the "
    "drive. The Niners led by seven at halftime despite dominating possession early. Shanahan's offense struggled to "
    "open the third quarter, going three-and-out on three consecutive possessions. A muffed punt in the third quarter "
    "set up the Chiefs' first TD and lead. Instead of a big lead, the Niners were in scramble mode. A blocked PAT "
    "also proved massive late. Credit Brock Purdy for making plays in the fourth quarter after a roller-coaster game "
    "that saw him miss a few throws and make some other excellent reads. The 49ers' last three possessions went TD, "
    "FG and FG. Unfortunately for Shanahan, his D couldn't stall Mahomes, allowing the Chiefs to go TD, FG, FG, "
    "TD. The NFL has seen two overtime Super Bowl games, and Shanahan has been on the losing end of both. Brutal.")


#other_sequence = (
 #   "K.C. wouldn't have been the first back-to-back Super Bowl champions since the 2003-2004 New England Patriots "
  #  "without Steve Spagnuolo's smothering defense. The Chiefs gave up some chunk plays, particularly in the first "
   # "half, but bowed up often. The lack of edge pressure on Purdy was noticeable, but Spagnuolo adjusted and made "
   ## "life more difficult on the QB in the third quarter. Spags brought the blitz on 51.2% of Purdy's dropbacks. The "
    #"signal-caller made some plays versus pressure, but speeding him up made a difference in the second half. Spags' "
    #"crew stuffed McCaffrey on the ground for the bulk of the game, holding the RB to 3.6 yards per carry on 22 "
    #"totes. Taking out those gashing runs put the onus on Purdy to carry the contest. The Chiefs were phenomenal on "
    #"third downs, allowing the Niners to convert just 3 of 12 in the contest. K.C. corner Trent McDuffie was the "
    ##"defensive MVP, making numerous big plays, including a TD-saving swat on Deebo Samuel early in the game. Nick "
    #"Bolton was all over the field, gobbling up 13 tackles, one TFL and two QB hits. And Chris Jones generated a "
    #"team-high six QB pressures, forcing several errant Purdy passes that could have gone for big gains. K.C.'s D "
    #"generated a season-high nine unblocked pressures in Super Bowl LVIII, all of which came on blitzes, per Next Gen "
    #"Stats. That is amazing scheming by Spags. "
#)
other_sequence = (
    "This morning I ran twelve miles. Then I brushed my teeth fast. I then took the bus to school, it took thirty minutes."
)

inputs = tokenizer.encode("summarize: " + one_piece_sequence,
                          return_tensors='pt',
                          max_length=512,
                          truncation=True)

print(inputs)
summarization_ids = model.generate(inputs, max_length=80, min_length=40, length_penalty=5., num_beams=2)

data = model(decoder_input_ids=summarization_ids, input_ids=inputs, output_hidden_states=True, output_attentions=True)
transformers.modeling_outputs.Seq2SeqLMOutput.encoder_last_hidden_state
last_state = data.encoder_last_hidden_state
encoder_attentions = data.encoder_attentions

#print(attentions)
print("LAST HIDDEN STATAE: \n")
print(last_state[0][0])
print(torch.FloatTensor.norm(last_state[0][0]))

summarization = tokenizer.decode(summarization_ids[0])

print(summarization)



otherinputs = tokenizer.encode("summarize: " + other_sequence,
                          return_tensors='pt',
                          max_length=512,
                          truncation=True)

print(otherinputs)
othersummarization_ids = model.generate(otherinputs, max_length=80, min_length=40, length_penalty=5., num_beams=2)

otherdata = model(decoder_input_ids=othersummarization_ids, input_ids=otherinputs, output_hidden_states=True, output_attentions=True)
transformers.modeling_outputs.Seq2SeqLMOutput.encoder_last_hidden_state
otherlast_state = otherdata.encoder_last_hidden_state
otherencoder_attentions = otherdata.encoder_attentions

#print(attentions)
print("LAST HIDDEN STATAE: \n")
print(otherlast_state[0][0])
print(torch.FloatTensor.norm(otherlast_state[0][0]))

othersummarization = tokenizer.decode(othersummarization_ids[0])

print(othersummarization)

dot = torch.FloatTensor.dot(last_state[0][0], otherlast_state[0][0])
print("DOT PRODUCT: \n")
print(dot)

norms = torch.FloatTensor.norm(last_state[0][0]) * torch.FloatTensor.norm(otherlast_state[0][0])
print("\nNORMS: \n")
print(norms)

compute = 1.0 - (dot / norms)
print("\nCOMPUTE: \n")
print(compute)
"""