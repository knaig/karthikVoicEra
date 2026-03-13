from ragged_parler_utils import *
from ragged_parler_tts import ParlerTTSModelRunner

device = "cuda:3" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

prompts = ["अरे, तुम आज कैसे हो?",
            "आपका नाम क्या है? आपका नाम क्या है?",
            "मेरा नाम विद्या है!"]
descriptions = ["Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
                "male voice's voice is very loud yet slightly slow in delivery, with a very close recording.",
                "Female voice"]


runner = ParlerTTSModelRunner(model, tokenizer, description_tokenizer)
model_state = runner.model_prefill(prompts[:2]*5, descriptions[:2]*5)
is_all_sequences_ended = runner.is_all_sequences_ended(model_state)
model_state_enter = runner.model_prefill(prompts[2:3], descriptions[2:3])
import time
idx = 0
time_taken = []
with torch.no_grad():
    while not is_all_sequences_ended:
        start_time = time.time()
        model_state = runner.model_step(model_state)
        end_time = time.time()
        time_taken.append((end_time - start_time) * 1000)  # time in milliseconds

        if idx == 50:
            model_state = runner.merge_model_states(model_state, model_state_enter )
        is_all_sequences_ended, model_state = runner.post_model_step_updates(model_state)
        idx = idx+1


for i in range(len(runner.evicted_streamers)):
    runner.evicted_streamers[i].save_chunks_to_file('files/output_'+str(i)+'.wav')