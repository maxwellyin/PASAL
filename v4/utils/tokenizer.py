# %%
from transformers import T5Tokenizer, T5Config

model_checkpoint = "google/flan-t5-base"

special_token = '<hl>'

# Load the T5 tokenizer and config
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
config = T5Config.from_pretrained(model_checkpoint)

# Add special token to the tokenizer's vocabulary
tokenizer.add_tokens([special_token])

# Resize the model's embeddings to account for the new token
config.vocab_size = len(tokenizer)
tokenizer.model_max_length = tokenizer.max_len_single_sentence

# Save the modified tokenizer to disk
tokenizer.save_pretrained('./qg-tokenizer')

# %%
