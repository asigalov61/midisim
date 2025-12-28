# midisim
## Calculate, search, and analyze MIDI-to-MIDI similarity at scale

<img width="1536" height="1024" alt="midisim" src="https://github.com/user-attachments/assets/0b379b3a-ec9f-42c7-ba09-6b7cce87a338" />

***

## Installation

### midisim PyPI package (for general use)

```sh
!pip install -U midisim
```

### x-transformers 2.3.1 (for raw/custom tasks)

```sh
!pip install x-transformers==2.3.1
```

***

## Basic use guide

### General use example

```python
# Import main midisim module
import midisim

# Download sample pre-computed embeddings corpus
emb_path = midisim.download_embeddings()

# Load downloaded embeddings corpus
corpus_midi_names, corpus_emb = midisim.load_embeddings()

# Download main pre-trained midisim model
model_path = midisim.download_model()

# Load midisim model
model, ctx, dtype = midisim.load_model(model_path)

# Load source MIDI
input_toks_seqs = midisim.midi_to_tokens('Come To My Window.mid')

# Compute source/query embeddings
query_emb = midisim.get_embeddings_bf16(model, input_toks_seqs)

# Calculate cosine similarity between source/query MIDI embeddings and embeddings corpus
idxs, sims = midisim.cosine_similarity_topk(query_emb, corpus_emb)

# Convert the results to sorted list with transpose values
idxs_sims_tvs_list = midisim.idxs_sims_to_sorted_list(idxs, sims)

# Print corpus matches (and optionally) convert the final result to a handy dict
midisim.print_sorted_idxs_sims_list(idxs_sims_tvs_list, corpus_midi_names, return_as_list=True)
```

### Raw/custom use example

```python
import torch
from x_transformers import TransformerWrapper, Encoder

# Original model hyperparameters
SEQ_LEN = 3072

MASK_IDX     = 384 # Use this value for masked modelling
PAD_IDX      = 385 # Model pad index
VOCAB_SIZE   = 386 # Total vocab size

MASK_PROB    = 0.15 # Original training mask probability value (use for masked modelling)

DEVICE = 'cuda' # You can use any compatible device or CPU
DTYPE  = torch.bfloat16 # Original training dtype

# Official main midisim model checkpoint name
MODEL_CKPT = 'midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth'

# Model architecture using x-transformers
model = TransformerWrapper(
    num_tokens = VOCAB_SIZE,
    max_seq_len = SEQ_LEN,
    attn_layers = Encoder(
        dim   = 512,
        depth = 8,
        heads = 8,
        rotary_pos_emb = True,
        attn_flash = True,
    ),
)

model.load_state_dict(torch.load(MODEL_CKPT))

model.to(DEVICE)
model.eval()

# Original training autoxast setup
autocast_ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE)
```

***

## Creating custom MIDI corpus embeddings

```python
# Load main midisim module
import midisim

# Import helper modules
import os
import tqdm

# Call included TMIDIX module through midisim to create MIDI files list
custom_midi_corpus_file_names = midisim.TMIDIX.create_files_list(['./custom_midi_corpus_dir/'])

# Create two lists: one with MIDI corpus file names 
# and another with MIDI corpus tokens representations suitable for embeddings generation
midi_corpus_file_names = []
midi_corpus_tokens = []

for midi_file in tqdm.tqdm(custom_midi_corpus_file_names):
    midi_corpus_file_names.append(os.path.splitext(os.path.basename(midi_file))[0])
    
    midi_tokens = midisim.midi_to_tokens(midi_file, transpose_factor=0, verbose=False)[0]
    midi_corpus_tokens.append(midi_tokens)

# Load main midisim model
model, ctx, dtype = midisim.load_model(verbose=False)

# Generate MIDI corpus embeddings
midi_corpus_embeddings = midisim.get_embeddings_bf16(model, midi_corpus_tokens)

# Save generated MIDI corpus embeddings and MIDI corpus file names in one handy NumPy file
midisim.save_embeddings(midi_corpus_file_names,
                        midi_corpus_embeddings,
                        verbose=False
                       )

# You can now use this saved custom MIDI corpus file with midisim.load_embeddings
# and the rest of the pipeline outlined in basic use section.
```

***

### Project Los Angeles
### Tegridy Code 2025
