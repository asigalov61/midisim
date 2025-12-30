# midisim
## Calculate, search, and analyze MIDI-to-MIDI similarity at scale

<img width="1536" height="1024" alt="midisim" src="https://github.com/user-attachments/assets/0b379b3a-ec9f-42c7-ba09-6b7cce87a338" />

***

## Main features

* Ultra-fast and flexible GPU/CPU MIDI-to-MIDI similarity calculation, search and analysis
* Quality pre-trained models and comprehensive pre-computed embeddings sets
* Stand-alone, versatile, and extensive codebase for general or custom MIDI-to-MIDI similarity tasks
* Full cross-platform compatibility and support

***

## Pre-trained models

* ```midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth``` - Very fast and accurate small model, suitable for all tasks. This model is included in PyPI package or it can be downloaded from Hugging Face
* ```midisim_large_pre_trained_model_2_epochs_86275_steps_0.2054_loss_0.9385_acc.pth``` - Fast large model for more nuanced embeddings generation. Download checkpoint from Hugging Face

#### Both pre-trained models were trained on full [Godzilla Piano](https://huggingface.co/datasets/asigalov61/Godzilla-Piano) dataset for 2 complete epochs

***

## Pre-computed embeddings sets

### For small pre-trained model

```discover_midi_dataset_37292_genres_midis_embeddings_cc_by_nc_sa.npy``` - 37292 genre MIDIs embeddings for genre (artist and song) identification tasks

```discover_midi_dataset_202400_identified_midis_embeddings_cc_by_nc_sa.npy``` - 202400 identified MIDIs embeddings for MIDI identification tasks

```discover_midi_dataset_3480123_clean_midis_embeddings_cc_by_nc_sa.npy``` - 3480123 select clean MIDIs embeddings for large scale similarity search and analysis tasks

### For large pre-trained model

```discover_midi_dataset_37303_genres_midis_embeddings_large_cc_by_nc_sa.npy``` - 37303 genre MIDIs embeddings for genre (artist and song) identification tasks

```discover_midi_dataset_202400_identified_midis_embeddings_large_cc_by_nc_sa.npy``` - 202400 identified MIDIs embeddings for MIDI identification tasks

```discover_midi_dataset_3480123_clean_midis_embeddings_large_cc_by_nc_sa.npy``` - 3480123 select clean MIDIs embeddings for large scale similarity search and analysis tasks

#### Pre-computed embeddings MIDI source dataset: [Discover MIDI Dataset](https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset)

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
# ================================================================================================
# Initalize midisim
# ================================================================================================

# Import main midisim module
import midisim

# ================================================================================================
# Prepare midisim embeddings
# ================================================================================================

# Option 1: Download sample pre-computed embeddings corpus from Hugging Face
emb_path = midisim.download_embeddings()

# Option 2: use custom pre-computed embeddings corpus
# See custom embeddings generation section of this README for details
# emb_path = './custom_midis_embeddings_corpus.npy'

# Load downloaded embeddings corpus
corpus_midi_names, corpus_emb = midisim.load_embeddings(emb_path)

# ================================================================================================
# Prepare midisim model
# ================================================================================================

# Option 1: Download main pre-trained midisim model from Hugging Face
model_path = midisim.download_model()

# Option 2: Use main pre-trained midisim model included in midisim PyPI package
# model_path = get_package_models()[0]['path']

# Load midisim model
model, ctx, dtype = midisim.load_model(model_path)

# ================================================================================================
# Prepare source MIDI
# ================================================================================================

# Load source MIDI
input_toks_seqs = midisim.midi_to_tokens('Come To My Window.mid')

# ================================================================================================
# Calculate and analyze embeddings
# ================================================================================================

# Compute source/query embeddings
query_emb = midisim.get_embeddings_bf16(model, input_toks_seqs)

# Calculate cosine similarity between source/query MIDI embeddings and embeddings corpus
idxs, sims = midisim.cosine_similarity_topk(query_emb, corpus_emb)

# ================================================================================================
# Processs, print and save results
# ================================================================================================

# Convert the results to sorted list with transpose values
idxs_sims_tvs_list = midisim.idxs_sims_to_sorted_list(idxs, sims)

# Print corpus matches (and optionally) convert the final result to a handy list for further processing
corpus_matches_list  midisim.print_sorted_idxs_sims_list(idxs_sims_tvs_list, corpus_midi_names, return_as_list=True)

# ================================================================================================
# Copy matched MIDIs from the MIDI corpus for listening and further evaluation and analysis
# ================================================================================================

# Copy matched corpus MIDI to a desired directory for easy evaluation and analysis
out_dir_path = midisim.copy_corpus_files(corpus_matches_list)

# ================================================================================================
```

### Raw/custom use example

#### Small model (2 epochs)

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

model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))

model.to(DEVICE)
model.eval()

# Original training autoxast setup
autocast_ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE)
```

#### Large model (2 epochs)

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
MODEL_CKPT = 'midisim_large_pre_trained_model_2_epochs_86275_steps_0.2054_loss_0.9385_acc.pth'

# Model architecture using x-transformers
model = TransformerWrapper(
    num_tokens = VOCAB_SIZE,
    max_seq_len = SEQ_LEN,
    attn_layers = Encoder(
        dim   = 512,
        depth = 16,
        heads = 8,
        rotary_pos_emb = True,
        attn_flash = True,
    ),
)

model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))

model.to(DEVICE)
model.eval()

# Original training autoxast setup
autocast_ctx = torch.amp.autocast(device_type=DEVICE, dtype=DTYPE)
```

***

## Creating custom MIDI corpus embeddings

```python
# ================================================================================================

# Load main midisim module
import midisim

# Import helper modules
import os
import tqdm

# ================================================================================================

# Call included TMIDIX module through midisim to create MIDI files list
custom_midi_corpus_file_names = midisim.TMIDIX.create_files_list(['./custom_midi_corpus_dir/'])

# ================================================================================================

# Create two lists: one with MIDI corpus file names 
# and another with MIDI corpus tokens representations suitable for embeddings generation
midi_corpus_file_names = []
midi_corpus_tokens = []

for midi_file in tqdm.tqdm(custom_midi_corpus_file_names):
    midi_corpus_file_names.append(os.path.splitext(os.path.basename(midi_file))[0])
    
    midi_tokens = midisim.midi_to_tokens(midi_file, transpose_factor=0, verbose=False)[0]
    midi_corpus_tokens.append(midi_tokens)

# It is highly recommended to sort the resulting corpus by tokens sequence length
# This greatly speeds up embeddings calculations
sorted_midi_corpus = sorted(zip(midi_corpus_file_names, midi_corpus_tokens), key=lambda x: len(x[1]))
midi_corpus_file_names, midi_corpus_tokens = map(list, zip(*sorted_midi_corpus))

# ================================================================================================
# Now you are ready to generate embeddings as follows:
# ================================================================================================

# Load main midisim model
model, ctx, dtype = midisim.load_model(verbose=False)

# Generate MIDI corpus embeddings
midi_corpus_embeddings = midisim.get_embeddings_bf16(model, midi_corpus_tokens)

# ================================================================================================

# Save generated MIDI corpus embeddings and MIDI corpus file names in one handy NumPy file
midisim.save_embeddings(midi_corpus_file_names,
                        midi_corpus_embeddings,
                        verbose=False
                       )

# ================================================================================================

# You now can use this saved custom MIDI corpus NumPy file with midisim.load_embeddings()
# and the rest of the pipeline outlined in the general use section above
```

***

## Main functions reference list

- ```midisim.midisim.copy_corpus_files``` — *Copy or synchronize MIDI corpus files from a source directory to a target corpus location.*  
- ```midisim.midisim.cosine_similarity_topk``` — *Compute cosine similarities between a query embedding and a set of embeddings and return the top‑K matches.*  
- ```midisim.midisim.download_all_embeddings``` — *Download an entire embeddings dataset snapshot from a Hugging Face dataset repository to a local directory.*  
- ```midisim.midisim.download_embeddings``` — *Download a single precomputed embeddings `.npy` file from a Hugging Face dataset repository.*  
- ```midisim.midisim.download_model``` — *Download a pre-trained model checkpoint file from a Hugging Face model repository to a local directory.*  
- ```midisim.midisim.get_embeddings_bf16``` — *Load or convert embeddings into bfloat16 format for memory-efficient inference on supported hardware.*  
- ```midisim.midisim.idxs_sims_to_sorted_list``` — *Convert parallel index and similarity arrays into a single sorted list of (index, similarity) pairs ordered by similarity.*  
- ```midisim.midisim.load_embeddings``` — *Load a saved NumPy embeddings file and return the arrays of MIDI names and corresponding embedding vectors.*  
- ```midisim.midisim.load_model``` — *Construct a Transformer model, load weights from a checkpoint, move it to the requested device, and return the model with an AMP autocast context and dtype.*  
- ```midisim.midisim.masked_mean_pool``` — *Compute a masked mean pooling over sequence embeddings, ignoring padded positions via a boolean or numeric mask.*  
- ```midisim.midisim.midi_to_tokens``` — *Convert a single-track MIDI file into one or more compact integer token sequences (with optional transpositions) suitable for model input.*  
- ```midisim.midisim.pad_and_mask``` — *Pad a batch of variable-length token sequences to a common length and produce an attention/mask tensor indicating real tokens vs padding.*  
- ```midisim.midisim.print_sorted_idxs_sims_list``` — *Pretty-print a sorted list of (index, similarity) pairs, optionally annotating entries with filenames or metadata.*  
- ```midisim.midisim.save_embeddings``` — *Save a list of name strings and their corresponding embedding vectors into a structured NumPy array and optionally persist it to disk.*

***

## Limitations

* Current code and models support only MIDI music elements similarity (start-times, durations and pitches)
* MIDI channels, instruments, velocities and drums similarites are not currently supported due to complexity and practicality considerations
* Current pre-trained models are limited by 3k sequence length (~1000 MIDI music notes) so long running MIDIs can only be analyzed in chunks
* Solo drum track MIDIs are not currently supported and can't be analyzed

***

## Citations

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { midisim (Revision 707e311) },
	year         = 2025,
	url          = { https://huggingface.co/projectlosangeles/midisim },
	doi          = { 10.57967/hf/7383 },
	publisher    = { Hugging Face }
}
```

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { midisim-embeddings (Revision 8ebb453) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/projectlosangeles/midisim-embeddings },
	doi          = { 10.57967/hf/7382 },
	publisher    = { Hugging Face }
}
```

```bibtex
@misc{project_los_angeles_2025,
	author       = { Project Los Angeles },
	title        = { Discover-MIDI-Dataset (Revision 0eaecb5) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset },
	doi          = { 10.57967/hf/7361 },
	publisher    = { Hugging Face }
}
```

***

### Project Los Angeles
### Tegridy Code 2025
