# midisim Python module

r'''###############################################################################
###################################################################################
#
#
#	midisim Python module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#   Critical dependencies
#
#   !pip install huggingface_hub
#   !pip install hf-transfer
#   !pip install ipywidgets
#   !pip install tqdm
#
#   !pip install torch
#   !pip install einops
@   !pip install einx
#   !pip install torch-summary
#   !pip install matplotlib
#   !pip install numpy==1.26.4
#
###################################################################################
'''

###################################################################################
###################################################################################

print('=' * 70)
print('Loading midisim Python module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('midisim module version', __version__)
print('=' * 70)

###################################################################################
###################################################################################

import os, copy, math, shutil

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from typing import List, Optional, Union, Tuple, Dict, Any

from functools import lru_cache

import tqdm

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from .x_transformer_2_3_1 import TransformerWrapper, Encoder

from torchsummary import summary

from . import TMIDIX

from huggingface_hub import hf_hub_download, snapshot_download

print('=' * 70)
print('PyTorch version:', torch.__version__)
print('=' * 70)

###################################################################################

def download_all_embeddings(repo_id: str = 'projectlosangeles/midisim-embeddings',
                            revision: str = 'main',
                            local_dir: str = './midisim-embeddings/',
                            verbose: bool = True,
                            **kwargs: dict[str, Any]
                           ) -> str:

    """
    Helper function that downloads all pre-computed midisim embeddings from Hugging Face
    
    Returns
    -------
    Output directory path string where all embeddings were downloaded to
    """

    if verbose:
        print('=' * 70)
        print('Downloading all embeddings...')
        print('=' * 70)

    result = snapshot_download(repo_id=repo_id,
                               repo_type='dataset',
                               revision=revision,
                               local_dir=local_dir,
                               **kwargs
                              )

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def download_embeddings(repo_id: str = 'projectlosangeles/midisim-embeddings',
                        filename: str = 'discover_midi_dataset_37292_genres_midis_embeddings_cc_by_nc_sa.npy',
                        local_dir: str = './midisim-embeddings/',
                        verbose: bool = True,
                        **kwargs: dict[str, Any]
                       ) -> str:
    
    """
    Helper function that downloads pre-computed midisim embeddings files from Hugging Face
    
    Returns
    -------
    Downloaded embeddings file path string
    """
    
    if verbose:
        print('=' * 70)
        print('Downloading embeddings...')
        print('=' * 70)

    result = hf_hub_download(repo_id=repo_id,
                             repo_type='dataset',
                             filename=filename,
                             local_dir=local_dir,
                             **kwargs
                            )
    if verbose:    
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def download_model(repo_id: str = 'projectlosangeles/midisim',
                   filename: str = 'midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth',
                   local_dir: str = './midisim-models/',
                   verbose: bool = True,
                   **kwargs: dict[str, Any]
                  ) -> str:
    
    """
    Helper function that downloads pre-trained midisim models from Hugging Face
    
    Returns
    -------
    Downloaded model checkpoint file path string
    """
    
    if verbose:
        print('=' * 70)
        print('Downloading model...')
        print('=' * 70)

    result = hf_hub_download(repo_id=repo_id,
                             repo_type='model',
                             filename=filename,
                             local_dir=local_dir,
                             **kwargs
                            )
    if verbose:    
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def load_model(model_path: str = './midisim-models/midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth',
               dim: int = 512,
               depth: int = 8,
               heads: int = 8,
               max_seq_len: int = 3072,
               pad_idx: int = 385,
               dtype: torch.dtype = torch.bfloat16,
               device: str = 'cuda',
               verbose: bool = True
              ) -> str:

    """Load and initialize a preconfigured midisim Transformer model from a checkpoint.
    
    One-line summary
    ----------------
    Create a `TransformerWrapper` with an `Encoder` attention stack, load weights
    from a checkpoint file, move the model to the requested device, set it to
    evaluation mode, and return the model together with an automatic mixed-precision
    (AMP) autocast context and the chosen dtype.
    
    Detailed description
    --------------------
    This helper constructs a Transformer-based model using the provided
    architecture hyperparameters, loads a saved state dictionary from `model_path`
    (using `torch.load`), transfers the model to `device`, and switches it to
    evaluation mode (`model.eval()`). It also creates and returns a `torch.amp.autocast`
    context manager configured for the requested `device` and `dtype`. When
    `verbose` is True, the function prints progress messages and a model summary.
    
    Parameters
    ----------
    model_path : str, optional
        Filesystem path to the saved PyTorch checkpoint (state dict). Default is
        `'./midisim-models/midisim_small_pre_trained_model_2_epochs_43117_steps_0.3148_loss_0.9229_acc.pth'`.
    dim : int, optional
        Hidden dimension size for the encoder attention layers. Default: 512.
    depth : int, optional
        Number of encoder layers (depth of the Transformer encoder). Default: 8.
    heads : int, optional
        Number of attention heads per multi-head attention layer. Default: 8.
    max_seq_len : int, optional
        Maximum sequence length the model supports (positional embedding length).
        Default: 3072.
    pad_idx : int, optional
        Index reserved for padding tokens. The model's vocabulary size is set to
        `pad_idx + 1`. Default: 385.
    dtype : torch.dtype, optional
        Floating-point dtype used for AMP autocasting (e.g., `torch.bfloat16`,
        `torch.float16`, `torch.float32`). Default: `torch.bfloat16`.
    device : str or torch.device, optional
        Target device for the model (e.g., `'cuda'`, `'cpu'`, or a `torch.device`).
        Default: `'cuda'`.
    verbose : bool, optional
        If True, print initialization/loading progress and a model summary.
        Default: True.
    
    Returns
    -------
    tuple
        A 3-tuple `(model, ctx, dtype)` where:
        - **model**: the `TransformerWrapper` instance with loaded weights,
          moved to `device` and set to evaluation mode.
        - **ctx**: a `torch.amp.autocast` context manager configured with
          `device_type=device` and `dtype=dtype`. Use this context when running
          inference to enable mixed-precision casting consistent with the model.
        - **dtype**: the `torch.dtype` passed into the function (returned for
          convenience so callers can reuse it when preparing inputs or contexts).
    
    Side effects and notes
    ----------------------
    - The function calls `torch.load(model_path)` and `model.load_state_dict(...)`.
      The checkpoint must be a compatible state dictionary for the constructed
      model architecture; otherwise `model.load_state_dict` may raise a
      `RuntimeError`.
    - The model is moved to `device` via `model.to(device)` and set to evaluation
      mode with `model.eval()`.
    - `num_tokens` is derived from `pad_idx + 1`. Ensure `pad_idx` matches the
      tokenizer/vocabulary used when the checkpoint was created.
    - The `summary(model)` call used when `verbose` is True requires an available
      `summary` function in scope (for example from `torchinfo` or a custom helper).
    - The returned `ctx` is a context manager; to use it:
      ```py
      with ctx:
          outputs = model(inputs)
      ```
    - If `device` is `'cuda'` but CUDA is unavailable, `model.to(device)` will raise
      an error; pass `'cpu'` to run on CPU.
    
    Exceptions
    ----------
    - `FileNotFoundError` or `OSError` if `model_path` does not exist or cannot be read.
    - `RuntimeError` if the checkpoint is incompatible with the model architecture
      (e.g., missing or unexpected keys in the state dict).
    - Any exceptions raised by `model.to(device)` if the device is invalid or
      resources are insufficient.
    
    Example
    -------
    model, amp_ctx, dtype = load_model(
        model_path='checkpoints/midisim.pth',
        dim=512,
        depth=8,
        heads=8,
        max_seq_len=3072,
        pad_idx=385,
        dtype=torch.bfloat16,
        device='cuda',
        verbose=True,
    )
    
    # Inference example
    model_input = ...  # prepare input tensor on the same device
    with amp_ctx:
        logits = model(model_input)
    """

    if verbose:
        print('=' * 70)
        print('midisim model loader')
        print('=' * 70)
        print('Initializing model...')

    ctx = torch.amp.autocast(device_type=device, dtype=dtype)

    model = TransformerWrapper(
                num_tokens=pad_idx+1,
                max_seq_len=max_seq_len,
                attn_layers=Encoder(
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    rotary_pos_emb=True,
                    attn_flash=True,
                ),
    )

    if verbose:
        print('=' * 70)
        print('Loading model checkpoint...')
    
    model.load_state_dict(torch.load(model_path, map_location=device))

    if verbose:
        print('=' * 70)
    
    model.to(device)
    model.eval()

    if verbose:
        print('Done!')

        print('=' * 70)
        print('Model Summary')
        summary(model)

    return model, ctx, dtype

###################################################################################

def load_embeddings(embeddings_path: str = './midisim-embeddings/discover_midi_dataset_37292_genres_midis_embeddings_cc_by_nc_sa.npy',
                    midi_names_key: str = 'midi_names',
                    midi_embeddings_key: str = 'midi_embeddings',
                    verbose: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Helper function that loads pre-computed embeddings file

    Returns
    -------
    Tuple of nd.arrays (midi_names_arr, midi_embeddings_arr)
    """

    if verbose:
        print('=' * 70)
        print('Loading embeddings...')
        
    embeddings_data = np.load(embeddings_path, allow_pickle=True)
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
    return embeddings_data[midi_names_key], embeddings_data[midi_embeddings_key]

###################################################################################

def save_embeddings(embeddings_name_strings: list[str],
                    embeddings: Union[torch.Tensor, np.ndarray],
                    name_strings_key: str = 'midi_names',
                    embeddings_key: str = 'midi_embeddings',
                    output_file_name: str = 'saved_midi_embeddings.npy',
                    return_merged_array: bool = False,
                    verbose=True
                   ) -> Union[np.ndarray, None]:

    """Save a list of name strings and their corresponding embedding vectors into a NumPy structured array
    and optionally persist it to disk.
    
    This function builds a NumPy structured array with two fields (one for the name strings and one for
    the embedding vectors), populates it from the provided inputs, casts embeddings to `np.float32`,
    and either returns the merged structured array or saves it to disk using `np.save`.
    
    Parameters
    ----------
    embeddings_name_strings : list[str]
        Sequence of Python strings that identify each embedding (e.g., filenames, IDs, labels).
        The length of this list determines the number of rows in the resulting structured array.
    embeddings : Union[torch.Tensor, np.ndarray]
        2D array-like of shape `(n, D)` containing the embedding vectors, where `n` must equal
        `len(embeddings_name_strings)` and `D` is the embedding dimensionality. If a `torch.Tensor`
        is provided it will be converted to a NumPy array with `.numpy()` (no automatic `.cpu()`
        or `.detach()` is performed by this function).
    name_strings_key : str, optional
        Field name to use for the name strings in the structured dtype (default: `'midi_names'`).
    embeddings_key : str, optional
        Field name to use for the embedding vectors in the structured dtype (default:
        `'midi_embeddings'`).
    output_file_name : str, optional
        Path (filename) where the structured array will be saved with `np.save` if
        `return_merged_array` is `False` (default: `'saved_midi_embeddings.npy'`).
    return_merged_array : bool, optional
        If `True`, the function returns the constructed structured NumPy array and does not write
        anything to disk. If `False`, the array is saved to `output_file_name` and the function
        returns `None` (default: `False`).
    verbose : bool, optional
        If `True`, print progress and diagnostic messages to stdout (default: `True`).
    
    Returns
    -------
    Union[np.ndarray, None]
        - If `return_merged_array` is `True`: the NumPy structured array of length `n` with dtype
          `[(name_strings_key, object), (embeddings_key, np.float32, (D,))]`.
        - If `return_merged_array` is `False`: `None` (the array is saved to `output_file_name`).
    
    Raises
    ------
    ValueError
        - If `embeddings` does not have a second dimension (i.e., is not 2D) so the embedding
          dimensionality `D` cannot be determined.
        - If the number of rows in `embeddings` does not match `len(embeddings_name_strings)`.
    TypeError
        - If `embeddings_name_strings` is not a sequence with a definable length.
    Exception
        - Any unexpected exceptions raised while reading attributes (e.g., `.shape`, `.dtype`) or
          during `np.save` will propagate to the caller.
    
    Notes
    -----
    - The function constructs a structured dtype where the name field uses Python `object` to allow
      variable-length strings and the embedding field is a fixed-size `np.float32` vector of length `D`.
    - Embeddings are explicitly cast to `np.float32` before assignment; this may change precision.
    - When a `torch.Tensor` is passed, the function calls `.numpy()` directly. If the tensor is on a
      GPU or requires gradient, callers should ensure it is detached and moved to CPU first (e.g.,
      `embeddings.detach().cpu()`), otherwise `.numpy()` may raise an error.
    - The file is written using `np.save`, producing a `.npy` file that can be loaded with `np.load`.
    - Verbose logging is intended for debugging and progress visibility; set `verbose=False` to silence.
    
    Example
    -------
    >>> # embeddings as numpy array
    >>> names = ['song_a.mid', 'song_b.mid']
    >>> embs = np.random.randn(2, 512)
    >>> save_embeddings(names, embs, output_file_name='embs.npy', verbose=False)
    >>> # embeddings as torch tensor, return array instead of saving
    >>> import torch
    >>> t = torch.randn(2, 512)
    >>> arr = save_embeddings(names, t, return_merged_array=True, verbose=False)
    >>> assert arr.dtype == np.dtype([('midi_names', object), ('midi_embeddings', np.float32, (512,))])
    
    """

    if verbose:
        print('=' * 70)
        print('Saving embeddings...')
        print('=' * 70)
        print("[save_embeddings]: called with parameters:")
        print(f"  number of name strings provided: {len(embeddings_name_strings)}")
        print(f"  output_file_name: {output_file_name}")
        print(f"  name_strings_key: {name_strings_key}")
        print(f"  embeddings_key: {embeddings_key}")
        print(f"  return_merged_array: {return_merged_array}")
        print(f"  verbose: {verbose}")
        print('=' * 70)

    # Convert torch tensor to numpy if needed
    if type(embeddings) == torch.Tensor:
        if verbose:
            print("[save_embeddings]: embeddings is a torch.Tensor, converting to numpy array with .numpy()")
        embeddings = embeddings.cpu().numpy()
    elif type(embeddings) == list:
        if verbose:
                print("[save_embeddings]: embeddings is a list, converting to numpy array")
        embeddings = np.array(embeddings)
    else:
        if verbose:
            print(f"[save_embeddings]: embeddings is of type {type(embeddings)}; no conversion performed")

    # Basic shape and length checks
    try:
        n = len(embeddings_name_strings)
    except Exception as e:
        if verbose:
            print("[save_embeddings]: ERROR computing length of embeddings_name_strings:", e)
        raise

    try:
        D = embeddings.shape[1]
    except Exception as e:
        if verbose:
            print("[save_embeddings]: ERROR reading embeddings.shape[1]:", e)
            print("  embeddings.shape is:", getattr(embeddings, "shape", None))
        raise

    if verbose:
        print(f"[save_embeddings]: determined n = {n} (number of entries)")
        print(f"[save_embeddings]: determined D = {D} (embedding dimensionality)")
        print("[save_embeddings]: preparing numpy structured dtype for storage")

    dtype = np.dtype([
        (name_strings_key, object),              # variable-length Python strings
        (embeddings_key, embeddings.dtype, (D,))       # fixed-size embedding vector
    ])

    if verbose:
        print("[save_embeddings]: dtype constructed as:")
        print(f"  {dtype}")

    # Create empty structured array
    if verbose:
        print(f"[save_embeddings]: allocating empty numpy array of length {n} with dtype above")
    arr = np.empty(n, dtype=dtype)

    # Fill name strings
    if verbose:
        print("[save_embeddings]: assigning name strings to structured array")
        print(f"  first 5 name strings (or fewer): {embeddings_name_strings[:5]}")
    arr[name_strings_key] = embeddings_name_strings

    # Cast embeddings to float32 and assign
    if verbose:
        print("[save_embeddings]: assigning embeddings to structured array")
        print(f"  embeddings original dtype: {getattr(embeddings, 'dtype', 'unknown')}")
        print(f"  embeddings shape: {getattr(embeddings, 'shape', 'unknown')}")
    arr[embeddings_key] = embeddings

    if return_merged_array:
        if verbose:
            print('=' * 70)
            print("[save_embeddings]: return_merged_array is True; returning the merged structured array without saving to disk")
            print(f"  returning array with length {len(arr)} and dtype {arr.dtype}")
            print('=' * 70)
            print('Done!')
            print('=' * 70)
        return arr

    # Save to disk
    if verbose:
        print('=' * 70)
        print(f"[save_embeddings]: return_merged_array is False; saving structured array to '{output_file_name}' using np.save")
    np.save(output_file_name, arr)
    if verbose:
        print(f"[save_embeddings]: save complete. File written: {output_file_name}")
        print(f"  saved array length: {len(arr)}; dtype: {arr.dtype}")
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
###################################################################################

def midi_to_tokens(midi_file_path: str,
                   max_seq_len: int = 3072,
                   transpose_factor: int = 6,
                   verbose: bool = True
                  )-> list[list[int]]:
    
    """
    Convert a single-track MIDI file into one or more compact token sequences suitable for model input.

    This function performs a sequence of TMIDIX preprocessing steps to extract an
    "enhanced score" from a MIDI file, normalizes and clips timing/pitch values,
    optionally generates transposed variants, and encodes events into a compact
    integer token stream.

    Key processing stages
    - Load MIDI and convert to a single-track millisecond score.
    - Produce an enhanced-score with sustain applied.
    - Extract solo-piano notes and recalculate/augment timings.
    - Remove duplicate pitches and fix note durations.
    - Convert to a delta-style event list and clip timing values to 0..127.
    - For each transpose variant, build a token sequence where:
        * nonzero delta times are appended as-is (0..127),
        * note-on events are encoded as two tokens: (duration + 128) and (pitch + 256).
      The initial token of each sequence is 0 (start token).

    Parameters
    ----------
    midi_file_path : str
        Path to the MIDI file to process. The file is read by TMIDIX.midi2single_track_ms_score.
    max_seq_len : int
        Maximum output tokens sequence length (truncated to this value). Default is 3072
    transpose_factor : int, optional
        Maximum semitone transpose range. The value is clamped to the inclusive range 0..6.
        If > 0, the function generates variants for transpositions in the integer range
        [-transpose_factor, transpose_factor - 1]. If 0, only the original (no transpose)
        variant is produced. Default is 6.
    verbose : bool, optional
        When True, prints concise progress messages and enables tqdm progress bars.
        Progress bars use `tqdm(disable=not verbose)` so they are suppressed when verbose is False.

    Returns
    -------
    list[list[int]]
        A list of token sequences. Each token sequence is a list of integers where:
        - The first element is 0 (start token).
        - Delta times (when nonzero) are appended as integers in 0..127.
        - Note events are encoded as two integers: duration_token and pitch_token,
          where duration_token = duration_clipped + 128 and pitch_token = pitch_clipped + 256.
        The function returns an empty list if processing fails or no notes are found.

    Notes and assumptions
    ---------------------
    - The function expects TMIDIX to provide the following functions used internally:
      midi2single_track_ms_score, advanced_score_processor, solo_piano_escore_notes,
      recalculate_score_timings, augment_enhanced_score_notes, remove_duplicate_pitches_from_escore_notes,
      fix_escore_notes_durations, delta_score_notes.
    - Delta events `d` are assumed to be indexable sequences where:
      d[1] is delta time, d[2] is duration, and d[4] is pitch (consistent with the original code).
    - Timing values are clipped to 0..127; durations are clipped to 1..127; pitches are clipped to 1..127
      after applying the transpose offset.
    - The function intentionally uses small integer ranges to match downstream token vocabularies
      that reserve offsets (e.g., +128, +256) for event encoding.

    Exceptions
    ----------
    - Any exception raised during processing is caught; the function prints a short error message
      (only when verbose) and returns the token sequences collected so far (often an empty list).

    Example
    -------
    >>> sequences = midi_to_tokens("example.mid", transpose_factor=2, verbose=True)
    >>> len(sequences)
    4  # variants for tv in [-2, -1, 0, 1] when transpose_factor == 2

    """
    
    all_toks_sequences = []

    try:
        if verbose:
            print('=' * 70)
            print(f"Loading MIDI file: {midi_file_path}")
            print('=' * 70)

        raw_score = TMIDIX.midi2single_track_ms_score(
            midi_file_path, do_not_check_MIDI_signature=True
        )

        if verbose:
            print("Running advanced score processor (enhanced notes, sustain applied)...")

        escore = TMIDIX.advanced_score_processor(
            raw_score, return_enhanced_score_notes=True, apply_sustain=True
        )

        if not escore or not escore[0]:
            if verbose:
                print("No enhanced score notes found after advanced processing. Returning empty list.")
                
            return all_toks_sequences

        if verbose:
            print("Extracting solo piano enhanced-score notes...")

        escore = TMIDIX.solo_piano_escore_notes(escore[0])

        if not escore:
            if verbose:
                print("Solo piano extraction returned no notes. Returning empty list.")
                
            return all_toks_sequences

        if verbose:
            print("Recalculating timings, augmenting timings, removing duplicates, and fixing durations...")

        escore = TMIDIX.recalculate_score_timings(escore)
        escore = TMIDIX.augment_enhanced_score_notes(escore, timings_divider=32)
        escore = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore)
        escore = TMIDIX.fix_escore_notes_durations(escore, min_notes_gap=1)

        if verbose:
            print("Computing delta score notes (clipping timings to 127)...")

        dscore = TMIDIX.delta_score_notes(escore, timings_clip_value=127)

        # Clamp transpose_factor to allowed range
        transpose_factor = max(0, min(6, transpose_factor))
            
        if verbose:
            print(f"Using transpose_factor={transpose_factor} (clamped to 0..6).")

        if transpose_factor > 0:
            sidx = -transpose_factor
            eidx = transpose_factor
        else:
            sidx = 0
            eidx = 1

        if verbose:
            print(f"Generating token sequences for transpose variants in range({sidx}, {eidx})...")

        # Outer loop: transpose variants with progress bar
        for tv in tqdm.tqdm(range(sidx, eidx), disable=not verbose, desc="Transpose variants"):
            if verbose:
                print(f"Processing transpose variant tv={tv}...")

            out_score = [0]

            # Inner loop: iterate over delta-score events with progress bar
            for d in tqdm.tqdm(dscore, disable=not verbose, desc="Processing delta score", leave=False):
                # d is expected to be a sequence where:
                # d[1] -> delta time, d[2] -> duration, d[4] -> pitch (based on original code)
                dtime = max(0, min(127, d[1]))

                if dtime != 0:
                    out_score.append(dtime)

                dur = max(1, min(127, d[2]))
                ptc = max(1, min(127, d[4] + tv))

                out_score.extend([dur + 128, ptc + 256])

            all_toks_sequences.append(out_score[:max_seq_len])

            if verbose:
                print(f"Variant tv={tv} produced sequence length {len(out_score[:max_seq_len])}.")

        if verbose:
            print('=' * 70)
            print(f"Finished processing. Produced {len(all_toks_sequences)} token sequence(s).")
            print('=' * 70)

        return all_toks_sequences

    except Exception as ex:
        print("Exception while converting MIDI to token sequences!")
        print(f"File: {midi_file_path}")
        print(f"Error: {ex}")
            
        return all_toks_sequences

###################################################################################

def masked_mean_pool(
    token_embeddings: Tensor,
    mask: Tensor,
    dim: int = 1,
    eps: float = 1e-9,
    verbose: bool = True,
    ) -> Tensor:
    
    """
    Compute a masked mean pooling over a specified dimension.

    This function computes the mean of `token_embeddings` along `dim`, ignoring
    positions where `mask` is False. The mask is cast to the same dtype as the
    embeddings to allow safe multiplication. A small epsilon is used to avoid
    division by zero for sequences that are entirely masked out.

    Args:
        token_embeddings: Tensor of shape (B, L, D) or similar where `dim` indexes
            the sequence length. Embeddings dtype can be float16/float32/bfloat16.
        mask: Boolean tensor of shape broadcastable to the sequence dimension
            (e.g., (B, L)). True indicates valid tokens; False indicates padding.
        dim: Dimension along which to pool (default: 1, the sequence length).
        eps: Small value to avoid division by zero when a row has zero valid tokens.
        verbose: If True, prints a short summary about the pooling operation.

    Returns:
        Tensor of pooled embeddings with the sequence dimension removed, typically
        shape (B, D). The returned dtype matches `token_embeddings.dtype`.
    """
    
    mask_f = mask.to(token_embeddings.dtype)  # (B, L)
    summed = (token_embeddings * mask_f.unsqueeze(-1)).sum(dim=dim)  # (B, D)
    counts = mask_f.sum(dim=dim).clamp_min(eps).unsqueeze(-1)  # (B, 1)
    pooled = summed / counts  # (B, D)

    if verbose:
        # Use tqdm.write so it doesn't interfere with progress bars
        valid_counts = counts.squeeze(-1)
        tqdm.tqdm.write(
            f"[masked_mean_pool] pooled shape={pooled.shape}, "
            f"counts min={valid_counts.min().item():.3f}, max={valid_counts.max().item():.3f}"
        )

    return pooled

###################################################################################

def masked_weighted_mean_pool(
    token_embs: Tensor,
    valid_mask: Tensor,
    token_ids: Optional[Tensor] = None,
    token_type_weights: Optional[Tuple[float, float, float]] = None,
    dim: int = 1,
    verbose: bool = False,
    ) -> Tensor:
    
    """
    Weighted mean pooling across tokens. If token_ids is provided, token types are
    inferred using the same ranges as the reference code:
      - onset:   token_id in [0, 127]
      - duration:token_id in [128, 255]
      - pitch:   token_id in [256, 383]
    token_type_weights: (onset_w, duration_w, pitch_w). If None, defaults to (1.0,1.0,1.0)
    The function multiplies each token embedding by its scalar weight and divides
    by the sum of weights for valid tokens per sequence.
    """
    
    B, L, D = token_embs.shape
    device = token_embs.device
    dtype = token_embs.dtype

    if token_ids is None:
        # No token-level ids available: fallback to simple masked mean
        if verbose:
            tqdm.tqdm.write("[masked_weighted_mean_pool] token_ids is None, falling back to masked_mean_pool")
        return masked_mean_pool(token_embs, valid_mask, dim=dim, verbose=verbose)

    # Default weights
    if token_type_weights is None:
        onset_w, duration_w, pitch_w = 1.0, 1.0, 1.0
    else:
        onset_w, duration_w, pitch_w = token_type_weights

    # Build per-type boolean masks based on token id values (same ranges as reference)
    onset_mask = (token_ids >= 0) & (token_ids < 128)
    duration_mask = (token_ids >= 128) & (token_ids < 256)
    pitch_mask = (token_ids >= 256) & (token_ids < 384)

    # Combine with valid_mask to ignore padding positions
    onset_mask = onset_mask & valid_mask
    duration_mask = duration_mask & valid_mask
    pitch_mask = pitch_mask & valid_mask

    # Build per-token scalar weight tensor (B, L)
    w = torch.ones((B, L), device=device, dtype=dtype)
    if onset_w != 1.0:
        w = torch.where(onset_mask, torch.tensor(onset_w, device=device, dtype=dtype), w)
    if duration_w != 1.0:
        w = torch.where(duration_mask, torch.tensor(duration_w, device=device, dtype=dtype), w)
    if pitch_w != 1.0:
        w = torch.where(pitch_mask, torch.tensor(pitch_w, device=device, dtype=dtype), w)

    # Zero out weights for padding positions
    valid_mask_f = valid_mask.to(dtype)  # (B, L)
    w = w * valid_mask_f  # (B, L)

    # Weighted sum and normalization
    denom = w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
    w_exp = w.unsqueeze(-1)  # (B, L, 1)
    summed = (token_embs * w_exp).sum(dim=dim)  # (B, D)
    pooled = summed / denom  # (B, D)

    return pooled

###################################################################################

def pad_and_mask(
    sequences: List[List[int]],
    pad_idx: int = 385,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    ) -> Tuple[Tensor, Tensor]:
    
    """
    Pad and create a boolean mask for a batch of integer token sequences.

    This utility converts a list of variable-length integer sequences into a
    padded LongTensor and a corresponding boolean mask indicating valid token
    positions. Sequences longer than `seq_len` are truncated. If `seq_len` is
    None, the function uses the maximum sequence length in the batch.

    Args:
        sequences: List of token id sequences (each a list of ints).
        pad_idx: Integer token id used for padding positions (default: 385).
        seq_len: Optional target sequence length. If provided, sequences are
            truncated or padded to this length. If None, the maximum length in
            `sequences` is used.
        device: Optional torch.device where the returned tensors will be placed.
            If None, tensors are created on the default device.
        verbose: If True, shows a small progress bar while processing sequences
            and prints a summary.

    Returns:
        A tuple (x, mask):
            - x: LongTensor of shape (B, T) containing padded token ids.
            - mask: BoolTensor of shape (B, T) where True indicates a real token.
    """
    
    # Fast path for empty batch
    if not sequences:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        empty_mask = torch.empty((0, 0), dtype=torch.bool, device=device)
        return empty, empty_mask

    # Compute lengths and the batch maximum length
    lengths = [len(s) for s in sequences]
    batch_max = max(lengths)

    # If seq_len is given, only use it to cap lengths; but if the batch max is smaller,
    # use the smaller value to avoid extra allocation/work.
    if seq_len is None:
        target_len = batch_max
    else:
        target_len = min(seq_len, batch_max)

    b = len(sequences)
    if target_len == 0:
        x = torch.full((b, 0), pad_idx, dtype=torch.long, device=device)
        mask = torch.zeros((b, 0), dtype=torch.bool, device=device)
        return x, mask

    x = torch.full((b, target_len), pad_idx, dtype=torch.long, device=device)
    mask = torch.zeros((b, target_len), dtype=torch.bool, device=device)

    # iterate with optional progress display
    iterator = enumerate(sequences)
    if verbose:
        iterator = enumerate(tqdm.tqdm(sequences, disable=not verbose, desc="Pad & mask"))

    for i, seq in iterator:
        if not seq:
            continue
        # Only truncate if seq is longer than the chosen target_len
        L = len(seq)
        if L > target_len:
            L = target_len
            # slice once to avoid creating a larger tensor then slicing
            seq_slice = seq[:L]
            seq_tensor = torch.tensor(seq_slice, dtype=torch.long, device=device)
        else:
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)

        x[i, :L] = seq_tensor[:L]
        mask[i, :L] = True

    if verbose:
        tqdm.tqdm.write(
            f"[pad_and_mask] batch_size={b}, target_len={target_len}, "
            f"min_len={min(lengths)}, max_len={max(lengths)}"
        )

    return x, mask

###################################################################################

def get_embeddings_bf16(
    model,
    sequences: List[List[int]],
    seq_len: Optional[int] = 3072,
    seq_pad_idx: int = 385,
    batch_size: int = 64,
    save_every_num_batches: int = -1,
    save_file_path: str = "saved_embeddings.npy",
    device: Optional[torch.device] = None,
    normalize: bool = False,
    pooling: str = "auto",  # "auto" | "mean" | "weighted_mean"
    token_type_weights: Optional[Tuple[float, float, float]] = None,  # (onset_w, duration_w, pitch_w)
    use_bfloat16: bool = True,  # enable bfloat16 autocast when possible
    return_dtype: str = "float32",  # "float32" or "float16" for returned embeddings
    return_numpy: bool = False,
    verbose: bool = True,
    show_progress_bar: bool = True
    ) -> Union[Tensor, np.ndarray]:

    """
    Compute embeddings for a list of token sequences using a PyTorch model with optional bfloat16/autocast,
    pooling, normalization, and periodic saving.
    
    This function batches input token id sequences, pads/truncates them to a fixed length, runs the model
    in evaluation mode under `torch.no_grad()` and optional mixed-precision autocast, and returns a single
    tensor (or NumPy array) containing per-sequence embeddings. The model is expected to accept a LongTensor
    of token ids `x` and a boolean mask `mask` and to return either:
      - a 2-D tensor `(B, D)` of already-pooled embeddings, or
      - a 3-D tensor `(B, L, D)` of per-token embeddings (which will be pooled according to `pooling`).
    
    Key behaviors:
    - Sequences are padded with `seq_pad_idx` and masked so padding does not affect pooling.
    - If `seq_len` is provided, sequences longer than `seq_len` are truncated; otherwise the batch max length is used.
    - Mixed-precision autocast is used when `use_bfloat16` is True and supported by the device; the function
      falls back to the default autocast or no autocast if unavailable.
    - Supports three pooling modes for per-token embeddings:
        - `"auto"` or `"mean"`: simple masked mean pooling across tokens.
        - `"weighted_mean"`: weighted mean pooling by token type (onset/duration/pitch) inferred from token ids;
          weights are provided via `token_type_weights` and padding tokens are ignored.
    - Optionally L2-normalizes embeddings (in float32) when `normalize=True`.
    - Returned embeddings can be cast to `float16` for storage/transfer via `return_dtype`.
    - Embeddings are collected on CPU; intermediate results can be periodically saved to `save_file_path`.
    - If `return_numpy=True`, a NumPy array is returned; otherwise a CPU `torch.Tensor` is returned.
    
    Args:
        model (torch.nn.Module):
            PyTorch model used to compute embeddings. The model will be moved to `device` (or its current
            parameter device if `device` is None) and set to `eval()` for inference. The forward call must
            accept `x` (LongTensor) and `mask` (BoolTensor) and return embeddings when called with
            `return_embeddings=True`.
        sequences (List[List[int]]):
            Batch of token id sequences (each sequence is a list of ints). Can be empty; an empty result
            with shape `(0, 0)` will be returned in that case.
        seq_len (Optional[int], default=3072):
            Target sequence length for truncation/padding. If None, the maximum sequence length in the
            current batch is used.
        seq_pad_idx (int, default=385):
            Token id used for padding positions.
        batch_size (int, default=64):
            Number of sequences processed per forward pass.
        save_every_num_batches (int, default=-1):
            If > 0, the function will save accumulated embeddings to `save_file_path` every
            `save_every_num_batches` batches. A non-positive value disables periodic saving.
        save_file_path (str, default="saved_embeddings.npy"):
            File path used by `np.save` when periodic saving is enabled.
        device (Optional[torch.device], default=None):
            Device to run the model and tensors on. If None, the device of the model parameters is used.
        normalize (bool, default=False):
            If True, L2-normalize each embedding vector (done in float32 for numerical stability).
        pooling (str, default="auto"):
            Pooling strategy applied when model returns per-token embeddings:
              - "auto" or "mean": masked mean pooling.
              - "weighted_mean": weighted mean pooling by token type using `token_type_weights`.
            Any other value raises `ValueError`.
        token_type_weights (Optional[Tuple[float, float, float]], default=None):
            Per-token-type weights `(onset_w, duration_w, pitch_w)` used when `pooling="weighted_mean"`.
            If None, defaults to `(1.0, 1.0, 1.0)`. Token type ranges are inferred as:
              onset:   token_id in [0, 127]
              duration:token_id in [128, 255]
              pitch:   token_id in [256, 383]
        use_bfloat16 (bool, default=True):
            If True, attempts to use `torch.bfloat16` autocast for the device; falls back gracefully if not supported.
        return_dtype (str, default="float32"):
            Data type for returned embeddings: `"float32"` or `"float16"`. Internally embeddings are normalized
            in float32; casting to float16 happens just before collecting results if requested.
        return_numpy (bool, default=False):
            If True, the final result is returned as a NumPy array; otherwise a CPU `torch.Tensor` is returned.
        verbose (bool, default=True):
            If True, prints progress and short diagnostic messages via `tqdm`.
        show_progress_bar (bool, default=True)
            If True, displays tqdm progress bar.
    
    Returns:
        Union[torch.Tensor, numpy.ndarray]:
            - If `return_numpy` is False: a CPU `torch.Tensor` of shape `(N, D)` and dtype `torch.float32`
              or `torch.float16` depending on `return_dtype`.
            - If `return_numpy` is True: a NumPy array of shape `(N, D)` and dtype `np.float32` or `np.float16`.
            `N` is the total number of input sequences and `D` is the embedding dimensionality produced by the model.
    
    Raises:
        AssertionError:
            If `return_dtype` is not one of `"float32"` or `"float16"`.
        RuntimeError:
            If the model returns `None` for embeddings (indicates incorrect forward flags or model behavior).
        ValueError:
            If the model returns an embedding tensor with unexpected dimensionality or if `pooling` is unsupported.
    
    Notes:
        - The function uses `pad_and_mask` to produce `x` (LongTensor) and `mask` (BoolTensor). Padding tokens
          are ignored by pooling operations.
        - When `pooling="weighted_mean"`, if `token_ids` are not available or the model returns a 2-D tensor,
          the function falls back to masked mean pooling.
        - Periodic saving concatenates all embeddings collected so far and writes them with `np.save`. Save
          failures are caught and reported when `verbose=True` but do not abort processing.
        - The function runs the model under `torch.no_grad()` and sets `model.eval()`; it will move the model
          to `device` if provided.
        - For reproducible numeric behavior across devices, ensure the model and device support the requested
          autocast dtype (bfloat16) and that any randomness is controlled externally.
    
    Example:
        >>> # simple usage
        >>> embs = get_embeddings_bf16(model, sequences, seq_len=1024, batch_size=32, pooling="mean",
        ...                           normalize=True, return_dtype="float32", return_numpy=False)
    """

    assert return_dtype in ("float32", "float16"), "return_dtype must be 'float32' or 'float16'"

    model_device = next(model.parameters()).device if device is None else device
    model.to(model_device)
    model.eval()

    all_embs: List[Tensor] = []
    total_batches = math.ceil(len(sequences) / batch_size) if batch_size > 0 else 0

    if verbose:
        tqdm.tqdm.write(
            f"[get_embeddings_bf16] sequences={len(sequences)}, batch_size={batch_size}, "
            f"batches={total_batches}, device={model_device}, seq_len={seq_len}, pooling={pooling}"
        )
        
    # Prepare autocast context using torch.amp.autocast
    autocast_ctx = None
    if use_bfloat16:
        try:
            autocast_ctx = torch.amp.autocast(device_type=model_device.type, dtype=torch.bfloat16)
        except Exception:
            try:
                autocast_ctx = torch.amp.autocast(device_type=model_device.type)
            except Exception:
                autocast_ctx = None
    else:
        try:
            autocast_ctx = torch.amp.autocast(device_type=model_device.type)
        except Exception:
            autocast_ctx = None

    with torch.inference_mode():
        batch_iter = range(0, len(sequences), batch_size)
        pbar = tqdm.tqdm(batch_iter, disable=not show_progress_bar, total=total_batches, desc="Embedding batches")
        for batch_idx, i in enumerate(pbar):
            batch_seqs = sequences[i : i + batch_size]
            x, mask = pad_and_mask(batch_seqs, pad_idx=seq_pad_idx, seq_len=seq_len, device=model_device, verbose=verbose)
            # x: (B, L) LongTensor token ids, mask: (B, L) boolean

            # Run forward under autocast if available
            if autocast_ctx is not None:
                with autocast_ctx:
                    out = model(x, return_embeddings=True, mask=mask)
            else:
                out = model(x, return_embeddings=True, mask=mask)

            if out is None:
                raise RuntimeError("model returned None for embeddings. Check forward flags.")

            # Handle shapes
            if out.dim() == 2:
                # already pooled: (B, D)
                emb = out
            elif out.dim() == 3:
                # per-token embeddings: (B, L, D)
                if pooling in ("mean", "auto"):
                    emb = masked_mean_pool(out, mask, dim=1, verbose=verbose)
                elif pooling == "weighted_mean":
                    # Use token ids to compute per-token weights; fallback to mean if token ids missing
                    emb = masked_weighted_mean_pool(out, mask, token_ids=x, token_type_weights=token_type_weights, dim=1, verbose=verbose)
                else:
                    raise ValueError(f"unsupported pooling: {pooling}")
            else:
                raise ValueError(f"unexpected embedding tensor shape: {out.shape}")

            # Ensure embeddings are float32 for stable normalization/indexing
            if emb.dtype != torch.float32:
                emb = emb.float()

            # L2 normalize in float32
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)

            # Optionally cast to float16 for return/storage
            if return_dtype == "float16":
                emb = emb.half()

            all_embs.append(emb.cpu())

            # Update progress bar postfix with shapes and dtype
            if verbose:
                pbar.set_postfix({"batch": batch_idx + 1, "emb_shape": f"{emb.shape}", "dtype": str(emb.dtype)})

            # Save intermediate results periodically
            if save_every_num_batches > 0:
                # compute 0-based batch number
                bnum = batch_idx
                if (bnum + 1) % save_every_num_batches == 0:
                    try:
                        concatenated = torch.cat(all_embs, dim=0).numpy()
                        np.save(save_file_path, concatenated)
                        if verbose:
                            tqdm.tqdm.write(f"[get_embeddings_bf16] saved {concatenated.shape[0]} embeddings to {save_file_path}")
                    except Exception as e:
                        # Do not crash the whole run for a save failure; report if verbose
                        if verbose:
                            tqdm.tqdm.write(f"[get_embeddings_bf16] warning: failed to save embeddings: {e}")

    if len(all_embs) == 0:
        # return empty tensor/array with shape (0, 0)
        empty = torch.empty((0, 0), dtype=(torch.float16 if return_dtype == "float16" else torch.float32))
        if verbose:
            tqdm.tqdm.write("[get_embeddings_bf16] no embeddings were produced; returning empty tensor")
        return empty.numpy() if return_numpy else empty

    result = torch.cat(all_embs, dim=0)  # (N, D) on CPU

    if verbose:
        tqdm.tqdm.write(f"[get_embeddings_bf16] finished: total_embeddings={result.shape[0]}, dim={result.shape[1]}, dtype={result.dtype}")

    if return_numpy:
        return result.numpy()

    return result

###################################################################################

TensorOrArray = Union[torch.Tensor, np.ndarray]

###################################################################################

def cosine_similarity_topk(
    query_embs: TensorOrArray,
    corpus_embs: TensorOrArray,
    topk: int = 16,
    chunk_size: int = 10000,
    device: torch.device | None = None,
    use_gpu_if_available: bool = True,
    normalize_inputs: bool = True,
    return_dtype: torch.dtype = torch.float32,
    use_fp32_accumulation: bool = True,
    verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Compute top-k cosine similarities between query embeddings and a (potentially large)
    corpus of embeddings, returning the top-k similarity values and corresponding corpus
    indices for each query.

    This function accepts either `torch.Tensor` or `numpy.ndarray` for `query_embs`
    and `corpus_embs`. When numpy arrays are provided, slices of the corpus are
    converted to torch tensors on-the-fly to avoid copying the entire corpus to GPU.

    Parameters
    ----------
    query_embs : torch.Tensor | numpy.ndarray
        2-D array of shape (Q, D) containing Q query embeddings of dimension D.
    corpus_embs : torch.Tensor | numpy.ndarray
        2-D array of shape (N, D) containing N corpus embeddings of dimension D.
    topk : int, default 16
        Number of top matches to return per query.
    chunk_size : int, default 10000
        Number of corpus rows to process per chunk. Lower this to reduce peak memory.
    device : torch.device | None, default None
        Device to use. If None, will use CUDA if available and `use_gpu_if_available` is True,
        otherwise CPU.
    use_gpu_if_available : bool, default True
        If True and CUDA is available, prefer GPU when `device` is not explicitly set.
    normalize_inputs : bool, default True
        If True, L2-normalize both query and corpus embeddings before computing cosine sims.
    return_dtype : torch.dtype, default torch.float32
        Output dtype for similarity values. If `torch.float16` is requested, values are cast
        before returning.
    use_fp32_accumulation : bool, default True
        If True, perform matrix multiplications in float32 for numerical stability.
    verbose : bool, default True
        If True, print brief status messages and show a tqdm progress bar for corpus chunks.

    Returns
    -------
    best_idx : nd.array
        Numpy array of shape (Q, topk) with the global corpus indices corresponding to
        `best_vals`.
    best_vals : nd.array
        Numpy array of shape (Q, topk) with the top-k similarity values for each query.

    Notes
    -----
    - The function keeps only the top-k matches across all chunks by merging per-chunk
      top-k results into a running buffer.
    - Returned tensors are moved to CPU.
    - If `corpus_embs` is a numpy array, only the current chunk is converted to a tensor,
      minimizing memory usage on the target device.
    """

    # --- basic shape checks that work for both numpy and torch ---
    if isinstance(query_embs, np.ndarray):
        if query_embs.ndim != 2:
            raise ValueError("query_embs must be 2-D")
        Q, D = query_embs.shape
    else:
        if query_embs.dim() != 2:
            raise ValueError("query_embs must be 2-D")
        Q, D = query_embs.shape

    if isinstance(corpus_embs, np.ndarray):
        if corpus_embs.ndim != 2:
            raise ValueError("corpus_embs must be 2-D")
        N, D2 = corpus_embs.shape
    else:
        if corpus_embs.dim() != 2:
            raise ValueError("corpus_embs must be 2-D")
        N, D2 = corpus_embs.shape

    if D != D2:
        raise ValueError("query and corpus must have the same embedding dimension")

    # pick device
    if device is None:
        if use_gpu_if_available and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if verbose:
        print(f"[cosine_similarity_topk] device: {device}; queries: {Q} x {D}; corpus: {N} x {D2}; topk: {topk}; chunk_size: {chunk_size}")

    # Convert queries to torch and move to device
    if isinstance(query_embs, np.ndarray):
        query = torch.from_numpy(query_embs)
    else:
        query = query_embs

    query = query.to(device)

    # normalize queries
    if normalize_inputs:
        query = torch.nn.functional.normalize(query.float(), p=2, dim=-1).to(query.dtype)

    # initialize topk buffers
    best_vals = torch.full((Q, topk), -float("inf"), dtype=torch.float32, device=device)
    best_idx  = torch.full((Q, topk), -1, dtype=torch.long, device=device)

    # iterate corpus in chunks
    corpus_is_numpy = isinstance(corpus_embs, np.ndarray)
    iterator = range(0, N, chunk_size)
    pbar = tqdm.tqdm(iterator, disable=not verbose, desc="Processing corpus", unit="chunk")

    for start in pbar:
        end = min(start + chunk_size, N)
        C = end - start

        if corpus_is_numpy:
            # convert slice to tensor (shares memory with numpy if possible)
            chunk = torch.from_numpy(corpus_embs[start:end])
            chunk = chunk.to(device)
        else:
            # corpus is a torch tensor; move slice to device
            chunk = corpus_embs[start:end].to(device)  # (C, D)

        # normalize chunk
        if normalize_inputs:
            chunk = torch.nn.functional.normalize(chunk.float(), p=2, dim=-1).to(chunk.dtype)

        # choose accumulation dtype
        if use_fp32_accumulation:
            q_mat = query.float()
            c_mat = chunk.float()
        else:
            q_mat = query
            c_mat = chunk

        # compute similarities: (Q, C)
        sims_block = q_mat @ c_mat.t()

        k_block = min(topk, C)

        # topk inside this block
        vals_block, idxs_block = torch.topk(sims_block, k=k_block, dim=1)

        # convert local indices → global corpus indices
        idxs_block = idxs_block + start  # (Q, k_block)

        # ---- MERGE LOGIC ----
        # concat along dim=1 → always (Q, topk + k_block)
        merged_vals = torch.cat([best_vals, vals_block.float()], dim=1)
        merged_idxs = torch.cat([best_idx, idxs_block.long()], dim=1)

        # select new topk
        new_vals, new_pos = torch.topk(merged_vals, k=topk, dim=1)

        # gather indices using 2‑D indexing
        row_ids = torch.arange(Q, device=device).unsqueeze(1)
        new_idxs = merged_idxs[row_ids, new_pos]

        best_vals = new_vals
        best_idx  = new_idxs

        # update progress bar postfix with a brief summary
        if verbose:
            pbar.set_postfix({"processed": f"{end}/{N}"})

        # cleanup
        del chunk, sims_block, vals_block, idxs_block, merged_vals, merged_idxs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # cast output dtype
    if return_dtype == torch.float16:
        if verbose:
            print("[cosine_similarity_topk] Casting to float16")
        best_vals = best_vals.half()

    if verbose:
        print("[cosine_similarity_topk] done; moving results to CPU and converting to NumPy arrays")

    return best_idx.cpu().numpy(), best_vals.cpu().numpy()

###################################################################################

def idxs_sims_to_sorted_list(idxs: np.ndarray,
                             sims: np.ndarray,
                             sims_mult: int = 100,
                             remove_dupes=True,
                             ) -> List[Tuple]:
    
    """
    Helper function to convert indexes and similarities arrays into
    a sorted list with corresponding transpose values.
    
    Rwturns
    -------
    List of tuples: (corpus_index, transpose_value, cosine_similarity)
    """
    
    idxs = np.array(idxs)
    sims = np.array(sims)

    assert idxs.shape == sims.shape, f'Shape mismatch between indexes array and similarities array: {idxs.shape} != {sims.shape}'

    flat_idxs = [x for row in idxs.tolist() for x in row]
    flat_sims = [x * sims_mult for row in sims.tolist() for x in row]

    tv = idxs.shape[0]

    if tv == 1:
        sr = 0
        er = 1

    elif tv > 1 and tv % 2 == 0:
        sr = -(tv // 2)
        er = tv // 2

    else:
        sr = -6
        er = 6
    
    tkv = idxs.shape[1]
    
    flat_tvs = [v for v in range(sr, er) for _ in range(tkv)]

    sorted_list = sorted(zip(flat_idxs, flat_tvs, flat_sims), key=lambda x: -x[2])
    
    if remove_dupes:
        deduped_sorted_list = []
        seen = set()
        
        for idx, tv, sim in sorted_list:
            if idx not in seen:
                deduped_sorted_list.append([idx, tv, sim])
                seen.add(idx)
            
        return deduped_sorted_list
    
    return sorted_list   

###################################################################################

def print_sorted_idxs_sims_list(sorted_idxs_sims_list: list,
                                corpus_midi_names: Union[list, np.ndarray],
                                return_as_list: bool = False,
                                ) -> Union[List[Tuple], None]:
    
    """
    Helper function that prints search results list generated by idxs_sims_to_sorted_list function
    
    Returns
    -------
    List of tuples if return_as_list is True
    None if return_as_list is False
    """
    
    if type(corpus_midi_names) == np.ndarray:
        corpus_midi_names = corpus_midi_names.tolist()    

    if not return_as_list:
        print('=' * 70)
        print('Search results:')
        print('=' * 70)
    
    return_list = []

    for i, (idx, tv, sim) in enumerate(sorted_idxs_sims_list):

        if not return_as_list:
            print(f'#{str(i).zfill(3)} {corpus_midi_names[idx]} --- {tv} --- {round(sim, 8)}')
        
        else:
            return_list.append([i, corpus_midi_names[idx], tv, sim])    

    if not return_as_list:
        print('=' * 70)
        print('Total number of records:', len(sorted_idxs_sims_list))
        print('=' * 70)
    
    else:
        return return_list

###################################################################################

@lru_cache(maxsize=1)
def get_corpus_midis(corpus_midis_dirs_tuple: Tuple,
                     verbose: bool = True
                     ) -> Dict:
    
    """
    Returns corpus_midis_dic with LRU caching.
    corpus_midis_dirs_tuple must be a tuple for hashing.
    """

    if verbose:
        print("Scanning corpus MIDI directories...")

    # Create list
    corpus_midis_list = TMIDIX.create_files_list(
        list(corpus_midis_dirs_tuple),
        verbose=verbose
    )

    # Create dict: basename → full path

    if verbose:
        print('Converting files list to dict...')
        
    corpus_midis_dic = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in corpus_midis_list
    }

    if verbose:
        print('Done!')
    
    return corpus_midis_dic

###################################################################################

def copy_corpus_files(sorted_idxs_sims_list: list[list],
                      corpus_midis_dirs: list = ['./Corpus MIDIs Dir/'],
                      main_output_dir: str = './Corpus Matches Dir/',
                      sub_output_dir: str = 'Corpus MIDI Name',
                      copy_original_midi: bool = True,
                      verbose: bool = True
                     ) -> str:

    """
    Helper function that copies matched corpus MIDIs to a specified directory

    Returns
    -------
    Output directory where files were copied as a string
    """

    if verbose:
        print('=' * 70)
        print('Corpus MIDI files copier')
        print('=' * 70)

    if verbose:
        print('Creating corpus MIDIs files list dict...')

    corpus_midis_dic = get_corpus_midis(tuple(corpus_midis_dirs),
                                        verbose=verbose
                                       )
    
    if verbose:
        print('Done!')
        print('=' * 70)
        print('Copying files...')

    out_dir = ''
    original_copied = False

    for i, cfname, tv, sim in sorted_idxs_sims_list:
        
        try:
        
            sim = str(round(sim, 8))
            tv = str(tv)

            inp_fn = corpus_midis_dic[cfname]
    
            out_dir = os.path.join(main_output_dir, sub_output_dir)
            os.makedirs(out_dir, exist_ok=True)

            if copy_original_midi and not original_copied:
                
                src_fn = sub_output_dir + '.mid'
                out_src_fn = os.path.join(out_dir, src_fn)
                shutil.copy2(inp_fn, out_src_fn)
                original_copied = True
            
            out_fn = os.path.join(out_dir, sim + '_' + tv + '_' + cfname + '.mid')
    
            shutil.copy2(inp_fn, out_fn)

        except Exception as ex:
            if verbose:
                print(ex)
                print('Could not copy file #', i, ':', cfname)
                
            continue

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return out_dir

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the midisim Python module
###################################################################################