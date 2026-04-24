# Unsupervised Neural Network for Multi-Genre Music Generation
Course: CSE425/EEE474 Neural Networks 

#Project Overview

Four progressive unsupervised generative models for multi-genre MIDI music generation:
| Task 1 | LSTM Autoencoder | Easy | Single Genre |
| Task 2 | Variational Autoencoder (VAE) | Medium | Multi-Genre |
| Task 3 | Transformer Decoder | Hard | Long Coherent Sequences |
| Task 4 | RLHF-Tuned Generator | Advanced | Human Preference Optimized |

# Project Structure
music-generation-unsupervised/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw_midi/              
|   |-- processed/            
|   +-- train_test_split/      
|-- notebooks/
|   |-- preprocessing.ipynb    
|   +-- baseline_markov.ipynb 
|-- src/
|   |-- config.py             
|   |-- preprocessing/
|   |   |-- midi_parser.py     
|   |   |-- tokenizer.py       
|   |   +-- piano_roll.py      
|   |-- models/
|   |   |-- autoencoder.py     
|   |   |-- vae.py            
|   |   +-- transformer.py     
|   |-- training/
|   |   |-- train_ae.py        
|   |   |-- train_vae.py       
|   |   |-- train_transformer.py 
|   |   +-- train_rlhf.py     
|   |-- evaluation/
|   |   |-- metrics.py        
|   |   |-- pitch_histogram.py 
|   |   +-- rhythm_score.py    
|   +-- generation/
|       |-- sample_latent.py  
|       |-- generate_music.py  
|       +-- midi_export.py    
+-- outputs/
    |-- generated_midis/      
    |-- plots/                 
    +-- survey_results/        
# Datasets
| MAESTRO v3 | Classical Piano | https://magenta.tensorflow.org/datasets/maestro |
| Lakh MIDI  | Multi-Genre     | https://colinraffel.com/projects/lmd/ |
| Groove MIDI | Jazz/Drums     | https://magenta.tensorflow.org/datasets/groove |

# Evaluation Metrics
| Pitch Histogram Similarity | H(p,q) = sum|pi - qi| | [0,2] lower=better |
| Rhythm Diversity | unique_durations / total_notes | [0,1] higher=better |
| Repetition Ratio | repeated_patterns / total_patterns | [0,1] lower=better |
| Human Listening Score | Survey mean 1-5 | [1,5] higher=better |

# Baseline Comparison

| Model | Perplexity | Rhythm Div. | Human Score | Genre Ctrl |
| Random Generator | -- | Low | 1.1 | None |
| Markov Chain | -- | Medium | 2.3 | Weak |
| Task 1: LSTM AE | -- | Medium | ~3.1 | Single |
| Task 2: VAE | -- | High | ~3.8 | Moderate |
| Task 3: Transformer | ~12.5 | Very High | ~4.4 | Strong |
| Task 4: RLHF | ~11.2 | Very High | ~4.8 | Strongest |
