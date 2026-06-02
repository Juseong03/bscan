"""
Dataloader for the circRNA expression regression task.

- Reads circRNA_expression.csv which contains coordinates and avg_reads.
- Extracts sequences based on coordinates from the genome file.
- Pairs sequences with the continuous 'avg_reads' label for regression.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from multimolecule import RnaTokenizer

# Add project root to Python path to find utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

class RegressionDataSetPrep():
    def __init__(
        self,
        coord_path,
        seq_dict_path,
        junction_bps=100,
        seed=42,
        log_transform=True,
    ):
        self.seed = seed
        self.coord_path = coord_path
        self.junction_bps = junction_bps
        self.seq_dict_path = seq_dict_path
        self.seq_dict = None  # Lazy loading
        self.log_transform = log_transform
        self.tokenizer = None

    def load_data(self):
        """Loads the processed circRNA expression data."""
        print(f"Loading coordinates and expression data from {self.coord_path}...")
        df = pd.read_csv(self.coord_path)

        # Apply log transform to reads if specified, as expression data is often skewed
        if self.log_transform:
            df['label'] = np.log1p(df['avg_reads'])
            print(f"Applied log1p transform to 'avg_reads'. New label range: {df['label'].min():.2f} to {df['label'].max():.2f}")
        else:
            df['label'] = df['avg_reads']
        
        # We need a unique key for each entry
        df['key'] = df.apply(lambda row: f"{row['chr']}|{row['start']}|{row['end']}", axis=1)
        
        self.dataframe = df
        print(f"Loaded {len(self.dataframe)} total circRNA samples.")
        return df

    def _ensure_seq_dict_loaded(self) -> None:
        """Load the genome sequence dictionary on-demand."""
        if self.seq_dict is None:
            print("Loading genome sequence dictionary (may take a moment)...")
            with open(self.seq_dict_path, 'r') as f:
                self.seq_dict = json.load(f)
            print("Genome loaded.")

    def reverse_complement(self, seq):
        """Get the reverse complement of a sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join([complement.get(base, 'N') for base in reversed(seq)])

    def get_sequences_for_dataframe(self):
        """
        Extracts sequences for all coordinates in the dataframe.
        This is a simplified version of get_junction_intron_seq from the original dataloader.
        """
        self._ensure_seq_dict_loaded()
        
        sequences = {}
        processed_keys = []
        
        print("Extracting sequences for all coordinates...")
        for _, row in self.dataframe.iterrows():
            key = row['key']
            chrom, start, end = row['chr'], int(row['start']), int(row['end'])
            # We don't have strand information from the processed file, default to '+'
            # This is a limitation we must accept for now.
            strand = '+' 

            dna_seq = self.seq_dict.get(chrom)
            if not dna_seq:
                continue

            # This logic assumes the coordinates from circAtlas define the exons that are joined.
            # upper_intron + upper_exon | lower_exon + lower_intron
            # We'll take flanking regions around the given start and end points.
            upper_seq_area = dna_seq[start - self.junction_bps : start + self.junction_bps]
            lower_seq_area = dna_seq[end - self.junction_bps : end + self.junction_bps]
            
            # --- Add length validation ---
            expected_len = self.junction_bps * 2
            if len(upper_seq_area) != expected_len or len(lower_seq_area) != expected_len:
                continue

            if 'N' in upper_seq_area or 'N' in lower_seq_area:
                continue

            if strand == '-':
                # If we had strand info, we would reverse complement
                # upper_seq_area, lower_seq_area = self.reverse_complement(lower_seq_area), self.reverse_complement(upper_seq_area)
                pass

            sequences[key] = {'upper_seq': upper_seq_area, 'lower_seq': lower_seq_area}
            processed_keys.append(key)

        self.sequences = sequences
        # Filter dataframe to only include keys for which we successfully extracted sequences
        self.dataframe = self.dataframe[self.dataframe['key'].isin(processed_keys)]
        print(f"Successfully extracted sequences for {len(self.dataframe)} samples.")

    def split_data(self, test_size=0.2, validation_size=0.2):
        """Splits the data into training, validation, and test sets."""
        keys = self.dataframe['key'].values
        labels = self.dataframe['label'].values
        
        # First split into training and temp (test)
        train_keys, test_keys, _, _ = train_test_split(
            keys, labels, test_size=test_size, random_state=self.seed
        )
        
        # Split temp (test) into validation and final test
        val_keys, test_keys, _, _ = train_test_split(
            test_keys, test_keys, test_size=0.5, random_state=self.seed
        ) # Adjust test_size to split remaining data
        
        print(f"Data split: Train={len(train_keys)}, Validation={len(val_keys)}, Test={len(test_keys)}")
        return train_keys, val_keys, test_keys

    def seq_to_matrix(self, seq):
        """Convert a sequence to a one-hot encoded matrix."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        matrix = np.zeros((4, len(seq)), dtype=np.float32)
        for i, base in enumerate(seq.upper()):
            if base in mapping:
                matrix[mapping[base], i] = 1
        return matrix

    def create_tensors(self, keys):
        """Creates tensors for a given set of keys."""
        df_subset = self.dataframe[self.dataframe['key'].isin(keys)]
        
        upper_features, lower_features = [], []
        labels = df_subset['label'].values.astype(np.float32)

        for key in df_subset['key']:
            seq_data = self.sequences.get(key)
            if not seq_data: continue

            upper_matrix = self.seq_to_matrix(seq_data['upper_seq'])
            lower_matrix = self.seq_to_matrix(seq_data['lower_seq'])
            
            upper_features.append(torch.from_numpy(upper_matrix))
            lower_features.append(torch.from_numpy(lower_matrix))

        upper_tensor = torch.stack(upper_features)
        lower_tensor = torch.stack(lower_features)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return circRegData(upper_tensor, lower_tensor, labels_tensor)

    def reverse_complement_lower_seq(self, lower_seq):
        return self.reverse_complement(lower_seq)

    def create_tensors_pretrained(self, keys, tokenizer='rnaernie', special_tokens=False):
        """Creates token tensors for pretrained-style three-input models."""
        if self.tokenizer is None:
            if tokenizer.lower() == 'rnaernie':
                self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
                print("Loaded RNA tokenizer (from RNAErnie).")
            elif tokenizer.lower() == 'rnabert':
                self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnabert')
                print("Loaded RNA tokenizer (from RNABert).")
            else:
                raise ValueError(f"Invalid tokenizer: {tokenizer}")

        df_subset = self.dataframe[self.dataframe['key'].isin(keys)]
        upper_seqs, lower_seqs, lower_rc_seqs, labels = [], [], [], []

        for _, row in df_subset.iterrows():
            key = row['key']
            seq_data = self.sequences.get(key)
            if not seq_data:
                continue
            upper = seq_data['upper_seq']
            lower = seq_data['lower_seq']
            upper_seqs.append(upper)
            lower_seqs.append(lower)
            lower_rc_seqs.append(self.reverse_complement_lower_seq(lower))
            labels.append(float(row['label']))

        upper_tensor = self.tokenizer(upper_seqs, padding=True, truncation=True, return_tensors='pt')['input_ids']
        lower_tensor = self.tokenizer(lower_seqs, padding=True, truncation=True, return_tensors='pt')['input_ids']
        lower_rc_tensor = self.tokenizer(lower_rc_seqs, padding=True, truncation=True, return_tensors='pt')['input_ids']

        if not special_tokens:
            upper_tensor = upper_tensor[:, 1:-1]
            lower_tensor = lower_tensor[:, 1:-1]
            lower_rc_tensor = lower_rc_tensor[:, 1:-1]

        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return circRegDataTriple(upper_tensor, lower_tensor, lower_rc_tensor, labels_tensor)

    def create_tensors_pretrained_double(self, keys, tokenizer='rnaernie', special_tokens=False):
        """Creates token tensors for two-input models that compute RC internally."""
        ds = self.create_tensors_pretrained(keys, tokenizer=tokenizer, special_tokens=special_tokens)
        return circRegData(ds.upper, ds.lower, ds.y)

    def create_tensors_single(self, keys):
        """Creates tensors for single-input models by concatenating sequences."""
        df_subset = self.dataframe[self.dataframe['key'].isin(keys)]
        
        features = []
        labels = df_subset['label'].values.astype(np.float32)

        for key in df_subset['key']:
            seq_data = self.sequences.get(key)
            if not seq_data: continue
            
            # Concatenate upper and lower sequences
            full_sequence = seq_data['upper_seq'] + seq_data['lower_seq']
            matrix = self.seq_to_matrix(full_sequence)
            features.append(torch.from_numpy(matrix))

        features_tensor = torch.stack(features)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return circRegDataSingle(features_tensor, labels_tensor)


class circRegData(Dataset):
    """Dataset class for regression task with two sequence inputs."""
    def __init__(self, seqs_upper, seqs_lower, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.y = labels
        self.n_samples = seqs_upper.size(0)

    def __getitem__(self, index):
        return self.upper[index], self.lower[index], self.y[index]

    def __len__(self):
        return self.n_samples

class circRegDataSingle(Dataset):
    """Dataset class for regression task with a single sequence input."""
    def __init__(self, sequences, labels):
        self.x = sequences
        self.y = labels
        self.n_samples = sequences.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class circRegDataTriple(Dataset):
    """Dataset class for regression task with upper/lower/lower_rc token inputs."""
    def __init__(self, seqs_upper, seqs_lower, seqs_lower_rc, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.lower_rc = seqs_lower_rc
        self.y = labels
        self.n_samples = seqs_upper.size(0)

    def __getitem__(self, index):
        return self.upper[index], self.lower[index], self.lower_rc[index], self.y[index]

    def __len__(self):
        return self.n_samples
