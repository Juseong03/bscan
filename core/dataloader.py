from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import product
from multimolecule import RnaTokenizer

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import itertools

class circData_single(Dataset):
    def __init__(self, seqs, labels):
        self.x = seqs
        self.y = labels
        self.n_samples = seqs.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
class circData_double(Dataset):
    def __init__(self, seqs_upper, seqs_lower, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.y = labels
        self.n_samples = seqs_upper.size(0)

    def __getitem__(self, index):
        return self.upper[index], self.lower[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
class circData_rcm(Dataset):
    def __init__(self, seqs_upper, seqs_lower, flanking_rcm, upper_rcm, lower_rcm, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.flanking_rcm = flanking_rcm
        self.upper_rcm = upper_rcm
        self.lower_rcm = lower_rcm
        self.y = labels
        self.n_samples = seqs_upper.size(0)
    
    def __getitem__(self, index):
        return self.upper[index], self.lower[index], self.flanking_rcm[index], self.upper_rcm[index], self.lower_rcm[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    
class circData_triple(Dataset):
    def __init__(self, seqs_upper, seqs_lower, seq_lower_rc, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.y = labels
        self.lower_rc = seq_lower_rc
        self.n_samples = seqs_upper.size(0)
    
    def __getitem__(self, index):
        return self.upper[index], self.lower[index], self.lower_rc[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
class circData_RNABert(Dataset):
    def __init__(self, seqs_upper, seqs_lower, attention_mask, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.attention_mask = attention_mask
        self.y = labels
        self.n_samples = seqs_upper.size(0)

    def __getitem__(self, index):
        return self.upper[index], self.lower[index], self.attention_mask[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
    
class circData(Dataset):
    def __init__(self, seqs_upper, seqs_lower, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.y = labels
        self.n_samples = seqs_upper.size(0)

    def __getitem__(self, index):
        return self.upper[index], self.lower[index], self.y[index]

    def __len__(self):
        return self.n_samples



def fm_cache_dir(fm_name, junction_bps=100):
    """FM embedding cache dir. junction_bps=100 keeps the legacy path (backward
    compatible); other window sizes get a separate dir so windows never collide."""
    base = f"./fm_embeddings/{fm_name}"
    return base if int(junction_bps) == 100 else f"{base}_jb{int(junction_bps)}"


class circData_cached_fm(Dataset):
    def __init__(self, keys, labels, fm_name, upper_oh, lower_rc_oh, junction_bps=100):
        self.keys = keys
        self.y = labels
        self.fm_name = fm_name
        self.cache_dir = fm_cache_dir(fm_name, junction_bps)
        self.upper_oh = upper_oh # [N, 4, L]
        self.lower_rc_oh = lower_rc_oh # [N, 4, L]
        self.n_samples = len(keys)

    def __getitem__(self, index):
        key = self.keys[index]
        label = self.y[index]
        path = os.path.join(self.cache_dir, f"{key.replace('|', '_')}.pt")
        # Load pre-computed hidden states
        data = torch.load(path, weights_only=True)
        # Returns: upper_emb, lower_emb, lower_rc_emb, upper_oh_intron, lower_rc_oh_intron, label
        return data['upper'], data['lower'], data['lower_rc'], self.upper_oh[index], self.lower_rc_oh[index], label

    def __len__(self):
        return self.n_samples


class circData_triple_oh(Dataset):
    def __init__(self, seqs_upper, seqs_lower, seq_lower_rc, upper_oh, lower_rc_oh, labels):
        self.upper = seqs_upper
        self.lower = seqs_lower
        self.lower_rc = seq_lower_rc
        self.upper_oh = upper_oh
        self.lower_rc_oh = lower_rc_oh
        self.y = labels
        self.n_samples = seqs_upper.size(0)

    def __getitem__(self, index):
        return (
            self.upper[index],
            self.lower[index],
            self.lower_rc[index],
            self.upper_oh[index],
            self.lower_rc_oh[index],
            self.y[index],
        )

    def __len__(self):
        return self.n_samples


class DataSetPrep():
    def __init__(
        self,
        coord_path,
        seq_dict_path,
        junction_bps=100,
        flanking_bps=100,
        seed=315,
        use_full_intron=False
    ):
        self.seed = seed
        self.coord_path = coord_path
        self.junction_bps = junction_bps
        self.flanking_bps = flanking_bps
        self.use_full_intron = use_full_intron
        # NOTE: Genome sequence dict JSONs in this repo can be multi-GB (hg19/hg38).
        # Most experiment runs *don't* need to load them because they use precomputed
        # junction/flanking sequences in `data/seq_dict/*`.
        #
        # To avoid unnecessary memory/time, lazily load the genome dictionary only
        # when `get_junction_intron_seq()` is called.
        self.seq_dict_path = seq_dict_path
        self.seq_dict = None
        self.tokenizer=None
        self.index_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def load_json(self, path):
        """Load JSON file"""
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading JSON file: {path}")
            return {}

    def _ensure_seq_dict_loaded(self) -> None:
        """Load the genome sequence dictionary on-demand."""
        if self.seq_dict is None:
            self.seq_dict = self.load_json(self.seq_dict_path)

    def save_json(self, data, path):
        """Save data to JSON file"""
        with open(path, 'w') as f:
            json.dump(data, f)
            
    def save_junction_flanking_seq(self, name=False):
        """Save junction and flanking sequences to JSON files"""
        if not name:
            name = str(self.junction_bps)
        
        dir_seq = f'./data/seq_dict/{name}/'
        os.makedirs(dir_seq, exist_ok=True)
        
        self.save_json(self.junction_seq, f'{dir_seq}/junction.json')
        print(f"Saved junction sequences to {dir_seq}/")
        self.save_json(self.flanking_seq, f'{dir_seq}/flanking_{self.flanking_bps}.json')
        print(f"Saved flanking sequences ({self.flanking_bps}bps) to {dir_seq}/")
        
    def load_junction_flanking_seq(self, name=False):
        """Load junction and flanking sequences from JSON files"""
        if not name:
            name = str(self.junction_bps)
        
        dir_seq = f'./data/seq_dict/{name}/'
        assert os.path.exists(dir_seq), f"Directory {dir_seq} does not exist."
    
        junction_path = f'{dir_seq}/junction.json'
        flanking_path = f'{dir_seq}/flanking_{self.flanking_bps}.json'
        
        self.junction_seq = self.load_json(junction_path)
        self.flanking_seq = self.load_json(flanking_path)
        
        return self.junction_seq, self.flanking_seq

    def reverse_complement(self, seq):
        """Get the reverse complement of a sequence"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join([complement[base] for base in reversed(seq)])

    def extract_kmers(self, seq, k):
        """Extract k-mers from a given sequence"""
        kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
        return kmers

    def get_junction_intron_seq(self):
        """Extract BS/LS junction and flanking intron sequences"""
        self._ensure_seq_dict_loaded()

        junction_seq = {}
        flanking_seq = {}

        df = pd.read_csv(self.coord_path, sep='\t')
        for _, row in df.iterrows():
            chrom, strand, start, end, label = row['chr'], row['strand'], int(row['start']), int(row['end']), row['Splicing_type']
            key = f'{chrom}|{start}|{end}|{strand}'
            dna_seq = self.seq_dict.get(chrom, "")

            if not dna_seq:
                continue

            spliced_seq = dna_seq[start:end].upper()

            if self.use_full_intron:
                # Use full intron sequences - find intron boundaries
                # This is a simplified approach; in reality, you'd need gene annotation data
                # For now, use a large flanking region (e.g., 2000bp) as approximation
                intron_length = min(2000, start, len(dna_seq) - end)  # Max 2000bp
            else:
                # Use junction_bps as before
                intron_length = self.junction_bps

            upper_intron_pos = dna_seq[max(0, start - intron_length):start].upper()
            lower_intron_pos = dna_seq[end:min(len(dna_seq), end + intron_length)].upper()

            if 'N' in upper_intron_pos or 'N' in lower_intron_pos:
                print(f'{key} contains "N" in junction sequences, skipping.')
                continue

            upper_flanking_pos = dna_seq[start - self.flanking_bps:start].upper()
            lower_flanking_pos = dna_seq[end:end + self.flanking_bps].upper()

            if strand == '-':
                spliced_seq = self.reverse_complement(spliced_seq)    
                upper_intron = self.reverse_complement(lower_intron_pos)
                lower_intron = self.reverse_complement(upper_intron_pos)
                upper_flanking = self.reverse_complement(lower_flanking_pos)
                lower_flanking = self.reverse_complement(upper_flanking_pos)
            else:
                upper_intron = upper_intron_pos
                lower_intron = lower_intron_pos
                upper_flanking = upper_flanking_pos
                lower_flanking = lower_flanking_pos

            upper_exon = spliced_seq[:self.junction_bps]
            lower_exon = spliced_seq[-self.junction_bps:]

            # --- Add length validation ---
            if any(len(s) != self.junction_bps for s in [upper_intron, lower_intron, upper_exon, lower_exon]):
                # print(f"Skipping {key} due to inconsistent sequence length.")
                continue
            
            lower_seq_rc = self.reverse_complement(lower_exon + lower_intron)
            junction_seq[key] = {
                'spliced_seq': spliced_seq,
                'upper_intron': upper_intron,
                'lower_intron': lower_intron,
                'upper_exon': upper_exon,
                'lower_exon': lower_exon,
                'label': label,
                'junction_bps': self.junction_bps,
                'upper_seq': upper_intron + upper_exon,
                'lower_seq': lower_exon + lower_intron,
                'lower_seq_rc': lower_seq_rc,
            }

            lower_flanking_rc = self.reverse_complement(lower_flanking)
            flanking_seq[key] = {
                'upper_flanking': upper_flanking,
                'lower_flanking': lower_flanking,
                'label': label,
                'flanking_bps': self.flanking_bps,
                'lower_flanking_rc': lower_flanking_rc,
            }
        
        junction_seq_BS = {}
        junction_seq_LS = {}
        
        for key, value in junction_seq.items():
            if value['label'] == 'BS':
                junction_seq_BS[key] = (
                    tuple(value['upper_intron']), 
                    tuple(value['upper_exon']), 
                    tuple(value['lower_exon']), 
                    tuple(value['lower_intron'])
                )
            else:
                junction_seq_LS[key] = (
                    tuple(value['upper_intron']), 
                    tuple(value['upper_exon']), 
                    tuple(value['lower_exon']), 
                    tuple(value['lower_intron'])
                )
                                        
        overlap_junction_seq = set(junction_seq_BS.values()).intersection(set(junction_seq_LS.values()))
        print(f"Overlap junction sequences: {len(overlap_junction_seq)}")
        
        junction_seq_BS_wo_overlap = {key: value for key, value in junction_seq_BS.items() if value not in overlap_junction_seq}
        junction_seq_LS_wo_overlap = {key: value for key, value in junction_seq_LS.items() if value not in overlap_junction_seq}
        
        num_repeated_seq_BS = len(junction_seq_BS_wo_overlap.values()) - len(set(junction_seq_BS_wo_overlap.values()))
        num_repeated_seq_LS = len(junction_seq_LS_wo_overlap.values()) - len(set(junction_seq_LS_wo_overlap.values()))
        print(f"Number of repeated junction sequences in Back-Splicing: {num_repeated_seq_BS}")
        print(f"Number of repeated junction sequences in Linear-Splicing: {num_repeated_seq_LS}")
        
        BS_temp = {value: key for key, value in junction_seq_BS_wo_overlap.items()}
        BS_res_dict = {value: key for key, value in BS_temp.items()}
        
        LS_temp = {value: key for key, value in junction_seq_LS_wo_overlap.items()}
        LS_res_dict = {value: key for key, value in LS_temp.items()}
        
        BS_LS_res_dict = {**BS_res_dict, **LS_res_dict}
        
        junction_seq_final = {key: junction_seq[key] for key in BS_LS_res_dict.keys()}
        flanking_seq_final = {key: flanking_seq[key] for key in BS_LS_res_dict.keys()}
        
        self.junction_seq = junction_seq_final
        self.flanking_seq = flanking_seq_final
        
        return junction_seq_final, flanking_seq_final

    def seq_to_matrix(self, seq):
        """Convert a sequence to a one-hot encoded matrix"""
        mapping = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
        matrix = np.zeros((4, len(seq)))

        for i, base in enumerate(seq):
            if base in mapping:
                matrix[mapping[base], i] = 1
        return matrix

    def seq_to_tensor(self, keys, is_concat=False):
        """Convert sequences to PyTorch tensors"""
        if is_concat:
            features = []
        else:
            upper_features, lower_features = [], []

        labels = []
        for key in keys:
            flanking_seq = self.flanking_seq[key]

            value = self.junction_seq[key]
            label = value['label']
            assert label == flanking_seq['label'], f"Same sequence key {key} has different labels"

            labels.append(1 if label == 'BS' else 0)

            expected_len = 2 * self.junction_bps
            if is_concat:
                matrix_concat = self.seq_to_matrix(value['upper_seq'] + value['lower_seq'])
                t = torch.from_numpy(matrix_concat).to(torch.float32)
                if t.shape[1] < 2 * expected_len:
                    t = F.pad(t, (0, 2 * expected_len - t.shape[1]))
                features.append(t)
            else:
                upper_matrix = self.seq_to_matrix(value['upper_seq'])
                lower_matrix = self.seq_to_matrix(value['lower_seq'])

                individual_upper = torch.from_numpy(upper_matrix).to(torch.float32)
                individual_lower = torch.from_numpy(lower_matrix).to(torch.float32)

                if individual_upper.shape[1] < expected_len:
                    individual_upper = F.pad(individual_upper, (0, expected_len - individual_upper.shape[1]))
                if individual_lower.shape[1] < expected_len:
                    individual_lower = F.pad(individual_lower, (0, expected_len - individual_lower.shape[1]))

                upper_features.append(individual_upper)
                lower_features.append(individual_lower)
                
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        if is_concat:
            upper_tensor = torch.stack(features)
            lower_tensor = torch.stack(features)
        else:
            upper_tensor = torch.stack(upper_features)
            lower_tensor = torch.stack(lower_features)
            
        return upper_tensor, lower_tensor, labels_tensor
        
        
    def seq_to_tensor_w_rcm(self, keys, is_concat=False, rcm_folder='./rcm_scores', flanking_list=None, kmer_list=[5, 7, 9, 11, 13]):
        if is_concat:
            features = []
        else:
            upper_features, lower_features = [], []
        
        labels = []
        
        if flanking_list is None:
            flanking_list = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000]
        
        if kmer_list is None:
            kmer_list = [5, 7, 9, 11, 13]

        flanking_rcm_scores, upper_rcm_scores, lower_rcm_scores = [], [], []
        flanking_rcm_list, upper_rcm_list, lower_rcm_list = [], [], []
        
        for flanking in flanking_list:
            for k in kmer_list:
                with open(f'{rcm_folder}/flanking_{flanking}_bps_{k}mer_scores.json') as f:
                    flanking_rcm_list.append(json.load(f))
                with open(f'{rcm_folder}/upper_{flanking}_bps_{k}mer_scores.json') as f:
                    upper_rcm_list.append(json.load(f))
                with open(f'{rcm_folder}/lower_{flanking}_bps_{k}mer_scores.json') as f:
                    lower_rcm_list.append(json.load(f))

        for key in keys:
            flanking_seq = self.flanking_seq[key]
        
            value = self.junction_seq[key]
            label = value['label']
            assert label == flanking_seq['label'], f"Same sequence key {key} has different labels"
            
            labels.append(1 if label == 'BS' else 0)
                
            if is_concat:
                matrix_concat = self.seq_to_matrix(value['upper_seq'] + value['lower_seq'])
                individual_concat = torch.from_numpy(matrix_concat).to(torch.float32)
                features.append(individual_concat)    
            else:
                upper_matrix = self.seq_to_matrix(value['upper_seq'])
                lower_matrix = self.seq_to_matrix(value['lower_seq'])
                
                individual_upper = torch.from_numpy(upper_matrix).to(torch.float32)
                individual_lower = torch.from_numpy(lower_matrix).to(torch.float32)
                
                upper_features.append(individual_upper)
                lower_features.append(individual_lower)
            
            flanking_rcm_kmer_list = [np.log(np.array(flanking_rcm[key]).reshape(5, 5)+1) for flanking_rcm in flanking_rcm_list]
            flanking_rcm_kmers = torch.from_numpy(np.concatenate(flanking_rcm_kmer_list, axis=1)).to(torch.float32)
            
            upper_rcm_kmer_list = [np.log(np.array(upper_rcm[key]).reshape(5, 5)+1) for upper_rcm in upper_rcm_list]
            upper_rcm_kmers = torch.from_numpy(np.concatenate(upper_rcm_kmer_list, axis=1)).to(torch.float32)
            
            lower_rcm_kmer_list = [np.log(np.array(lower_rcm[key]).reshape(5, 5)+1) for lower_rcm in lower_rcm_list]
            lower_rcm_kmers = torch.from_numpy(np.concatenate(lower_rcm_kmer_list, axis=1)).to(torch.float32)
        
            flanking_rcm_scores.append(flanking_rcm_kmers)
            upper_rcm_scores.append(upper_rcm_kmers)
            lower_rcm_scores.append(lower_rcm_kmers)
            
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        if is_concat:
            upper_tensor = torch.stack(features)
            lower_tensor = torch.stack(features)
        else:
            upper_tensor = torch.stack(upper_features)
            lower_tensor = torch.stack(lower_features)
        
        flanking_rcm_tensor = torch.stack(flanking_rcm_scores, dim=0)
        upper_rcm_tensor = torch.stack(upper_rcm_scores, dim=0)
        lower_rcm_tensor = torch.stack(lower_rcm_scores, dim=0)

        return upper_tensor, lower_tensor, flanking_rcm_tensor, upper_rcm_tensor, lower_rcm_tensor, labels_tensor

    def seq_matrix(self, keys, nucleotide=True, rc=False):
        seq_features, upper_features, lower_features, lower_rc_features = [], [], [], []
        labels = []
        for key in keys:
            value = self.junction_seq[key]
            labels.append(1 if value['label'] == 'BS' else 0)
            seq = value['upper_seq'] + value['lower_seq']
            upper_seq = value['upper_seq']
            lower_seq = value['lower_seq']
            lower_seq_rc = value['lower_seq_rc']
    
            if nucleotide:
                seq_features.append(np.array(list(seq)))
                upper_features.append(np.array(list(upper_seq)))
                lower_features.append(np.array(list(lower_seq)))
                lower_rc_features.append(np.array(list(lower_seq_rc)))
            else:
                seq_features.append(''.join(seq))
                upper_features.append(''.join(upper_seq))
                lower_features.append(''.join(lower_seq))
                lower_rc_features.append(''.join(lower_seq_rc))
    
        seq_matrix = np.array(seq_features)
        upper_matrix = np.array(upper_features)
        lower_matrix = np.array(lower_features)
        if rc:
            lower_rc_matrix = np.array(lower_rc_features)
        else:
            lower_rc_matrix = None
    
        return seq_matrix, upper_matrix, lower_matrix, labels, lower_rc_matrix

    def matrix_to_index(self, matrix, k=1):
        """Convert a sequence matrix to an indexed k-mer representation."""
        self.index_mapping = {''.join(kmer): idx for idx, kmer in enumerate(product(self.index_mapping.keys(), repeat=k))}

        indexed_matrix = []
        
        for seq in matrix:
            # Ensure k-mers are strings (by using ''.join())
            kmers = self.extract_kmers(''.join(seq), k)
            indexed_kmers = [self.index_mapping[kmer] for kmer in kmers if kmer in self.index_mapping]
            indexed_matrix.append(indexed_kmers)
        
        return np.array(indexed_matrix)

    def split_data(self, ratio=[0.6, 0.2, 0.2]):
        
        all_keys = list(self.junction_seq.keys())
        all_labels = [1 if self.junction_seq[key]['label'] == 'BS' else 0 for key in all_keys]  # 1 for BS, 0 for LS

        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=ratio[2], random_state=42)
        
        for train_valid_idx, test_idx in strat_split.split(all_keys, all_labels):
            train_valid_keys = [all_keys[i] for i in train_valid_idx]
            test_keys = [all_keys[i] for i in test_idx]
            train_valid_labels = [all_labels[i] for i in train_valid_idx]
        
        strat_split_valid = StratifiedShuffleSplit(n_splits=1, test_size=ratio[1] / (ratio[0] + ratio[1]), random_state=42)
        
        for train_idx, valid_idx in strat_split_valid.split(train_valid_keys, train_valid_labels):
            train_keys = [train_valid_keys[i] for i in train_idx]
            valid_keys = [train_valid_keys[i] for i in valid_idx]

        return train_keys, valid_keys, test_keys

    def split_data_grouped(self, ratio=[0.6, 0.2, 0.2], group_by="transcript"):
        """Split data while keeping all samples from the same group in one partition."""
        if group_by not in {"transcript", "chromosome"}:
            raise ValueError(f"Unsupported group_by={group_by!r}; supported values are 'transcript' and 'chromosome'.")

        coord = pd.read_csv(self.coord_path, sep="\t")
        coord["key"] = coord.apply(lambda r: f"{r['chr']}|{int(r['start'])}|{int(r['end'])}|{r['strand']}", axis=1)
        coord["transcript_base"] = coord["Transcript"].astype(str).str.replace(r"\.\d+$", "", regex=True)
        transcript_by_key = dict(zip(coord["key"], coord["transcript_base"]))
        chr_by_key = dict(zip(coord["key"], coord["chr"].astype(str)))

        all_keys = list(self.junction_seq.keys())
        all_labels = [1 if self.junction_seq[key]["label"] == "BS" else 0 for key in all_keys]
        if group_by == "transcript":
            all_groups = [transcript_by_key.get(key, key) for key in all_keys]
        else:
            all_groups = [chr_by_key.get(key, key) for key in all_keys]

        try:
            from sklearn.model_selection import StratifiedGroupKFold

            test_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed)
            train_valid_idx, test_idx = next(test_split.split(all_keys, all_labels, all_groups))

            train_valid_keys = [all_keys[i] for i in train_valid_idx]
            train_valid_labels = [all_labels[i] for i in train_valid_idx]
            train_valid_groups = [all_groups[i] for i in train_valid_idx]
            test_keys = [all_keys[i] for i in test_idx]

            valid_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=self.seed)
            train_idx, valid_idx = next(valid_split.split(train_valid_keys, train_valid_labels, train_valid_groups))
            train_keys = [train_valid_keys[i] for i in train_idx]
            valid_keys = [train_valid_keys[i] for i in valid_idx]
        except Exception:
            from sklearn.model_selection import GroupShuffleSplit

            test_split = GroupShuffleSplit(n_splits=1, test_size=ratio[2], random_state=self.seed)
            train_valid_idx, test_idx = next(test_split.split(all_keys, all_labels, all_groups))

            train_valid_keys = [all_keys[i] for i in train_valid_idx]
            train_valid_labels = [all_labels[i] for i in train_valid_idx]
            train_valid_groups = [all_groups[i] for i in train_valid_idx]
            test_keys = [all_keys[i] for i in test_idx]

            valid_split = GroupShuffleSplit(n_splits=1, test_size=ratio[1] / (ratio[0] + ratio[1]), random_state=self.seed)
            train_idx, valid_idx = next(valid_split.split(train_valid_keys, train_valid_labels, train_valid_groups))
            train_keys = [train_valid_keys[i] for i in train_idx]
            valid_keys = [train_valid_keys[i] for i in valid_idx]

        return train_keys, valid_keys, test_keys
    
    def tensor_for_pretrained(self, keys, special_tokens=True, attention_mask=False, rc=False, tokenizer='RNABert'):

        if self.tokenizer is not None:
            pass
        elif tokenizer.lower() == 'rnabert':
            self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnabert')
            print("Loaded RNA tokenizer (from RNABert).")
        elif tokenizer.lower() == 'rnaernie':
            self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnaernie')
            print("Loaded RNA tokenizer (from RNAErnie).")
        elif tokenizer.lower() == 'rnafm':
            self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnafm')
            print("Loaded RNA tokenizer (from RNA-FM).")
        elif tokenizer.lower() == 'rnamsm':
            self.tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnamsm')
            print("Loaded RNA tokenizer (from RNA-MSM).")
        else:
            raise ValueError(f"Invalid tokenizer: {tokenizer}")

        seqs, seqs_upper, seqs_lower, labels, seqs_lower_rc = self.seq_matrix(keys, nucleotide=False, rc=rc)
        upper_tensor = self.tokenizer(list(seqs_upper), padding=True, truncation=True, return_tensors='pt')
        lower_tensor = self.tokenizer(list(seqs_lower), padding=True, truncation=True, return_tensors='pt')
        lower_rc_tensor = self.tokenizer(list(seqs_lower_rc), padding=True, truncation=True, return_tensors='pt') if rc else None
        
        if special_tokens:
            upper_seq_tensor = upper_tensor['input_ids']
            lower_seq_tensor = lower_tensor['input_ids']
            lower_rc_tensor = lower_rc_tensor['input_ids'] if rc else None
        else:
            upper_seq_tensor = upper_tensor['input_ids'][:, 1:-1]
            lower_seq_tensor = lower_tensor['input_ids'][:, 1:-1]
            lower_rc_tensor = lower_rc_tensor['input_ids'][:, 1:-1] if rc else None

        if attention_mask:
            return upper_seq_tensor, lower_seq_tensor, lower_rc_tensor, torch.tensor(labels, dtype=torch.float32), upper_tensor['attention_mask'], lower_tensor['attention_mask']
        else:
            return upper_seq_tensor, lower_seq_tensor, lower_rc_tensor, torch.tensor(labels, dtype=torch.float32)

    def seq_to_index(self, keys, rc=False, kmer=1):
        """
        Convert sequences to indexed format with the option to use k-mers.
        :param keys: List of sequence keys
        :param rc: Boolean flag to include reverse complement (default: False)
        :param kmer: Length of k-mers (default: 1 for single nucleotide encoding)
        :return: Indexed upper and lower sequences and labels as tensors
        """
        seqs, seqs_upper, seqs_lower, labels, seqs_lower_rc = self.seq_matrix(keys, nucleotide=True, rc=rc)

        # Define possible nucleotides
        nucleotides = ['A', 'C', 'G', 'T']

        # Dynamically generate the index_mapping based on k-mer size
        kmer_combinations = [''.join(kmer_tuple) for kmer_tuple in itertools.product(nucleotides, repeat=kmer)]
        index_mapping = {kmer: idx for idx, kmer in enumerate(kmer_combinations)}

        # Function to extract k-mers from a sequence
        def seq_to_kmers(seq, k):
            return [''.join(seq[i:i+k]) for i in range(len(seq) - k + 1)]

        if kmer > 1:
            # If kmer > 1, convert sequences into k-mers
            seqs = [seq_to_kmers(seq, kmer) for seq in seqs]
            seqs_upper = [seq_to_kmers(seq, kmer) for seq in seqs_upper]
            seqs_lower = [seq_to_kmers(seq, kmer) for seq in seqs_lower]
            seqs_lower_rc = [seq_to_kmers(seq, kmer) for seq in seqs_lower_rc] if rc else None

        # Convert sequences to indexed format using list comprehension and the dynamically generated index_mapping
        def index_sequence(sequence_list):
            indexed_sequences = []
            for sequence in sequence_list:
                indexed_sequence = [index_mapping.get(kmer, 0) for kmer in sequence]  # Use 0 for unknown k-mers
                indexed_sequences.append(indexed_sequence)
            return np.array(indexed_sequences)

        # Convert upper, lower, and reverse complement sequences
        indexed_seqs = index_sequence(seqs)
        indexed_upper = index_sequence(seqs_upper)
        indexed_lower = index_sequence(seqs_lower)
        indexed_lower_rc = index_sequence(seqs_lower_rc) if rc else None

        # Convert lists to tensors for PyTorch
        indexed_seqs_tensor = torch.tensor(indexed_seqs, dtype=torch.long)
        indexed_upper_tensor = torch.tensor(indexed_upper, dtype=torch.long)
        indexed_lower_tensor = torch.tensor(indexed_lower, dtype=torch.long)
        indexed_lower_rc_tensor = torch.tensor(indexed_lower_rc, dtype=torch.long) if rc else None

        # Use list comprehension for labels (1 for 'BS', 0 for 'LS')
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return indexed_seqs_tensor, indexed_upper_tensor, indexed_lower_tensor, indexed_lower_rc_tensor, labels_tensor
