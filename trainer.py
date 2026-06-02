import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, matthews_corrcoef, auc
from multimolecule import RnaTokenizer, RnaBertForSequencePrediction
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # For progress bar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score


# Custom imports (make sure utils and models are available in the path)
from utils import seed_everything, get_device, count_parameters, get_optimizer

# DeepCircCode (2019, Bioinformatics)
from models.deepCircCode import DeepCircCode

# circDeep (2020, Bioinformatics)
from models.circDeep import CircDeep 

# CircNet (2020, Neural Computing and Applications)
from models.circNet import CircNet

# JEDI (2021, Bioinformatics, ISMB/ECCB 2021)
from models.jedi import JEDI

# CircCNN (2022, BMC Genomics)
from models.circCNN import CircCNN

# CircDC (2024, BMC Biology)
from models.circDC import CircDC

# CircCNNs (2024, Scientific Reports)
from models.circCNNSingle import CircCNNSingle
from models.circCNNDouble import CircCNNDouble
from models.circCNNDoubleShare import CircCNNDoubleShare
from models.circCNNtri import CircCNNtri
from models.circCNNRCM import CircCNNRCM

# Ours
from models.circATTRCM import CircATTRCM
from models.circCNNATT import CircCNNATT
from models.circRCM import CircRCM
from models.bscan_unified import BSCANUnified, BSCANUnifiedEmbedOnly
from models.circMamba import circMamba
from models.circFusion import CircFusionMambaATTRCM
from models.circAlignMap import CircAlignMap
from models.circUnified import CircUnifiedBlock
from models.circStem import CircStem
from models.circStemV2 import CircStemV2
from models.circBiAlign import CircBiAlign
from models.circMotif import CircMotif
from models.circCombine import CircCombine
from models.circSplice import CircSplice
from models.bscan_v2 import BSCANv2
from models.bscan_seq import BSCANSeq
from models.bscan_seq_lite import BSCANSeqLite
from models.bscan_seq_rcaug import BSCANSeqRCAug
from models.bscan_seq_rcattn import BSCANSeqRCAttn
from models.bscan_seq_mamba_aux import BSCANSeqMambaAux
from models.bscan_plus import BSCANPlus
from models.bscan_mamba_xattn import BSCANMambaXAttn
from models.bscan_region_interact import BSCANRegionInteract
from models.bscan_region_stem import BSCANRegionStem


class Trainer:
    def __init__(
        self,
        seed=315,
        device='cpu',
        dir_save='./saved_models/',
        dir_log='./logs/'
    ):
        self.seed = seed
        self.device = device
        seed_everything(self.seed)  # Ensure reproducibility
        self.loss_fn = nn.CrossEntropyLoss()  # Binary cross-entropy loss
        
        # Directories for saving models and logs
        self.dir_save = dir_save
        os.makedirs(self.dir_save, exist_ok=True)  # Create if doesn't exist
        self.dir_log = dir_log
        os.makedirs(self.dir_log, exist_ok=True)

    def set_pretrained_model(self, name_pretrained):
        if name_pretrained == 'rnabert':
            pretrained = RnaBertForSequencePrediction.from_pretrained('multimolecule/rnabert')
            self.embeddings = pretrained.rnabert.embeddings
            self.encoder = pretrained.rnabert.encoder
            self.pooler = pretrained.rnabert.pooler
            self.embeddings.to(self.device)
            self.encoder.to(self.device)
            self.pooler.to(self.device)
        else:
            print("to be implemented")
        
    def define_model(self, model_name, **kwargs):
        """Define the model architecture."""
        self.model_name = model_name        
        if model_name == 'circattrcm':
            self.model = CircATTRCM(**kwargs).to(self.device)
            self.num_inputs = 3
            
        elif model_name == 'circattrcm_scratch':
            self.model = CircATTRCM(use_pretrained=False, **kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'circcnnatt':
            self.model = CircCNNATT(**kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'bscan_v2':
            # BSCAN-v2: CircCNNATT predictive backbone + explicit intronic WC-stem branch.
            self.model = BSCANv2(**kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'bscan_seq':
            # BSCAN-Seq: two sequence inputs, shared token encoder, internal RC handling.
            self.model = BSCANSeq(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'bscan_seq_lite':
            # BSCAN-Seq-Lite: shared CNN + explicit RC stem profile, no attention.
            self.model = BSCANSeqLite(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_seq_lite_xattn':
            # BSCAN-Seq-Lite + bidirectional cross-attention over upper/lower embeddings.
            self.model = BSCANSeqLite(use_cross_attention=True, **kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_seq_rcaug':
            # BSCAN-RCAug: shared encoder over upper/lower and their reverse complements.
            self.model = BSCANSeqRCAug(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_seq_rcattn':
            # BSCAN-RCAttn: existing CNN/stem backbone + RC-paired cross-attention summaries.
            self.model = BSCANSeqRCAttn(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_seq_mamba_aux':
            # BSCAN-MambaAux: existing CNN/stem backbone + small sequential Mamba summary.
            self.model = BSCANSeqMambaAux(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_plus':
            # BSCAN+: local token encoder + explicit WC/wobble/continuity stem branch.
            self.model = BSCANPlus(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_mamba_xattn':
            # BSCAN-Mamba-XAttn: Mamba encoder + cross-attention + stem profile.
            self.model = BSCANMambaXAttn(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_region_interact':
            # BSCAN-Region: region-token compression + pairwise interaction.
            self.model = BSCANRegionInteract(**kwargs).to(self.device)
            self.num_inputs = 2
        elif model_name == 'bscan_region_stem':
            # BSCAN-RegionStem: one-hot region encoders + WC/GU stem continuity.
            self.model = BSCANRegionStem(**kwargs).to(self.device)
            self.num_inputs = 2
        
        elif model_name == 'deepcirccode':
            self.model = DeepCircCode().to(self.device)
            self.num_inputs = 1
        
        elif model_name == 'circdeep':
            self.model = CircDeep(**kwargs).to(self.device)
            self.num_inputs = 1

        elif model_name == 'circcnn':
            self.model = CircCNN(**kwargs).to(self.device)
            self.num_inputs = 2
            
        elif model_name == 'circcnnsingle':
            self.model = CircCNNSingle().to(self.device)
            self.num_inputs = 1
            
        elif model_name == 'circcnndouble':
            self.model = CircCNNDouble(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcnndoubleshare':
            self.model = CircCNNDoubleShare(**kwargs).to(self.device)
            self.num_inputs = 2
            
        elif model_name == 'circcnntri':
            self.model = CircCNNtri(**kwargs).to(self.device)
            self.num_inputs = 5
            
        elif model_name == 'circcnnrcm':
            self.model = CircCNNRCM(**kwargs).to(self.device)
            self.num_inputs = 3
        
        elif model_name == 'circdc':
            self.model = CircDC(**kwargs).to(self.device)        
            self.num_inputs = 2
        
        elif model_name == 'circmamba':
            self.model = circMamba(**kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'circfusion':
            # Fusion of CircATTRCM + circMamba (logit-level learnable gate)
            self.model = CircFusionMambaATTRCM(**kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'circalignmap':
            # Single-block: cross-attention -> attention map -> 2D CNN classifier
            self.model = CircAlignMap(**kwargs).to(self.device)
            self.num_inputs = 3
            
        elif model_name == 'circalignmap_scratch':
            self.model = CircAlignMap(use_pretrained=False, **kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'circunified':
            # Integrated CNN-Attention-Mamba with shared encoder
            self.model = CircUnifiedBlock(**kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'circunified_scratch':
            self.model = CircUnifiedBlock(use_pretrained=False, **kwargs).to(self.device)
            self.num_inputs = 3

        elif model_name == 'circstem':
            # Two-stage biological model: stem scoring (intron pairing) → gated junction scoring.
            # Accepts double one-hot input (same as circcnn/circdc); RC is computed internally.
            # junction_bps is required to split the sequence at the intron/exon boundary.
            self.model = CircStem(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circstemv2':
            # CircStemV2: full cross-attention (power) + intron stem map (interpretability).
            # Branch A: shared CNN features (upper + lower).
            # Branch B: multi-layer cross-attention on full sequences (upper ↔ lower_rc).
            # Branch C: intron-only stem cross-attention → 2D CNN → gate junction features.
            self.model = CircStemV2(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circbialign':
            # CircBiAlign: bilinear alignment S = u@W@l_rc.T + α*bp_prior.
            # W initialized as identity; α learned scalar; bp_prior from one-hot base-pairing.
            self.model = CircBiAlign(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circmotif':
            # CircMotif: K learnable PWM motif filters + WC stem branch.
            # Shared one-hot filters across 4 sequence regions (intron/exon × upper/lower).
            # Interpretable: filters → PWMs, row/col max → per-position pairing strength.
            self.model = CircMotif(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine':
            # CircCombine: all-in-one modular model (CNN + Motif + Stem + Attention).
            # One-hot input only — no pretrained embedding → fast.
            # All branches configurable for ablation studies.
            self.model = CircCombine(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine_cnn':
            self.model = CircCombine(use_cnn=True,  use_motif=False, use_stem=False, use_attn=False, **kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine_motif':
            self.model = CircCombine(use_cnn=False, use_motif=True,  use_stem=False, use_attn=False, **kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine_stem':
            self.model = CircCombine(use_cnn=False, use_motif=False, use_stem=True,  use_attn=False, **kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine_attn':
            self.model = CircCombine(use_cnn=False, use_motif=False, use_stem=False, use_attn=True,  **kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine_no_motif':
            self.model = CircCombine(use_cnn=True,  use_motif=False, use_stem=True, use_attn=True,  **kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'bscan':
            # BSCAN: Back-Splice CNN-Attention Network (full model)
            # CircCNN-strength local backbone + Stem (WC base-pairing) + Cross-Attention.
            self.model = CircCombine(
                use_cnn=True, use_motif=False, use_stem=True, use_attn=True,
                cnn_style="circcnn", cnn_kernels=(12, 30), **kwargs
            ).to(self.device)
            self.num_inputs = 2

        elif model_name == 'bscan_cnn':
            # BSCAN ablation: CNN branch only
            self.model = CircCombine(
                use_cnn=True, use_motif=False, use_stem=False, use_attn=False,
                cnn_style="circcnn", cnn_kernels=(12, 30), **kwargs
            ).to(self.device)
            self.num_inputs = 2

        elif model_name == 'bscan_stem':
            # BSCAN ablation: CNN + Stem branch
            self.model = CircCombine(
                use_cnn=True, use_motif=False, use_stem=True, use_attn=False,
                cnn_style="circcnn", cnn_kernels=(12, 30), **kwargs
            ).to(self.device)
            self.num_inputs = 2

        elif model_name == 'bscan_attn':
            # BSCAN ablation: CNN + Cross-Attention branch
            self.model = CircCombine(
                use_cnn=True, use_motif=False, use_stem=False, use_attn=True,
                cnn_style="circcnn", cnn_kernels=(12, 30), **kwargs
            ).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine_no_stem':
            self.model = CircCombine(use_cnn=True,  use_motif=True,  use_stem=False, use_attn=True,  **kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'circcombine_no_attn':
            self.model = CircCombine(use_cnn=True,  use_motif=True,  use_stem=True,  use_attn=False, **kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name in ('circsplice', 'circsplice_v2'):
            # Biologically-motivated: Global CNN + Junction CNN + PWM splice detectors
            # v2 adds full lower_exon PWM scan (AATAAA anywhere in exon)
            self.model = CircSplice(**kwargs).to(self.device)
            self.num_inputs = 2

        elif model_name == 'bscan_unified_onehot':
            self.model = BSCANUnified(encoder_type='onehot', **kwargs).to(self.device)
            self.num_inputs = 3
        elif model_name == 'bscan_unified_ernie':
            self.model = BSCANUnified(encoder_type='rnaernie', use_cached=True, **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_unified_bert':
            self.model = BSCANUnified(encoder_type='rnabert', use_cached=True, **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_unified_fm':
            self.model = BSCANUnified(encoder_type='rnafm', use_cached=True, **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_unified_msm':
            self.model = BSCANUnified(encoder_type='rnamsm', use_cached=True, **kwargs).to(self.device)
            self.num_inputs = 6
        # FM + adapter ablation variants
        elif model_name == 'bscan_unified_fm_cnnadapter':
            self.model = BSCANUnified(encoder_type='rnafm', use_cached=True, adapter_type='cnn', adapter_layers=2, **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_unified_fm_mambaadapter':
            self.model = BSCANUnified(encoder_type='rnafm', use_cached=True, adapter_type='mamba', adapter_layers=1, **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_unified_ernie_cnnadapter':
            self.model = BSCANUnified(encoder_type='rnaernie', use_cached=True, adapter_type='cnn', adapter_layers=2, **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_unified_ernie_mambaadapter':
            self.model = BSCANUnified(encoder_type='rnaernie', use_cached=True, adapter_type='mamba', adapter_layers=1, **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_embedonly_ernie':
            self.model = BSCANUnifiedEmbedOnly(encoder_type='rnaernie', **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_embedonly_bert':
            self.model = BSCANUnifiedEmbedOnly(encoder_type='rnabert', **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_embedonly_fm':
            self.model = BSCANUnifiedEmbedOnly(encoder_type='rnafm', **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_embedonly_msm':
            self.model = BSCANUnifiedEmbedOnly(encoder_type='rnamsm', **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_random_bert':
            self.model = BSCANUnifiedEmbedOnly(encoder_type='random120', **kwargs).to(self.device)
            self.num_inputs = 6
        elif model_name == 'bscan_random_msm':
            self.model = BSCANUnifiedEmbedOnly(encoder_type='random768', **kwargs).to(self.device)
            self.num_inputs = 6

        elif model_name == 'jedi':
            self.model = JEDI(**kwargs).to(self.device)
            self.num_inputs = 2
        
        elif model_name == 'circnet':
            self.model = CircNet(**kwargs).to(self.device)
            self.num_inputs = 1

        else:
            raise ValueError('Invalid model name.')
        
        # Memory optimization: freeze pretrained components if the model uses them
        if getattr(self.model, 'use_pretrained', False):
            if getattr(self.model, 'embeddings', None) is not None:
                for param in self.model.embeddings.parameters():
                    param.requires_grad = False
            if getattr(self.model, 'encoder', None) is not None:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
        
        print(f'Model: {model_name}, Number of parameters: {count_parameters(self.model)}')
        
    def set_dataloader(self, dataset, batch_size=32, part=0, shuffle=True):
        """Set up DataLoader for train, validation, or test dataset."""
        if part == 0:
            self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif part == 1:
            self.valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif part == 2:
            self.test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            raise ValueError('Invalid part, must be 0 (train), 1 (valid), or 2 (test).')
        
    def set_optimizer(self, optimizer_name='adam', lr=1e-4, w_decay=None):
        """Set up optimizer for training."""
        assert hasattr(self, 'model'), 'Model is not defined'
        self.optimizer = get_optimizer(self.model.parameters(), optimizer_name, lr=lr, w_decay=w_decay)
        
    def save_model_params(self, **kwargs):
        """Save model parameters to disk."""
        assert hasattr(self, 'model'), 'Model is not defined'
        dir_save = os.path.join(self.dir_save, self.model_name, str(self.seed))
        os.makedirs(dir_save, exist_ok=True)
        with open(os.path.join(dir_save, 'model_params.json'), 'w') as f:
            json.dump(kwargs, f)
            
    def save_model(self):
        """Save the model's state dictionary."""
        dir_save = os.path.join(self.dir_save, self.model_name, str(self.seed))
        os.makedirs(dir_save, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(dir_save, 'model.pth'))
        
    def train(
        self, 
        optimizer="adamw", 
        epochs=100,
        earlystop=20,
        verbose=True,
        progress=True,
    ):
        """Train the model with early stopping."""
        self.verbose = verbose
        self.progress = progress
        optimizer = get_optimizer(self.model.parameters(), optimizer)
        
        best_loss = np.inf
        best_epoch = 0
        
        best_score = 0.0
        patience = 0
        
        if self.verbose:
            print(f'Training {self.model_name} for {epochs} epochs...')
            print(f'Number of inputs: {self.num_inputs}')
            print('-'*80)
            print('| Data | Epoch | Loss | Acc | Macro F1 | Prec(M) | Recall(M) | MCC | AUC | PRC |')
            
        for epoch in range(epochs):
            # Training step
            loss_train, _, _, labels_train, preds_train = self.step_loader(self.train_loader, is_train=True)
            scores = self.get_scores(preds_train, labels_train)
            
            if self.verbose:
                print(f'\r| Train | {epoch+1}/{epochs} | {loss_train:.4f} | {scores[0]:.4f} | {scores[1]:.4f} | '
                      f'{scores[2]:.4f} | {scores[3]:.4f} | {scores[4]:.4f} | {scores[5]:.4f} | {scores[6]:.4f} |')
            
            # Validation step
            with torch.no_grad():
                loss_valid, _, _, labels_valid, preds_valid = self.step_loader(self.valid_loader, is_train=False)
                scores = self.get_scores(preds_valid, labels_valid)
                
                if self.verbose:
                    print(f'\r| Valid | {epoch+1}/{epochs} | {loss_valid:.4f} | {scores[0]:.4f} | {scores[1]:.4f} | '
                          f'{scores[2]:.4f} | {scores[3]:.4f} | {scores[4]:.4f} | {scores[5]:.4f} | {scores[6]:.4f} |', end='')
                
                if scores[0] > best_score:
                    best_score = scores[0]
                    best_epoch = epoch
                    patience = 0
                    self.save_model()  # Save the best model
                    if self.verbose:
                        print('Best!')
                else:
                    patience += 1
                    if self.verbose:
                        print() # New line
            if self.verbose:
                print('-'*100) # Separator

            # Early stopping
            if patience > earlystop:
                if self.verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
            
        # Load the best model's state dict
        best_model_path = os.path.join(self.dir_save, self.model_name, str(self.seed), 'model.pth')
        # Load the best model with weights_only=True
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True), strict=False)
        
        preds_test, labels_test, scores = self.inference(self.test_loader)
        if self.verbose:
            print('| Data | Epoch | Acc | Macro F1 | Prec(M) | Recall(M) | MCC | AUC | PRC |')
            print(f'\r| Test | {best_epoch+1}/{epochs} | {scores[0]:.4f} | {scores[1]:.4f} | {scores[2]:.4f} | {scores[3]:.4f} | {scores[4]:.4f} | {scores[5]:.4f} | {scores[6]:.4f} |')
            print('-'*80)
            
                        
    def step_loader(self, data_loader, is_train=True):
        """Run one step of training or validation."""
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        upper_seqs = []
        lower_seqs = []
        labels = []
        preds = []
        
        for i, data in enumerate(data_loader):
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
            upper_seq, lower_seq, label, pred = self.forward(data)
            
            loss = self.loss_fn(pred, label)
            if is_train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            
            upper_seqs.append(upper_seq.detach().cpu())
            lower_seqs.append(lower_seq.detach().cpu())
            labels.append(label.detach().cpu())
            preds.append(pred.detach().cpu())
            if self.verbose and self.progress:
                print(f'\r[{i+1}/{len(data_loader)}] Loss: {loss.item():.4f}', end='', flush=True)

        upper_seqs = torch.cat(upper_seqs, 0) if upper_seqs else torch.empty(0)
        lower_seqs = torch.cat(lower_seqs, 0) if lower_seqs else torch.empty(0)
        labels = torch.cat(labels, 0) if labels else torch.empty(0)
        preds = torch.cat(preds, 0) if preds else torch.empty(0)

        return total_loss/len(data_loader), upper_seqs, lower_seqs, labels, preds
    
    def forward(self, data):
        upper_seq, lower_seq = data[0].to(self.device), data[1].to(self.device)
        label = data[-1].to(self.device).long()
        
        if self.num_inputs == 1:
            pred = self.model(upper_seq)
        elif self.num_inputs == 2:
            pred = self.model(upper_seq, lower_seq)
        elif self.num_inputs == 3:
            lower_rc_seq = data[2].to(self.device)
            pred = self.model(upper_seq, lower_seq, lower_rc_seq)
        elif self.num_inputs == 5:
            rcm_flanking = data[2].to(self.device)
            rcm_upper = data[3].to(self.device)
            rcm_lower = data[4].to(self.device)
            pred = self.model(upper_seq, lower_seq, rcm_flanking, rcm_upper, rcm_lower)
        elif self.num_inputs == 6:
            # BSCANUnified Cached FM mode
            lower_rc_emb = data[2].to(self.device)
            upper_oh = data[3].to(self.device)
            lower_rc_oh = data[4].to(self.device)
            pred = self.model(upper_seq, lower_seq, lower_rc_emb, upper_oh, lower_rc_oh)
        else:
            assert False, 'Invalid model name'
            
        return upper_seq, lower_seq, label, pred

    def get_scores(self, preds, labels):
        print('\rCalculating metrics...', end='')

        # Flatten to ensure compatibility with sklearn metrics
        preds_probs = torch.softmax(preds, dim=1).cpu().detach().numpy()  # Probabilities
        labels_np = labels.cpu().detach().numpy()
        
        # Convert probabilities to predicted classes for label-based metrics
        preds_class_np = np.argmax(preds_probs, axis=1)

        # Calculate metrics
        acc = accuracy_score(labels_np, preds_class_np) if len(labels_np) > 0 else 0.0
        f1_macro = f1_score(labels_np, preds_class_np, average='macro', zero_division=0)
        precision_macro = precision_score(labels_np, preds_class_np, average='macro', zero_division=0)
        recall_macro = recall_score(labels_np, preds_class_np, average='macro', zero_division=0)
        mcc = matthews_corrcoef(labels_np, preds_class_np) if len(set(labels_np)) > 1 else 0.0

        # Handle AUC/PRC calculation (Binary classification BS vs LS)
        # Use probabilities of the positive class (BS, usually index 1)
        try:
            if len(set(labels_np)) > 1:
                auc_score = roc_auc_score(labels_np, preds_probs[:, 1])
                prc_auc = average_precision_score(labels_np, preds_probs[:, 1])
            else:
                auc_score = 0.5
                prc_auc = 0.0
        except Exception:
            auc_score = 0.0
            prc_auc = 0.0

        return acc, f1_macro, precision_macro, recall_macro, mcc, auc_score, prc_auc

    # def get_scores(self, preds, labels):
    #     """Calculate and return various evaluation metrics."""
    #     preds_np = torch.softmax(preds, dim=1).cpu().detach().numpy()  # Softmax for multiclass
    #     labels_np = labels.cpu().detach().numpy()

    #     # Convert probabilities to predicted classes
    #     preds_np = np.argmax(preds_np, axis=1)

    #     acc = np.mean(preds_np == labels_np)
    #     macro_f1 = f1_score(labels_np, preds_np, average='macro')
    #     micro_f1 = f1_score(labels_np, preds_np, average='micro')
    #     weighted_f1 = f1_score(labels_np, preds_np, average='weighted')
    #     mcc = matthews_corrcoef(labels_np, preds_np)
    #     auc_roc = roc_auc_score(labels_np, preds_np)
        
    #     # Precision-recall curve and its AUC
    #     prc_precision, prc_recall, _ = precision_recall_curve(labels_np, preds_np)
    #     prc_auc = auc(prc_recall, prc_precision)
        
    #     return acc, macro_f1, micro_f1, weighted_f1, mcc, auc_roc, prc_auc

    def inference(self, data_loader, verbose=False):
        """Run inference on the test dataset."""
        self.model.eval()
        with torch.no_grad():
            _, _, _, labels, preds = self.step_loader(data_loader, is_train=False)
            scores = self.get_scores(preds, labels)
        if verbose:
            print(f'\rAccuracy: {scores[0]:.4f}')
            print(f'Macro F1: {scores[1]:.4f}')
            print(f'Micro F1: {scores[2]:.4f}')
            print(f'Weighted F1: {scores[3]:.4f}')
            print(f'MCC: {scores[4]:.4f}')
            print(f'AUC: {scores[5]:.4f}')
            print(f'PRC AUC: {scores[6]:.4f}')
        return preds, labels, scores
