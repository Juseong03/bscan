import sys as _sys, os as _os  # path shim (core/ + pipeline/ layout)
_ROOT = "/workspace/volume/bscan"
for _p in (_ROOT, _ROOT + "/core", _ROOT + "/pipeline"):
    if _p not in _sys.path: _sys.path.insert(0, _p)

import argparse
from utils import seed_everything, get_device
from trainer import Trainer
from dataloader import DataSetPrep, circData_single, circData_double, circData_rcm, circData_triple, circData_cached_fm, circData_triple_oh
import gc
import torch
import random
        
def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def experiment(args_dict: dict)->None:
    """
    Run the training process based on the provided arguments.
    """
    seed_everything(args_dict['seed'])
    device = get_device(args_dict['device'])
    print(f'Using device: {device}')
    
    data = DataSetPrep(
        coord_path='./data/BS_LS_coordinates_final.csv',
        seq_dict_path='./data/hg19_seq_dict.json',
        junction_bps=args_dict['junction_bps'],
        flanking_bps=args_dict['flanking_bps'],
        seed=args_dict['seed'],
        use_full_intron=args_dict.get('use_full_intron', False)
    )    
    
    try: 
        print('try to load the data...')
        junction, intronic = data.load_junction_flanking_seq()
        print('success the loading the data!')
    except:
        print('failed to load the data, process the data')
        junction, intronic = data.get_junction_intron_seq()
        data.save_junction_flanking_seq()
        
    split_strategy = args_dict.get('split_strategy', 'sample')
    if split_strategy == 'sample':
        keys_train, keys_valid, keys_test = data.split_data()
    elif split_strategy == 'transcript':
        keys_train, keys_valid, keys_test = data.split_data_grouped(group_by='transcript')
    elif split_strategy == 'chromosome':
        keys_train, keys_valid, keys_test = data.split_data_grouped(group_by='chromosome')
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")
    print(
        f"Split strategy: {split_strategy} "
        f"(train={len(keys_train)}, valid={len(keys_valid)}, test={len(keys_test)})"
    )

    max_samples = args_dict.get("max_samples", None)
    if max_samples is not None:
        rng = random.Random(args_dict["seed"])
        keys_train = list(keys_train)
        keys_valid = list(keys_valid)
        keys_test = list(keys_test)
        if len(keys_train) > max_samples:
            keys_train = rng.sample(keys_train, max_samples)
        if len(keys_valid) > max_samples:
            keys_valid = rng.sample(keys_valid, max_samples)
        if len(keys_test) > max_samples:
            keys_test = rng.sample(keys_test, max_samples)
        print(f"Subsampled splits to max_samples={max_samples}: train={len(keys_train)}, valid={len(keys_valid)}, test={len(keys_test)}")
    
    if args_dict['model_name'] in ['deepcirccode', 'circcnnsingle', 'circnet']:
        seq_tensor_train, _, label_tensor_train = data.seq_to_tensor(keys_train, is_concat=True)
        seq_tensor_valid, _, label_tensor_valid = data.seq_to_tensor(keys_valid, is_concat=True)
        seq_tensor_test, _, label_tensor_test = data.seq_to_tensor(keys_test, is_concat=True)
        train_dataset = circData_single(seq_tensor_train, label_tensor_train)
        valid_dataset = circData_single(seq_tensor_valid, label_tensor_valid)
        test_dataset = circData_single(seq_tensor_test, label_tensor_test)
        
    elif args_dict['model_name'] in ['circcnn', 'circcnndouble', 'circcnndoubleshare', 'circdc', 'circstem', 'circstemv2', 'circbialign', 'circmotif',
                                      'circcombine', 'circcombine_cnn', 'circcombine_motif', 'circcombine_stem', 'circcombine_attn',
                                      'circcombine_no_motif', 'circcombine_no_stem', 'circcombine_no_attn',
                                      'circsplice', 'circsplice_v2',
                                      'bscan', 'bscan_cnn', 'bscan_stem', 'bscan_attn',
                                      'bscan_region_stem']:
        upper_tensor_train, lower_tensor_train, label_tensor_train = data.seq_to_tensor(keys_train)
        upper_tensor_valid, lower_tensor_valid, label_tensor_valid = data.seq_to_tensor(keys_valid)
        upper_tensor_test, lower_tensor_test, label_tensor_test = data.seq_to_tensor(keys_test)
        train_dataset = circData_double(upper_tensor_train, lower_tensor_train, label_tensor_train)
        valid_dataset = circData_double(upper_tensor_valid, lower_tensor_valid, label_tensor_valid)
        test_dataset = circData_double(upper_tensor_test, lower_tensor_test, label_tensor_test)
    
    elif args_dict['model_name'] in [
        'bscan_unified_ernie', 'bscan_unified_bert', 'bscan_unified_fm', 'bscan_unified_msm',
        'bscan_unified_fm_cnnadapter', 'bscan_unified_fm_mambaadapter',
        'bscan_unified_ernie_cnnadapter', 'bscan_unified_ernie_mambaadapter',
        # Branch ablation variants
        'bscan_unified_fm_fulltr', 'bscan_unified_fm_mlponly', 'bscan_unified_fm_nocnn', 'bscan_unified_fm_nostem',
        'bscan_unified_fm_noattn', 'bscan_unified_fm_cnnonly',
        'bscan_unified_fm_stemonly', 'bscan_unified_fm_attnonly',
    ]:
        fm_map = {
            'bscan_unified_ernie': 'rnaernie',
            'bscan_unified_bert': 'rnabert',
            'bscan_unified_fm': 'rnafm',
            'bscan_unified_msm': 'rnamsm',
            'bscan_unified_fm_cnnadapter':     'rnafm',
            'bscan_unified_fm_mambaadapter':   'rnafm',
            'bscan_unified_ernie_cnnadapter':  'rnaernie',
            'bscan_unified_ernie_mambaadapter':'rnaernie',
            # Branch ablation (all use rnafm)
            'bscan_unified_fm_fulltr':   'rnafm',
            'bscan_unified_fm_mlponly':  'rnafm',
            'bscan_unified_fm_nocnn':    'rnafm',
            'bscan_unified_fm_nostem':   'rnafm',
            'bscan_unified_fm_noattn':   'rnafm',
            'bscan_unified_fm_cnnonly':  'rnafm',
            'bscan_unified_fm_stemonly': 'rnafm',
            'bscan_unified_fm_attnonly': 'rnafm',
        }
        fm_name = fm_map[args_dict['model_name']]
        
        # We need one-hot for the Stem branch (Upper Intron and Lower Intron RC)
        u_oh_train, l_oh_train, label_tensor_train = data.seq_to_tensor(keys_train)
        u_oh_valid, l_oh_valid, label_tensor_valid = data.seq_to_tensor(keys_valid)
        u_oh_test, l_oh_test, label_tensor_test = data.seq_to_tensor(keys_test)
        
        L = args_dict['junction_bps']
        _RC_PERM = [3, 2, 1, 0]
        def get_intron_rc(l_oh, L):
            # l_oh is [N, 4, 2L]. lower_intron is [:, :, L:]
            l_int = l_oh[:, :, L:]
            return l_int[:, _RC_PERM, :].flip(dims=[2])

        train_dataset = circData_cached_fm(keys_train, label_tensor_train, fm_name, 
                                           upper_oh=u_oh_train[:, :, :L], 
                                           lower_rc_oh=get_intron_rc(l_oh_train, L))
        valid_dataset = circData_cached_fm(keys_valid, label_tensor_valid, fm_name, 
                                           upper_oh=u_oh_valid[:, :, :L], 
                                           lower_rc_oh=get_intron_rc(l_oh_valid, L))
        test_dataset = circData_cached_fm(keys_test, label_tensor_test, fm_name, 
                                          upper_oh=u_oh_test[:, :, :L], 
                                          lower_rc_oh=get_intron_rc(l_oh_test, L))

    elif args_dict['model_name'] in ['bscan_embedonly_ernie', 'bscan_embedonly_bert', 'bscan_embedonly_fm', 'bscan_embedonly_msm',
                                      'bscan_random_bert', 'bscan_random_msm']:
        fm_map = {
            'bscan_embedonly_ernie': 'rnaernie',
            'bscan_embedonly_bert': 'rnabert',
            'bscan_embedonly_fm': 'rnafm',
            'bscan_embedonly_msm': 'rnamsm',
            'bscan_random_bert': 'rnabert',
            'bscan_random_msm': 'rnamsm',
        }
        fm_name = fm_map[args_dict['model_name']]

        L = args_dict['junction_bps']
        _RC_PERM = [3, 2, 1, 0]
        def get_intron_rc(l_oh, L):
            l_int = l_oh[:, :, L:]
            return l_int[:, _RC_PERM, :].flip(dims=[2])

        # Token inputs use each FM tokenizer; one-hot tensors only support the stem branch.
        upper_train, lower_train, lower_rc_train, label_tensor_train = data.tensor_for_pretrained(
            keys_train, rc=True, tokenizer=fm_name, special_tokens=False
        )
        data.tokenizer = None
        upper_valid, lower_valid, lower_rc_valid, label_tensor_valid = data.tensor_for_pretrained(
            keys_valid, rc=True, tokenizer=fm_name, special_tokens=False
        )
        data.tokenizer = None
        upper_test, lower_test, lower_rc_test, label_tensor_test = data.tensor_for_pretrained(
            keys_test, rc=True, tokenizer=fm_name, special_tokens=False
        )

        u_oh_train, l_oh_train, _ = data.seq_to_tensor(keys_train)
        u_oh_valid, l_oh_valid, _ = data.seq_to_tensor(keys_valid)
        u_oh_test, l_oh_test, _ = data.seq_to_tensor(keys_test)

        train_dataset = circData_triple_oh(
            upper_train, lower_train, lower_rc_train,
            u_oh_train[:, :, :L], get_intron_rc(l_oh_train, L), label_tensor_train
        )
        valid_dataset = circData_triple_oh(
            upper_valid, lower_valid, lower_rc_valid,
            u_oh_valid[:, :, :L], get_intron_rc(l_oh_valid, L), label_tensor_valid
        )
        test_dataset = circData_triple_oh(
            upper_test, lower_test, lower_rc_test,
            u_oh_test[:, :, :L], get_intron_rc(l_oh_test, L), label_tensor_test
        )

    elif args_dict['model_name'] in ['circattrcm', 'circcnnatt', 'bscan_v2', 'circmamba', 'circfusion', 'circalignmap', 'circunified',
                                   'circattrcm_scratch', 'circalignmap_scratch', 'circunified_scratch',
                                   'bscan_unified_onehot']:
        upper_train, lower_train, lower_rc_train, label_tensor_train = data.tensor_for_pretrained(
            keys_train, rc=True, tokenizer='rnaernie', special_tokens=False
        )
        upper_valid, lower_valid, lower_rc_valid, label_tensor_valid = data.tensor_for_pretrained(
            keys_valid, rc=True, tokenizer='rnaernie', special_tokens=False
        )
        upper_test, lower_test, lower_rc_test, label_tensor_test = data.tensor_for_pretrained(
            keys_test, rc=True, tokenizer='rnaernie', special_tokens=False
        )
        train_dataset = circData_triple(upper_train, lower_train, lower_rc_train, label_tensor_train)
        valid_dataset = circData_triple(upper_valid, lower_valid, lower_rc_valid, label_tensor_valid)
        test_dataset = circData_triple(upper_test, lower_test, lower_rc_test, label_tensor_test)

    elif args_dict['model_name'] in ['bscan_seq', 'bscan_seq_lite', 'bscan_seq_lite_xattn', 'bscan_seq_rcaug', 'bscan_seq_rcattn', 'bscan_seq_mamba_aux', 'bscan_plus', 'bscan_mamba_xattn', 'bscan_region_interact']:
        upper_train, lower_train, _, label_tensor_train = data.tensor_for_pretrained(
            keys_train, rc=False, tokenizer='rnaernie', special_tokens=False
        )
        upper_valid, lower_valid, _, label_tensor_valid = data.tensor_for_pretrained(
            keys_valid, rc=False, tokenizer='rnaernie', special_tokens=False
        )
        upper_test, lower_test, _, label_tensor_test = data.tensor_for_pretrained(
            keys_test, rc=False, tokenizer='rnaernie', special_tokens=False
        )
        train_dataset = circData_double(upper_train, lower_train, label_tensor_train)
        valid_dataset = circData_double(upper_valid, lower_valid, label_tensor_valid)
        test_dataset = circData_double(upper_test, lower_test, label_tensor_test)
        
    elif args_dict['model_name'] in ['circcnnrcm']:
        rcm_flanking_list = [args_dict["flanking_bps"]]
        rcm_kmer_list = [5, 7, 9, 11, 13]
        upper_train, lower_train, flanking_rcm_train, upper_rcm_train, lower_rcm_train, label_tensor_train = data.seq_to_tensor_w_rcm(
            keys_train, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
        )
        upper_valid, lower_valid, flanking_rcm_valid, upper_rcm_valid, lower_rcm_valid, label_tensor_valid = data.seq_to_tensor_w_rcm(
            keys_valid, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
        )
        upper_test, lower_test, flanking_rcm_test, upper_rcm_test, lower_rcm_test, label_tensor_test = data.seq_to_tensor_w_rcm(
            keys_test, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
        )
        train_dataset = circData_triple(flanking_rcm_train, upper_rcm_train, lower_rcm_train, label_tensor_train)
        valid_dataset = circData_triple(flanking_rcm_valid, upper_rcm_valid, lower_rcm_valid, label_tensor_valid)
        test_dataset = circData_triple(flanking_rcm_test, upper_rcm_test, lower_rcm_test, label_tensor_test)

    elif args_dict['model_name'] in ['circcnntri']:
        rcm_flanking_list = [args_dict["flanking_bps"]]
        rcm_kmer_list = [5, 7, 9, 11, 13]
        upper_train, lower_train, flanking_rcm_train, upper_rcm_train, lower_rcm_train, label_tensor_train = data.seq_to_tensor_w_rcm(
            keys_train, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
        )
        upper_valid, lower_valid, flanking_rcm_valid, upper_rcm_valid, lower_rcm_valid, label_tensor_valid = data.seq_to_tensor_w_rcm(
            keys_valid, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
        )
        upper_test, lower_test, flanking_rcm_test, upper_rcm_test, lower_rcm_test, label_tensor_test = data.seq_to_tensor_w_rcm(
            keys_test, flanking_list=rcm_flanking_list, kmer_list=rcm_kmer_list
        )
        train_dataset = circData_rcm(upper_train, lower_train, flanking_rcm_train, upper_rcm_train, lower_rcm_train, label_tensor_train)
        valid_dataset = circData_rcm(upper_valid, lower_valid, flanking_rcm_valid, upper_rcm_valid, lower_rcm_valid, label_tensor_valid)
        test_dataset = circData_rcm(upper_test, lower_test, flanking_rcm_test, upper_rcm_test, lower_rcm_test, label_tensor_test)

    elif args_dict['model_name'] in ['circdeep']:
        seqs_train, _, _, _, label_tensor_train = data.seq_to_index(keys_train, kmer=args_dict['kmer'])
        seqs_valid, _, _, _, label_tensor_valid = data.seq_to_index(keys_valid, kmer=args_dict['kmer'])
        seqs_test, _, _, _, label_tensor_test = data.seq_to_index(keys_test, kmer=args_dict['kmer'])
        train_dataset = circData_single(seqs_train, label_tensor_train)
        valid_dataset = circData_single(seqs_valid, label_tensor_valid)
        test_dataset = circData_single(seqs_test, label_tensor_test)
        
    elif args_dict['model_name'] in ['jedi']:
        _, upper_train, lower_train, _, label_tensor_train = data.seq_to_index(keys_train, kmer=args_dict['kmer'])
        _, upper_valid, lower_valid, _, label_tensor_valid = data.seq_to_index(keys_valid, kmer=args_dict['kmer'])
        _, upper_test, lower_test, _, label_tensor_test = data.seq_to_index(keys_test, kmer=args_dict['kmer'])
        train_dataset = circData_double(upper_train, lower_train, label_tensor_train)
        valid_dataset = circData_double(upper_valid, lower_valid, label_tensor_valid)
        test_dataset = circData_double(upper_test, lower_test, label_tensor_test)
        
    else:
        assert False, 'Model name not found!'
        
    
    trainer = Trainer(seed=args_dict['seed'], device=device)
    
    trainer.set_dataloader(train_dataset, part=0, batch_size=args_dict['batch_size'])
    trainer.set_dataloader(valid_dataset, part=1, batch_size=args_dict['batch_size'])
    trainer.set_dataloader(test_dataset, part=2, batch_size=args_dict['batch_size'])
    
    # Model-specific kwargs
    model_name = args_dict["model_name"]
    kwargs = {}
    if model_name in ["deepcirccode"]:
        kwargs['d_model'] = args_dict["dim"]
    elif model_name in ["circstem", "circstemv2", "circbialign", "circmotif",
                        "circcombine", "circcombine_cnn", "circcombine_motif",
                        "circcombine_stem", "circcombine_attn",
                        "circcombine_no_motif", "circcombine_no_stem", "circcombine_no_attn",
                        "circsplice", "circsplice_v2", "circcnn", "bscan", "bscan_cnn", 
                        "bscan_stem", "bscan_attn"]:
        kwargs['junction_bps'] = args_dict["junction_bps"]
    elif model_name in ["circcnndouble", "circcnndoubleshare"]:
        kwargs['length_seq'] = 2 * args_dict["junction_bps"]
    elif model_name in ["bscan_v2", "bscan_seq", "bscan_seq_lite", "bscan_seq_lite_xattn", "bscan_seq_rcaug", "bscan_seq_rcattn", "bscan_seq_mamba_aux", "bscan_plus"]:
        kwargs['junction_bps'] = args_dict["junction_bps"]
        kwargs['length_seq'] = 2 * args_dict["junction_bps"]
    elif model_name in ["bscan_embedonly_ernie", "bscan_embedonly_bert", "bscan_embedonly_fm", "bscan_embedonly_msm"]:
        kwargs['junction_bps'] = args_dict["junction_bps"]
    elif model_name in ["bscan_mamba_xattn"]:
        kwargs['junction_bps'] = args_dict["junction_bps"]
    elif model_name in ["bscan_region_interact"]:
        kwargs['junction_bps'] = args_dict["junction_bps"]
    elif model_name in ["bscan_region_stem"]:
        kwargs['junction_bps'] = args_dict["junction_bps"]
    elif model_name in ["circcnnsingle"]:
        kwargs['upper_input_dim'] = 4 * args_dict["junction_bps"]
    elif model_name in ["circcnnrcm", "circcnntri"]:
        kwargs['n_rcm_features'] = 1 * 5
    
    trainer.define_model(model_name, **kwargs)
    trainer.set_optimizer(args_dict['optimizer'], args_dict['lr'])
    clean_gpu()
    trainer.train(epochs=args_dict['epochs'], earlystop=args_dict['earlystop'], verbose=args_dict['verbose'])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model based on provided arguments.")
    
    parser.add_argument('--junction_bps', type=int, default=100)
    parser.add_argument('--flanking_bps', type=int, default=100, choices=[100, 200, 300, 400, 500, 1000])
    parser.add_argument('--use_full_intron', action='store_true', help='Use full intron sequences instead of junction_bps')
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_samples', type=int, default=None, help='Optional: cap number of samples per split for quick testing.')
    parser.add_argument(
        '--split_strategy',
        type=str,
        default='sample',
        choices=['sample', 'transcript', 'chromosome'],
        help='sample: original stratified sample split; transcript: grouped split with transcript IDs held out across partitions; chromosome: grouped split with chromosome held out across partitions.',
    )
    
    parser.add_argument('--model_name', type=str, default='deepcirccode', 
                        choices=['deepcirccode', 
                                 'circdeep',
                                 'circdc',
                                 'circcnn',
                                 'jedi',
                                 'circcnnsingle', 
                                 'circcnndouble', 
                                 'circcnndoubleshare',
                                 'circcnntri', 
                                 'circcnnrcm', 
                                 'circcnnatt',
                                 'bscan_v2',
                                 'bscan_seq',
                                 'bscan_seq_lite',
                                 'bscan_seq_lite_xattn',
                                 'bscan_seq_rcaug',
                                 'bscan_seq_rcattn',
                                 'bscan_seq_mamba_aux',
                                 'bscan_plus',
                                 'bscan_mamba_xattn',
                                 'bscan_region_interact',
                                 'circattrcm',
                                 'circmamba',
                                 'circfusion',
                                 'circalignmap',
                                 'circunified',
                                 'circnet',
                                 'circstem',
                                 'circstemv2',
                                 'circbialign',
                                 'circmotif',
                                 'circcombine',
                                 'circcombine_cnn', 'circcombine_motif',
                                 'circcombine_stem', 'circcombine_attn',
                                 'circcombine_no_motif', 'circcombine_no_stem', 'circcombine_no_attn',
                                 'circsplice', 'circsplice_v2',
                                 'bscan', 'bscan_cnn', 'bscan_stem', 'bscan_attn',
                                 'bscan_region_stem',
                                 'bscan_unified_onehot', 'bscan_unified_ernie', 'bscan_unified_bert',
                                 'bscan_unified_fm', 'bscan_unified_msm',
                                 'bscan_unified_fm_cnnadapter', 'bscan_unified_fm_mambaadapter',
                                 'bscan_unified_ernie_cnnadapter', 'bscan_unified_ernie_mambaadapter',
                                 'bscan_embedonly_ernie', 'bscan_embedonly_bert',
                                 'bscan_embedonly_fm', 'bscan_embedonly_msm',
                                 'bscan_random_bert', 'bscan_random_msm',
                                 # Branch ablation variants
                                 'bscan_unified_fm_fulltr', 'bscan_unified_fm_mlponly', 'bscan_unified_fm_nocnn',
                                 'bscan_unified_fm_nostem', 'bscan_unified_fm_noattn',
                                 'bscan_unified_fm_cnnonly', 'bscan_unified_fm_stemonly',
                                 'bscan_unified_fm_attnonly',
                                 ])
    
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--kmer', type=int, default=3)
    
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--earlystop', type=int, default=30)
    
    parser.add_argument('--verbose', action='store_true')
    
    parser.add_argument('--device', type=int, default=-1, choices=[-1, 0, 1, 2, 3])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    args_dict = vars(args)
        
    experiment(args_dict)
