"""
Main experiment script for the circRNA expression regression task.
"""
import argparse
from utils import seed_everything, get_device
from trainer_regression import RegressionTrainer
from dataloader_regression import RegressionDataSetPrep
import gc
import torch
import random

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def experiment_regression(args_dict: dict):
    """
    Run the regression training process based on the provided arguments.
    """
    seed_everything(args_dict['seed'])
    device = get_device(args_dict['device'])
    print(f'Using device: {device}')

    # 1. Initialize the new Regression Dataloader
    data = RegressionDataSetPrep(
        coord_path='./circRNA_expression.csv',
        seq_dict_path='./data/hg38.json', # Using hg38 as it might be more recent
        junction_bps=args_dict['junction_bps'],
        seed=args_dict['seed'],
        log_transform=True
    )
    
    # Load coordinates/labels and extract sequences
    data.load_data()
    data.get_sequences_for_dataframe()
    
    # 2. Split data
    train_keys, valid_keys, test_keys = data.split_data()

    max_samples = args_dict.get("max_samples")
    if max_samples:
        rng = random.Random(args_dict["seed"])
        train_keys = rng.sample(list(train_keys), min(max_samples, len(train_keys)))
        valid_keys = rng.sample(list(valid_keys), min(max_samples, len(valid_keys)))
        test_keys = rng.sample(list(test_keys), min(max_samples, len(test_keys)))
        print(f"Subsampled to max {max_samples} samples per split.")

    # 3. Create PyTorch datasets based on model input type
    if args_dict['model_name'] == 'deepcirccode_regression':
        train_dataset = data.create_tensors_single(train_keys)
        valid_dataset = data.create_tensors_single(valid_keys)
        test_dataset = data.create_tensors_single(test_keys)
    elif args_dict['model_name'] in ['bscan_regression', 'circcnn_regression']:
        train_dataset = data.create_tensors(train_keys)
        valid_dataset = data.create_tensors(valid_keys)
        test_dataset = data.create_tensors(test_keys)
    elif args_dict['model_name'] == 'bscan_v2_regression':
        train_dataset = data.create_tensors_pretrained(train_keys, tokenizer='rnaernie', special_tokens=False)
        valid_dataset = data.create_tensors_pretrained(valid_keys, tokenizer='rnaernie', special_tokens=False)
        test_dataset = data.create_tensors_pretrained(test_keys, tokenizer='rnaernie', special_tokens=False)
    elif args_dict['model_name'] in ['bscan_seq_regression', 'bscan_seq_lite_regression', 'bscan_seq_lite_xattn_regression', 'bscan_seq_rcattn_regression', 'bscan_plus_regression', 'bscan_mamba_xattn_regression', 'bscan_region_interact_regression']:
        train_dataset = data.create_tensors_pretrained_double(train_keys, tokenizer='rnaernie', special_tokens=False)
        valid_dataset = data.create_tensors_pretrained_double(valid_keys, tokenizer='rnaernie', special_tokens=False)
        test_dataset = data.create_tensors_pretrained_double(test_keys, tokenizer='rnaernie', special_tokens=False)
    else:
        raise ValueError(f"Model {args_dict['model_name']} not supported in this script.")

    # 4. Initialize the new Regression Trainer
    trainer = RegressionTrainer(
        seed=args_dict['seed'],
        device=device,
        loss_name=args_dict['loss'],
        corr_loss_weight=args_dict['corr_loss_weight'],
        huber_delta=args_dict['huber_delta'],
        early_metric=args_dict['early_metric'],
    )
    
    trainer.set_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size=args_dict['batch_size'])
    
    # 5. Define the regression model
    model_kwargs = {'junction_bps': args_dict["junction_bps"]}
    if args_dict["model_name"] in ['bscan_v2_regression', 'bscan_seq_regression', 'bscan_seq_lite_regression', 'bscan_seq_lite_xattn_regression', 'bscan_seq_rcattn_regression', 'bscan_plus_regression']:
        model_kwargs['length_seq'] = 2 * args_dict["junction_bps"]
    elif args_dict["model_name"] in ['bscan_mamba_xattn_regression']:
        pass
    elif args_dict["model_name"] in ['bscan_region_interact_regression']:
        pass
    trainer.define_model(args_dict["model_name"], **model_kwargs)
        
    # 6. Start training
    clean_gpu()
    results = trainer.train(
        optimizer_name=args_dict['optimizer'],
        lr=args_dict['lr'],
        epochs=args_dict['epochs'],
        early_stop_patience=args_dict['earlystop'],
        verbose=args_dict['verbose']
    )
    print("Final test results:", results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a regression model for circRNA expression.")
    
    parser.add_argument('--junction_bps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=None, help='Cap samples for quick testing.')
    
    parser.add_argument('--model_name', type=str, default='bscan_regression', choices=['bscan_regression', 'bscan_v2_regression', 'bscan_seq_regression', 'bscan_seq_lite_regression', 'bscan_seq_lite_xattn_regression', 'bscan_seq_rcattn_regression', 'bscan_plus_regression', 'bscan_mamba_xattn_regression', 'bscan_region_interact_regression', 'circcnn_regression', 'deepcirccode_regression'])
    
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='smoothl1', choices=['mse', 'mae', 'smoothl1', 'huber'])
    parser.add_argument('--huber_delta', type=float, default=1.0)
    parser.add_argument('--corr_loss_weight', type=float, default=0.0)
    parser.add_argument('--early_metric', type=str, default='r2', choices=['mae', 'rmse', 'pearson', 'r2', 'composite'])
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--earlystop', type=int, default=10)
    
    parser.add_argument('--verbose', action='store_true', default=True)
    
    parser.add_argument('--device', type=int, default=0, choices=[-1, 0, 1, 2, 3])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args_dict = vars(args)
        
    experiment_regression(args_dict)
