import numpy as np
import os
import json
from collections import defaultdict
from itertools import product
from dataloader import DataSetPrep
import argparse
from tqdm import tqdm


class RCSFinder:
    '''
    This class finds the imperfect reverse complementary sequences (RCS) within a single intronic sequence 
    or between two flanking introns, and can also calculate a score matrix.
    '''
    def __init__(self, key=None, upper_seq=None, lower_seq=None, is_flanking_introns=None,
                 is_upper_intron=None, seq_fraction_of_spacer=None, kmers=None, allowed_seed_mismatch=None):
        self.key = key
        self.upper_seq = upper_seq
        self.lower_seq = lower_seq
        self.is_flanking_introns = is_flanking_introns
        self.is_upper_intron = is_upper_intron
        self.seq_fraction_of_spacer = seq_fraction_of_spacer
        self.kmers = kmers
        self.allowed_seed_mismatch = allowed_seed_mismatch

    def base_conversion(self, seq):
        base_mapping = {'A': 1, 'T': -1, 'C': 1j, 'G': -1j, 'N': 0}
        if len(seq) >= 2:
            seq_map = np.array([base_mapping[i] for i in seq])
        else:
            seq_map = base_mapping[seq]
        return seq_map

    def get_sub_seq_score(self, seq):
        aug_seq = 'N' + seq
        aug_seq_M = self.base_conversion(aug_seq)
        aug_seq_cum_M = aug_seq_M.cumsum()
        seq_cum_M = aug_seq_cum_M[1:]
        sub_seq_score = seq_cum_M[self.kmers - 1:] - aug_seq_cum_M[:len(aug_seq) - self.kmers]
        return sub_seq_score

    def get_allowed_window_index(self):
        '''Identify valid window pairs for potential reverse complementary sequences.'''
        if self.is_flanking_introns:
            subseq_score1 = self.get_sub_seq_score(self.upper_seq).astype(np.complex64)
            subseq_score2 = self.get_sub_seq_score(self.lower_seq).astype(np.complex64)
            subseq_scores_sum = np.add.outer(subseq_score1, subseq_score2)
            subseq_scores_abs_sum_mat = abs(subseq_scores_sum.imag) + abs(subseq_scores_sum.real)
            first_window, second_window = np.where(subseq_scores_abs_sum_mat <= self.allowed_seed_mismatch)
            return first_window, second_window
        else:
            seq = self.upper_seq if self.is_upper_intron else self.lower_seq
            
            subseq_scores = self.get_sub_seq_score(seq).astype(np.complex64)
            subseq_scores_sum = subseq_scores.reshape((-1, 1)) + subseq_scores.reshape((1, -1))
            subseq_scores_abs_sum_mat = abs(subseq_scores_sum.imag) + abs(subseq_scores_sum.real)
            tril_index = np.tril_indices(subseq_scores_abs_sum_mat.shape[0])
            subseq_scores_abs_sum_mat[tril_index] = self.allowed_seed_mismatch + 1
            first_window, second_window = np.where(subseq_scores_abs_sum_mat <= self.allowed_seed_mismatch)
            window_num_diff = second_window - first_window
            spacer = window_num_diff - self.kmers
            allowed_index = (spacer >= len(seq) * self.seq_fraction_of_spacer)
            allowed_first_window = first_window[allowed_index]
            allowed_second_window = second_window[allowed_index]
            return allowed_first_window, allowed_second_window

    def subseq_validity_check(self):
        '''Validate the identified subsequences for reverse complementarity and return the pairs with a score matrix.'''
        first_window, second_window = self.get_allowed_window_index()
        valid_subseq_pairs_list = []
        complement_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'N': 'N'}
        input_seq1 = self.upper_seq if self.is_upper_intron else self.lower_seq
        input_seq2 = self.lower_seq if self.is_flanking_introns else self.upper_seq


        for first_index, second_index in zip(first_window, second_window):
            seed_mismatch_score = 0

            first_subseq, second_subseq = input_seq1[first_index:self.kmers + first_index], \
                                          input_seq2[second_index:self.kmers + second_index]

            if first_subseq[0] == complement_dict[second_subseq[-1]]:
                for i in range(1, self.kmers):
                    if first_subseq[i] != complement_dict[second_subseq[-i - 1]]:
                        seed_mismatch_score += 2
                        if seed_mismatch_score > self.allowed_seed_mismatch:
                            break

                if seed_mismatch_score <= self.allowed_seed_mismatch:
                    valid_subseq_pairs_list.append((first_subseq, first_index, second_subseq, \
                                                    second_index, seed_mismatch_score))
        
        upper_equal_space_list = np.linspace(start=0, stop=len(input_seq1), num=5 + 1)
        lower_equal_space_list = np.linspace(start=0, stop=len(input_seq2), num=5 + 1)

        upper_interval_list = []
        lower_interval_list = []

        for i in range(len(upper_equal_space_list) - 1):
            upper_interval = (upper_equal_space_list[i], upper_equal_space_list[i + 1])
            upper_interval_list.append(upper_interval)

        for i in range(len(lower_equal_space_list) - 1):
            lower_interval = (lower_equal_space_list[i], lower_equal_space_list[i + 1])
            lower_interval_list.append(lower_interval)

        interval_combinations = list(product(upper_interval_list, lower_interval_list))
        self.interval_combinations = interval_combinations

        upper_lower_rcm_kmer_pos_pairs = []
        for i in valid_subseq_pairs_list:
            upper_lower_rcm_kmer_pos_pairs.append((i[1], i[3]))

        num_rcm_kmer_cross_intervals = []

        for interval_comb in self.interval_combinations:
            upper_interval_pos = interval_comb[0]
            lower_interval_pos = interval_comb[1]
            num_rcm_kmer_cross_intervals.append(sum([(upper + 0.5 * self.kmers >= upper_interval_pos[0] and \
                                                      upper + 0.5 * self.kmers < upper_interval_pos[1]) and \
                                                     (lower + 0.5 * self.kmers >= lower_interval_pos[0] and \
                                                      lower + 0.5 * self.kmers < lower_interval_pos[1]) \
                                                     for upper, lower in upper_lower_rcm_kmer_pos_pairs]))

        joint_rcm_kmer_dist = np.array(num_rcm_kmer_cross_intervals)

        return self.key, list(joint_rcm_kmer_dist)


# Processors for flanking, upper, and lower introns
def rcs_flanking_introns(seq_list, kmers):
    return [RCSFinder(key=key, upper_seq=value['upper_flanking'], lower_seq=value['lower_flanking'], 
                      is_flanking_introns=True, kmers=kmers, is_upper_intron=False, seq_fraction_of_spacer=0, allowed_seed_mismatch=0).subseq_validity_check() 
            for key, value in seq_list]


def rcs_upper_introns(seq_list, kmers):
    return [RCSFinder(key=key, upper_seq=value['upper_flanking'], lower_seq=value['lower_flanking'],
                      is_flanking_introns=False, kmers=kmers, is_upper_intron=True, seq_fraction_of_spacer=0, allowed_seed_mismatch=0).subseq_validity_check() 
            for key, value in seq_list]


def rcs_lower_introns(seq_list, kmers):
    return [RCSFinder(key=key, upper_seq=value['upper_flanking'], lower_seq=value['lower_flanking'], 
                      is_flanking_introns=False, kmers=kmers, is_upper_intron=False, seq_fraction_of_spacer=0, allowed_seed_mismatch=0).subseq_validity_check() 
            for key, value in seq_list]


def loop_rcs_processing(seq_dict, kmers, bps, rcm_type="flanking"):
    '''Main loop that processes RCS for the chosen intron type and saves results.'''
    score_dict = defaultdict(list)

    # Choose the processing function based on the RCM type
    if rcm_type == "flanking":
        process_function = rcs_flanking_introns
    elif rcm_type == "upper":
        process_function = rcs_upper_introns
    else:
        process_function = rcs_lower_introns

    # Generate sequence list and process it
    seq_list = [(key, seq_dict[key]) for key in tqdm(seq_dict.keys(), desc=f"Processing {rcm_type.capitalize()} Introns")]
    flat_list = process_function(seq_list=seq_list, kmers=kmers)

    for item in flat_list:
        score_dict[item[0]].append(item[1])  # Store the actual RCS result

    # Save results and scores to files
    save_results_and_scores(score_dict, rcm_type, bps, kmers)


def save_results_and_scores(score_dict, rcm_type, bps, kmers):
    '''Helper function to save results and scores into JSON files.'''
    rcm_scores_folder = './rcm_scores/'
    os.makedirs(rcm_scores_folder, exist_ok=True)

    # Convert NumPy-specific types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        elif isinstance(obj, np.integer):
            return int(obj)  # Convert np.int64 to int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert np.float64 to float
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively process dict
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]  # Recursively process list
        else:
            return obj  # Return the object if it’s already JSON-serializable

    # Create a JSON-serializable version of score_dict
    json_serializable_dict = convert_to_serializable(score_dict)

    # Save the RCM scores (now JSON-compatible)
    scores_filename = f"{rcm_type}_{bps}_bps_{kmers}mer_scores.json"
    with open(os.path.join(rcm_scores_folder, scores_filename), 'w') as f:
        json.dump(json_serializable_dict, f)

    print(f"{rcm_type.capitalize()} RCM processing completed for {bps}bps with {kmers}-mer.")



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--junction_bps', type=int, default=100, help='Length of the junction introns.')
    argparser.add_argument('--flanking_bps', type=int, default=1000, help='Length of the flanking introns.')
    argparser.add_argument('--flanking_list', type=int, nargs='*', default=None, help='Optional list of flanking lengths to compute (overrides --flanking_bps).')
    argparser.add_argument('--rcm_type', type=str, default="flanking", choices=["flanking", "upper", "lower"], help='Type of RCM to process.')
    argparser.add_argument('--kmers', type=int, nargs='*', default=None, help='Optional list of k values (default: 5 7 9 11 13).')
    args = argparser.parse_args()

    # Load the input data (example assumes it's in JSON format)
    flanking_list = args.flanking_list if args.flanking_list is not None else [args.flanking_bps]
    for flanking_bps in flanking_list:
        data = DataSetPrep(
            coord_path='./data/BS_LS_coordinates_final.csv', 
            seq_dict_path='./data/hg19_seq_dict.json', 
            junction_bps=args.junction_bps,
            flanking_bps=flanking_bps
        )
        try:
            junction, flanking = data.load_junction_flanking_seq()
            print('Successfully loaded the data!')
        except Exception as e:
            print(f"Failed to load the data: {e}. Processing the data.")
            junction, flanking = data.get_junction_intron_seq()
            
        kmers = args.kmers if args.kmers is not None else [5, 7, 9, 11, 13]
        for k in kmers:
            loop_rcs_processing(flanking, k, flanking_bps, args.rcm_type)
        print(f"Processing completed for {flanking_bps}bps flanking introns.")
    print("All processing completed!")
