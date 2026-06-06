import torch
import numpy as np
import json
import os

class CalPPM():
    def __init__(
        self, 
        mode,
        splice_type, 
        train_key, 
        model_path, 
        input_seq_len, 
        padding_len, 
        kernel_len,
        ppm_file_name
    ):
        if splice_type.lower() not in ['bs', 'ls']:
            print('splice_type should be bs or ls')
            return
        self.splice_type = splice_type.lower()

        self.train_key = train_key

        if mode not in [0, 1, 2]:
            print('mode should be 0, 1, or 2')
            return
        self.mode = mode

        self.model_path = model_path
        self.input_seq_len = input_seq_len
        self.padding_len = padding_len
        self.kernel_len = kernel_len
        self.ppm_file_name = ppm_file_name

        self.half_train_keys = None
        self.BS_upper_seqs = None
        self.BS_lower_seqs = None
        self.BS_upper_lower_concat_seqs = None
        self.activation = None
        self.subseq_starting_index_all_kernel = None
        self.sub_seq_all_kernels = None

    def get_sequences(
        self, 
        junction_dict, 
        mode: int = 0,
    ):
        if mode not in [0, 1, 2]:
            print('mode should be 0, 1, or 2')
            return
        
        sequences_list = []

        half_train_keys = []
        for key in self.train_key:
            label = junction_dict[key]['label']
            if label == self.splice_type:
                half_train_keys.append(key)
                if mode == 0:
                    sequences_list.append(junction_dict[key]['upper_intron'] + junction_dict[key]['upper_exon'])
                elif mode == 1:
                    sequences_list.append(junction_dict[key]['lower_exon'] + junction_dict[key]['lower_intron'])
                else:
                    sequences_list.append(junction_dict[key]['upper_intron'] + junction_dict[key]['upper_exon'] + \
                                                 junction_dict[key]['lower_exon'] + junction_dict[key]['lower_intron'])
        self.half_train_keys = half_train_keys
        return sequences_list

    def seq_to_matrix(self, seq):
        """Convert a sequence to a one-hot encoded matrix"""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        matrix = np.zeros((4, len(seq)))

        for i, base in enumerate(seq):
            if base in mapping:
                matrix[mapping[base], i] = 1
        return matrix

    def get_seq_tensor(self, seq_list):
        seq_matrix_tensor_list = []
        for seq in seq_list:
            seq_matrix = self.seq_to_matrix(seq)
            seq_matrix_tensor = torch.from_numpy(seq_matrix).to(torch.float32)
            seq_matrix_tensor_list.append(seq_matrix_tensor)
        return torch.stack(seq_matrix_tensor_list, dim=0)

    def get_conv1_activation(self, junction_dict):
        best_base_model = torch.load(self.model_path)
        best_base_model.eval()

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        if self.mode in [0, 1]:
            best_base_model.cnn_upper.conv1.register_forward_hook(get_activation('upper_conv1'))
            best_base_model.cnn_lower.conv1.register_forward_hook(get_activation('lower_conv1'))

            best_base_model.to('cpu')
            self.BS_upper_seqs = self.get_sequences(junction_dict, mode=0)
            self.BS_lower_seqs = self.get_sequences(junction_dict, mode=1)

            X1 = self.get_seq_tensor(self.BS_upper_seqs).to('cpu')
            X2 = self.get_seq_tensor(self.BS_lower_seqs).to('cpu')

            output = best_base_model(X1, X2)

            self.activation = activation

        else:
            best_base_model.cnn.conv1.register_forward_hook(get_activation('conv1'))

            ### move the model to cpu
            best_base_model.to('cpu')
            self.BS_upper_lower_concat_seqs = self.get_sequences(junction_dict, mode=2)

            X = self.get_seq_tensor(self.BS_upper_lower_concat_seqs).to('cpu')

            output = best_base_model(X)

            self.activation = activation

    def get_subseq_starting_index(self):
        self.get_conv1_activation()
        if self.mode == 0:
            all_activation_values = self.activation['upper_conv1']
        elif self.mode == 1:
            all_activation_values = self.activation['lower_conv1']
        elif self.mode == 2:
            all_activation_values = self.activation['conv1']

        subseq_starting_index_all_kernel = []

        for i in all_activation_values.permute(1, 0, 2):  ## from 11000 X 512 X 200 to 512 X 11000 X 200
            max_index = torch.argmax(i[:, self.padding_len:self.input_seq_len - self.padding_len - self.kernel_len], dim=1).numpy()
            positive_row = torch.any(i[:, self.padding_len:self.input_seq_len - self.padding_len - self.kernel_len] > 0, dim=1)
            positive_row_bool = np.array([np.nan if not is_positive else 1 for is_positive in positive_row])

            subseq_starting_index_individual_kernel = (max_index * positive_row_bool).tolist()
            subseq_starting_index_all_kernel.append(subseq_starting_index_individual_kernel)

        self.subseq_starting_index_all_kernel = subseq_starting_index_all_kernel

    def get_all_subseqs(self):
        self.get_subseq_starting_index()

        if self.mode == 0:
            sequences = self.BS_upper_seqs

        elif self.mode == 1:
            sequences = self.BS_lower_seqs

        else:
            sequences = self.BS_upper_lower_concat_seqs

        sub_seq_all_kernels = []
        for subseq_indexs in self.subseq_starting_index_all_kernel:
            sub_seq_each_kernel = []
            for seq_index, starting_index in enumerate(subseq_indexs):
                if not np.isnan(starting_index):  # test for the row that have no positive activation values
                    sub_seq_each_kernel.append(
                        sequences[seq_index][int(starting_index):int(starting_index) + self.kernel_len])
            sub_seq_all_kernels.append(sub_seq_each_kernel)

        self.sub_seq_all_kernels = sub_seq_all_kernels

    def seq_to_matrix_for_tomtom(self, input_seq):
        row_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # should exclude the 'Ns' in the input sequence
        input_mat = np.zeros((4, len(input_seq)))

        for col_index, base in enumerate(input_seq):
            input_mat[row_index[base]][col_index] = 1
        return input_mat

    def get_position_prob_matrix(self, subseq_list):
        one_hot_seq_list = []
        for seq in subseq_list:
            one_hot_seq = self.seq_to_matrix_for_tomtom(seq)
            one_hot_seq_list.append(one_hot_seq)
        z = np.zeros_like(one_hot_seq)
        for one_hot in one_hot_seq_list:
            z += one_hot

        return z / np.sum(z, axis=0)

    def write_out_PPM(self):

        self.get_all_subseqs()

        os.makedirs('./Extracted_motifs', exist_ok=True)
        with open(f'./Extracted_motifs/{self.ppm_file_name}.txt', 'w') as f:
            f.write('MEME version 4.9.0\n\n'
                    'ALPHABET= ACGT\n\n'
                    'strands: + -\n\n'
                    'Background letter frequencies (from uniform background):\n'
                    'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')

            for kernel_number in range(len(self.sub_seq_all_kernels)):
                f.write(f'MOTIF {kernel_number}\n')
                f.write( f"letter-probability matrix: alength= 4 w= {self.kernel_len} nsites= {self.kernel_len} E= 1e-6\n")
                for line in self.get_position_prob_matrix(self.sub_seq_all_kernels[kernel_number]).T:
                    f.write('\t'.join([str(item) for item in line]) + '\n')
                f.write('\n')
