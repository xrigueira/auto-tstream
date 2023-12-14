import utils
import pandas as pd

timestamp_col_name = "time"

# Read the data
data = utils.read_data(timestamp_col_name=timestamp_col_name)
data.drop('Unnamed: 0', axis=1, inplace=True)

def get_indices(data: pd.DataFrame, window_size: int, step_size: int):
    
    # Define the stop position
    stop_position = len(data) - 1 # because of 0 indexing in Python
    
    # Start the first sub-sequence at index 0
    subseq_first_idx = 0
    subseq_last_idx = window_size
    
    indices = []
    while subseq_last_idx <= stop_position:
        
        indices.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_last_idx += step_size
    
    return indices

encoder_sequence_len = 96
decoder_sequence_len = 12
output_sequence_len = 96

window_size = encoder_sequence_len + output_sequence_len
step_size = 1

indices = get_indices(data=data, window_size=window_size, step_size=step_size)
print(len(data))
print(indices[0])
print(indices[-2])
print(indices[-1])