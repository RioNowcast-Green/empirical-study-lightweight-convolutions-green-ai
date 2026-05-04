def validate_loaded_sequence_length(
    seq_len: int, input_seq_len: int, output_seq_len: int
):
    if input_seq_len + output_seq_len > seq_len:
        raise Exception(
            f"There aren't any images left ({seq_len - input_seq_len}) to fulfill the output sequence length (={output_seq_len})"
        )
