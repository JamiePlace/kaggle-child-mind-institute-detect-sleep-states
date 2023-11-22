import numpy as np


def split_array_into_chunks(array: np.ndarray, window_size: int) -> np.ndarray:
    num_rows = array.shape[0]
    num_chunks = num_rows // window_size
    remaining_rows = num_rows % window_size

    # Split the array into chunks of size window_size
    chunks = np.split(array[: num_chunks * window_size], num_chunks)

    # If there are remaining rows, create a chunk with the remaining rows and pad with zeros
    if remaining_rows > 0:
        remaining_chunk = np.concatenate(
            (
                array[num_chunks * window_size :],
                np.zeros((window_size - remaining_rows, array.shape[1])),
            )
        )
        chunks.append(remaining_chunk)

    return np.array(chunks)
