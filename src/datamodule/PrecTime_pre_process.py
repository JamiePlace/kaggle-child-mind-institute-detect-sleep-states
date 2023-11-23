import polars as pl
import numpy as np


def split_array_into_chunks(
    array: np.ndarray, event_df: pl.DataFrame, window_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_rows = array.shape[0]
    num_chunks = num_rows // window_size
    remaining_rows = num_rows % window_size

    # clean up the array such that we don't have data passed the last event
    array = array[: event_df["wakeup"].max(), :]
    # create the label from the event_df
    label = np.zeros(array.shape[0])
    # for each row of event_df find the corresponding rows in array from
    # event_df["onset"] to event_df["wakeup"] to set the label to 1
    for row in event_df.rows(named=True):
        label[row["onset"] : row["wakeup"] + 1] = 1

    array_with_label = np.concatenate((array, label[:, None]), axis=1)
    # Split the array into chunks of size window_size
    chunks = np.split(array_with_label[: num_chunks * window_size], num_chunks)

    # If there are remaining rows, create a chunk with the remaining rows and pad with zeros
    if remaining_rows > 0:
        remaining_chunk = np.concatenate(
            (
                array_with_label[num_chunks * window_size :],
                np.zeros(
                    (window_size - remaining_rows, array_with_label.shape[1])
                ),
            )
        )
        chunks.append(remaining_chunk)
    chunks = np.array(chunks)

    dense_labels = chunks[:, :, -1]
    sparse_labels = np.array(
        [
            0 if val < window_size // 2 else 1
            for val in chunks[:, :, -1].sum(axis=1)
        ]
    )

    return chunks[:, :, :-1], dense_labels, sparse_labels
