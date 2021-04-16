import tensorflow as tf
import numpy as np
import musdb
import random
import gc
from tqdm import tqdm
from typing import Union, List, Tuple, Dict


class DatasetParam:
    """Contains parameters for dataset generation.

    Attributes:
        num_songs: Total number of songs
        num_samples: Total number of samples in one batch
        num_fragments: Total number of fragments in one sample
        len_fragment: The length of each fragment
        overlap: Number of samples in which each adjacent pair of fragments overlap
        repeat: Number of repeats
    """

    __slots__ = 'num_songs', 'num_samples', 'num_fragments', 'len_fragment', 'overlap', 'repeat'

    def __init__(self,
                 num_songs: int = 100,
                 num_samples: int = 100,
                 num_fragments: int = 40,
                 len_fragment: int = 16,
                 overlap: int = 8,
                 repeat: int = 400):
        if overlap >= len_fragment:
            raise ValueError("overlap must be smaller than len_fragment")

        self.num_songs = num_songs
        self.num_samples = num_samples
        self.num_fragments = num_fragments
        self.len_fragment = len_fragment
        self.overlap = overlap
        self.repeat = repeat


class DecodedTrack:
    """Contains decoded audio from the database.

    Attributes:
        length: Number of samples
        mixed: A tuple of numpy arrays from the mixture
        stems: Dictionary where the key is the name of the stem and the value is a tuple of numpy arrays from the stem
    """

    __slots__ = 'length', 'mixed', 'stems'

    @staticmethod
    def from_track(track):
        mixed = (track.audio[:, 0], track.audio[:, 1])
        length = mixed[0].shape[-1]
        stems = {}
        for stem in Provider.STEMS:
            audio = track.targets[stem].audio
            stems[stem] = (audio[:, 0], audio[:, 1])
        return DecodedTrack(length, mixed, stems)

    def __init__(self,
                 length: int,
                 mixed: Tuple[np.ndarray, np.ndarray],
                 stems: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        self.length = length
        self.mixed = mixed
        self.stems = stems


class Provider:
    """Decodes audio from the database.

    Attributes:
        tracks: List of tracks
        num_tracks: Number of tracks
        decoded: List of decoded tracks
        num_decoded: Number of decoded tracks in `decoded`
        max_decoded: Maximum number of decoded tracks
        ord_decoded: The order in which each track is decoded
        next_ord: The order which will be granted to the next decoded track
    """

    STEMS = "vocals", "drums", "bass", "other"

    def __init__(self, root: str, subsets: Union[str, List[str]] = "train", max_decoded: int = 100):
        if max_decoded < 1:
            raise ValueError("max_decoded must be greater than 0")

        self.tracks = list(musdb.DB(root=root, subsets=subsets))
        self.num_tracks = len(self.tracks)
        self.decoded: Dict[str, Union[NoneType, DecodedTrack]] = [
            None] * self.num_tracks
        self.num_decoded = 0
        self.max_decoded = max_decoded
        self.ord_decoded = [-1] * self.num_tracks
        self.next_ord = 0

    def decode(self, indices: Union[int, List[int]]):
        if type(indices) == int:
            indices = [indices]
        if len(indices) > self.max_decoded:
            raise ValueError("Cannot decode more than `max_decoded` tracks")

        indices = [idx for idx in indices if self.decoded[idx] == None]
        if indices:
            print(f"Decoding Audio {indices}...")
            for idx in tqdm(indices):
                self.ord_decoded
                if self.num_decoded == self.max_decoded:
                    idx = np.argmin(self.ord_decoded)
                    self.decoded[idx] = None
                    self.num_decoded -= 1
                    self.ord_decoded[idx] = -1
                    gc.collect()
                self.decoded[idx] = DecodedTrack.from_track(self.tracks[idx])
                self.num_decoded += 1
                self.ord_decoded[idx] = self.next_ord
                self.next_ord += 1

    def generate(self, p: DatasetParam):
        indices = list(range(self.num_tracks))
        random.shuffle(indices)
        indices = indices[:p.num_songs]
        self.decode(indices)

        duration = p.num_fragments * p.len_fragment - \
            (p.num_fragments - 1) * p.overlap

        # Make `p.repeat` batches
        for _ in range(p.repeat):
            x_batch = np.zeros(
                (p.num_samples * 2, p.num_fragments, p.len_fragment))
            y_batch = np.zeros(
                (p.num_samples * 2, len(Provider.STEMS), p.num_fragments, p.len_fragment))

            # Make `2 * p.num_samples` samples for each batch
            for i in range(p.num_samples):
                track = self.decoded[random.choice(indices)]
                start = random.randint(0, track.length - duration)

                for j in range(p.num_fragments):
                    left = i * 2
                    right = left + 1
                    begin = start + j * (p.len_fragment - p.overlap)
                    end = begin + p.len_fragment
                    x_batch[left][j] = track.mixed[0][begin:end]
                    x_batch[right][j] = track.mixed[1][begin:end]

                    for c, stem in enumerate(Provider.STEMS):
                        y_batch[left][c][j] = track.stems[stem][0][begin:end]
                        y_batch[right][c][j] = track.stems[stem][1][begin:end]

            yield x_batch, y_batch

    def make_dataset(self, p: DatasetParam) -> tf.data.Dataset:
        output_types = (tf.float32, tf.float32)
        output_shapes = (
            tf.TensorShape(
                (p.num_samples * 2, p.num_fragments, p.len_fragment)),
            tf.TensorShape(
                (p.num_samples * 2, len(Provider.STEMS), p.num_fragments, p.len_fragment)))
        return tf.data.Dataset.from_generator(lambda: self.generate(p),
                                              output_types=output_types,
                                              output_shapes=output_shapes)
