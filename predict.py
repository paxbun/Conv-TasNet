# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import youtube_dl

from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir

from conv_tasnet.param import ConvTasNetParam
from conv_tasnet.model import ConvTasNet
from dataset import Provider

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None,
                    "Directory containing saved weights", required=True)
flags.DEFINE_string("video_id", None, "YouTube video ID", required=True)
flags.DEFINE_bool("interpolate", False,
                  "Interpolate overlapping part of each rows")


def youtube_dl_hook(d):
    if d["status"] == "finished":
        print("Done downloading...")


def main(argv):
    checkpoint_dir = FLAGS.checkpoint
    if not path.exists(checkpoint_dir):
        raise ValueError(f"'{checkpoint_dir}' does not exist")

    checkpoints = [name for name in listdir(checkpoint_dir) if "ckpt" in name]
    if not checkpoints:
        raise ValueError(f"No checkpoint exists")
    checkpoints.sort()
    checkpoint_name = checkpoints[-1].split(".")[0]

    param = ConvTasNetParam.load(f"{checkpoint_dir}/config.txt")
    model = ConvTasNet.make(param)
    model.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    video_id = FLAGS.video_id

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "44100",
        }],
        "outtmpl": "%(title)s.wav",
        "progress_hooks": [youtube_dl_hook],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_id, download=False)
        status = ydl.download([video_id])

    title = info.get("title", None)
    filename = title + ".wav"
    audio, sr = librosa.load(filename, sr=44100, mono=True)

    num_samples = audio.shape[0]
    num_portions = (num_samples - param.overlap) // (param.THat *
                                                     (param.L - param.overlap))
    num_samples_output = num_portions * param.THat * (param.L - param.overlap)
    num_samples = num_samples_output + param.overlap

    if FLAGS.interpolate:
        def filter_gen(n):
            if n < param.overlap:
                return n / param.overlap
            elif n > param.L - param.overlap:
                return (param.L - n) / param.overlap
            else:
                return 1
        output_filter = np.array([filter_gen(n) for n in range(param.L)])

    print("predicting...")

    audio = audio[:num_samples]
    model_input = np.zeros((num_portions, param.THat, param.L))
    for i in range(num_portions):
        for j in range(param.THat):
            begin = (i * param.THat + j) * (param.L - param.overlap)
            end = begin + param.L
            model_input[i][j] = audio[begin:end]
    separated = model.predict(model_input)
    separated = np.transpose(separated, (1, 0, 2, 3))

    if FLAGS.interpolate:
        separated = output_filter * separated
        overlapped = separated[:, :, :, (param.L - param.overlap):]
        overlapped = np.pad(
            overlapped,
            pad_width=((0, 0), (0, 0), (0, 0),
                       (0, param.L - 2 * param.overlap)),
            mode="constant",
            constant_values=0)
        overlapped = np.reshape(overlapped, (param.C, num_samples_output))
        overlapped[:, 1:] = overlapped[:, :-1]
        overlapped[:, 0] = 0

    separated = separated[:, :, :, :(param.L - param.overlap)]
    separated = np.reshape(separated, (param.C, num_samples_output))

    if FLAGS.interpolate:
        separated += overlapped

    print("saving...")

    for idx, stem in enumerate(Provider.STEMS):
        sf.write(f"{title}_{stem}.wav", separated[idx], sr)


if __name__ == '__main__':
    app.run(main)
