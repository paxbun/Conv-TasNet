from absl import app
from absl import flags
from pathlib import Path
from os import path, listdir

from conv_tasnet.param import ConvTasNetParam
from conv_tasnet.model import ConvTasNet
from dataset import DatasetParam, Provider

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", None,
                    "Directory to save weights", required=True)
flags.DEFINE_string("dataset_path", f"{Path.home()}/musdb18", "Dataset Path")
flags.DEFINE_integer("epochs", None, "Number of epochs to repeat")
flags.DEFINE_integer(
    "num_songs", 5, "Number of songs to get samples from for each epoch")
flags.DEFINE_integer("num_samples", 100, "Number of samples ")
flags.DEFINE_integer("max_decoded", 100,
                     "The maximum number of decoded songs in the memory")
flags.DEFINE_integer("repeat", 400, "Number of batches for each epoch")
flags.DEFINE_integer("N", 512, "Number of filters in autoencoder")
flags.DEFINE_integer("L", 16, "Length of the filters in samples")
flags.DEFINE_integer(
    "B", 128, "Number of channels in bottleneck and the residual paths' 1x1-conv blocks")
flags.DEFINE_integer("H", 512, "Number of channels in convolutional blocks")
flags.DEFINE_integer(
    "Sc", 128, "Number of channels in skip-connection paths' 1x1-conv blocks")
flags.DEFINE_integer("P", 3, "Kernel size in convolultional blocks")
flags.DEFINE_integer("X", 8, "Number of convolutional blocks in each repeat")
flags.DEFINE_integer("R", 3, "Number of repeats")
flags.DEFINE_string("Ha", "linear", "Activation function used in the encoder")
flags.DEFINE_integer("THat", 128, "Total number of segments in the input")
flags.DEFINE_integer(
    "overlap", 8, "Number of samples in which each adjacent pair of fragments overlap")


def get_model_param() -> ConvTasNetParam:
    return ConvTasNetParam(
        N=FLAGS.N, L=FLAGS.L, B=FLAGS.B, H=FLAGS.H, Sc=FLAGS.Sc,
        P=FLAGS.P, X=FLAGS.X, R=FLAGS.R, Ha=FLAGS.Ha, THat=FLAGS.THat,
        C=len(Provider.STEMS),
        overlap=FLAGS.overlap
    )


def get_dataset_param() -> DatasetParam:
    return DatasetParam(
        num_songs=FLAGS.num_songs, num_samples=FLAGS.num_samples,
        num_fragments=FLAGS.THat, len_fragment=FLAGS.L,
        repeat=FLAGS.repeat,
        overlap=FLAGS.overlap
    )


def main(argv):
    model = ConvTasNet.make(get_model_param())
    dataset = Provider(FLAGS.dataset_path, max_decoded=FLAGS.max_decoded)
    checkpoint_dir = FLAGS.checkpoint

    epoch = 0
    if path.exists(checkpoint_dir):
        checkpoints = [name for name in listdir(
            checkpoint_dir) if "ckpt" in name]
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split(".")[0]
        epoch = int(checkpoint_name) + 1
        model.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    epochs_to_inc = FLAGS.epochs
    while epochs_to_inc == None or epochs_to_inc > 0:
        print(f"Epoch: {epoch}")
        history = model.fit(dataset.make_dataset(get_dataset_param()))
        model.save_weights(f"{checkpoint_dir}/{epoch:05d}.ckpt")
        epoch += 1
        if epochs_to_inc != None:
            epochs_to_inc -= 1
        model.param.save(f"{checkpoint_dir}/config.txt")
        model.save(f"{checkpoint_dir}/model")


if __name__ == '__main__':
    app.run(main)
