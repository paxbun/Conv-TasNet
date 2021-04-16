from absl import app
from absl import flags
from pathlib import Path

from conv_tasnet.param import ConvTasNetParam
from conv_tasnet.model import ConvTasNet

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", f"{Path.home()}/musdb18", "Dataset Path")
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
flags.DEFINE_integer("THat", 40, "Total number of segments in the input")


def convert_flag_values(flags) -> ConvTasNetParam:
    return ConvTasNetParam(
        N=FLAGS.N, L=FLAGS.L, B=FLAGS.B, H=FLAGS.H, Sc=FLAGS.Sc,
        P=FLAGS.P, X=FLAGS.X, R=FLAGS.R, Ha=FLAGS.Ha, THat=FLAGS.THat
    )


def main(argv):
    param = convert_flag_values(FLAGS)
    model = ConvTasNet.make(param)
    model.summary()


if __name__ == '__main__':
    app.run(main)
