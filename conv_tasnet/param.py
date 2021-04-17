class ConvTasNetParam:
    """Contains hyperparameters.

    Attributes:
        N: Number of filters in autoencoder (p. 1260)
        L: Length of the filters in samples (p. 1260)
        B: Number of channels in bottleneck and the residual paths' 1x1-conv blocks (p. 1260)
        H: Number of channels in convolutional blocks (p. 1260)
        Sc: Number of channels in skip-connection paths' 1x1-conv blocks (p. 1260)
        P: Kernel size in convolutional blocks (p. 1260)
        X: Number of convolutional blocks in each repeat (p. 1260)
        R: Number of repeats (p. 1260)
        Ha: Activation function used in encoder ("relu" or "linear"; p. 1258)
        THat: Total number of segments in the input (p. 1257)
        C: Number of speakers (p. 1258)
        epsilon: Small constant for numerical stability (p. 1259)
        overlap: Number of samples in which each adjacent pair of fragments overlap
    """

    __slots__ = 'N', 'L', 'B', 'Sc', 'H', 'P', 'X', 'R', 'Ha', 'THat', 'C', 'epsilon', 'overlap'

    def __init__(self,
                 N: int = 512,
                 L: int = 16,
                 B: int = 128,
                 H: int = 512,
                 Sc: int = 128,
                 P: int = 3,
                 X: int = 8,
                 R: int = 3,
                 Ha: str = "linear",
                 THat: int = 128,
                 C: int = 4,
                 epsilon: float = 1e-8,
                 overlap: int = 8):
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.Sc = Sc
        self.P = P
        self.X = X
        self.R = R
        self.Ha = Ha
        self.THat = THat
        self.C = C
        self.epsilon = epsilon
        self.overlap = overlap

    def get_config(self) -> dict:
        return {
            "N": self.N,
            "L": self.L,
            "B": self.B,
            "H": self.H,
            "Sc": self.Sc,
            "P": self.P,
            "X": self.X,
            "R": self.R,
            "Ha": self.Ha,
            "THat": self.THat,
            "C": self.C,
            "epsilon": self.epsilon,
            "overlap": self.overlap
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf8") as f:
            f.write('\n'.join(f"{key}={value}" for key,
                              value in self.get_config()))

    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf8") as f:
            return ConvTasNetParam(**dict(line.split('=')
                                          for line in f.readlines()))

    def __str__(self) -> str:
        return str(self.get_config())
