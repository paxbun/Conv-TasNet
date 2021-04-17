# Conv-TasNet

Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation," in *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 27, no. 8, pp. 1256-1266, Aug. 2019, doi: 10.1109/TASLP.2019.2915167.

# Training

To train the model, run:
```
python main.py --checkpoint=checkpoint --dataset_path=/path/to/MUSDB18
```

# Prediction

To separate audio using the trained model, run:
```
python predict.py --checkpoint=checkpoint --video_id=YOUTUBE_VIDEO_ID_HERE
```