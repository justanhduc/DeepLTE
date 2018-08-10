# DeepLTE
Implementation of the paper "Video Frame Interpolation by Plug-and-Play Deep Locally Temporal Embedding"

## Requirements

[Theano](http://deeplearning.net/software/theano/)

[neuralnet](https://github.com/justanhduc/neuralnet)

## How to run

To reproduce the results in the paper, first pickle the UCF data into pkl files. A sample of the pkl file is provided in [data](https://github.com/justanhduc/DeepLTE/blob/master/data). Then the path to a pkl file in [DeepLTE_UCF.config](https://github.com/justanhduc/DeepLTE/blob/master/DeepLTE_UCF.config). Feel free to change any parameters in the config file. After that, execute

```
python interpolate_UCF.py DeepLTE_UCF.config (--gpu 0)
```

To interpolate frames for a video sequence, first extract all frames into a folder. Then specify the path in [DeepLTE.config](https://github.com/justanhduc/DeepLTE/blob/master/DeepLTE.config). Feel free to change any parameters in the config file. After that, execute

```
python interpolate.py DeepLTE.config (--gpu 0)
```

## Results

To be available.
