# Multi-Resolution-GAIL

Code is written with PyTorch v0.3 (Python 3.6.4).

Dataset is available from [STATS](https://www.stats.com/data-science/). A pre-processed version is available [here](https://drive.google.com/open?id=1zhj4jJsHomXoXIVhdVFpnt3TXt-qCvdU). Download the data into the `bball_data/data/` folder.

## Model Structure: 

For basketball game trajectory prediction, it is a five-level model.

`5th level` predicts every 16 steps. For a trajectory of 50 steps, it basically predicts the 1/3 point, 2/3 point and the final position.

`4th  level` predicts every 8 steps. For any start point, it predicts the mid point given the end point predicted by the previous model, then it copies the end point from the previous result.

`3rd level` predicts every 4 steps.

`2nd level` predicts every 2 steps.

`1st level` predicts every step.

## Usage:

As with the structure of the model, the training consists of 5 stages: 

Edit `train_model.sh` and run that script. 

`--subsample` controls the level of the model it is going to train. `16` is the 5th level, `8` is the 4th level, `4` is the 3rd level, `2` is the second level, `1` is the first level.

To train the full model from scratch, basically need to first set `--subsample` to 16 to train the 5th level, then set it to 8 to train the 4th level, then 3rd, 2nd, and 1st level.

`test_model.py` can be used to test the fully trained model. A trained model is provided with the trial number 100.
