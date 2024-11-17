# ChessSL
Chess engine trained with supervised learning using Leela Chess Zero T80 data.

As of 17/11/2024 on lichess.org, the engine has a bullet rating of 2070 and a blitz rating of 2160, which is stronger than 87.1% and 95.4% of the players respectively.
(https://lichess.org/@/chesssl_bot) 

## Installation
Python version 3.10 or higher is required to run the engine.

To install the required packages, run
```
pip3 install -r requirements.txt
```
or install the following packages manually:
```
numpy==1.26.4
torch==2.4.0
```
Then, compile the dynamic-linked library by 
```
make [CC=gcc]
```
If you don't have a C compiler, the precompiled library is already included in the repository,
but they are not guaranteed to work on your system.

## Universal Chess Interface (UCI)
Due to time constraint, only the bare minimal commands are implemented to work with the lichess-bot API (https://github.com/lichess-bot-devs/lichess-bot).

To run the engine, simply run
```
python3 uci.py
```

Example usage:
```
python3 uci.py

>>> ChessSL version 0.0.1

<<< uci
>>> id name ChessSL
>>> id author Ching-Yin Ng
>>>
>>> option name UCI_ShowWDL type check default false
>>> uciok

<<< isready
>>> readyok

<<< ucinewgame

<<< position startpos moves e2e4 e7e5 g1f3

<<< go wtime 300000 btime 300000 
>>> bestmove b8c6

# If `print_tree_info` is set to true in `config.yaml`
# For this example, I have set the number of nodes to 1024 in `config.yaml`
<<< go wtime 1000 btime 1000 
>>> info depth 1 wdl 316 339 345 nps 46 nodes 4 time 64 pv b8c6 
>>> info depth 2 wdl 374 310 317 nps 136 nodes 32 time 205 pv b8c6 f1c4 
>>> info depth 3 wdl 300 285 415 nps 1322 nodes 64 time 24 pv b8c6 f3e5 c6e5 
>>> info depth 4 wdl 303 286 411 nps 907 nodes 77 time 14 pv b8c6 f3e5 c6e5 d2d4 
>>> info depth 5 wdl 316 277 406 nps 1101 nodes 93 time 14 pv b8c6 f3e5 c6e5 d2d4 e5g6 
>>> info depth 7 wdl 332 334 333 nps 2161 nodes 317 time 14 pv b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 
>>> info depth 8 wdl 328 339 332 nps 2125 nodes 349 time 15 pv b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 b1d2 
>>> info depth 9 wdl 306 369 325 nps 2230 nodes 957 time 14 pv b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 a2a4 a7a5 
>>> info depth 9 wdl 303 371 325 nps 346 nodes 1024 time 8 pv b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 a2a4 a7a5 
>>> bestmove b8c6

<<< debug
>>> White board:
>>> [[-4 -2 -3 -5 -6 -3 -2 -4]
>>>  [-1 -1 -1 -1  0 -1 -1 -1]
>>>  [ 0  0  0  0  0  0  0  0]
>>>  [ 0  0  0  0 -1  0  0  0]
>>>  [ 0  0  0  0  1  0  0  0]
>>>  [ 0  0  0  0  0  2  0  0]
>>>  [ 1  1  1  1  0  1  1  1]
>>>  [ 4  2  3  5  6  3  0  4]]
>>> Black board:
>>> [[-4  0 -3 -6 -5 -3 -2 -4]
>>>  [-1 -1 -1  0 -1 -1 -1 -1]
>>>  [ 0  0 -2  0  0  0  0  0]
>>>  [ 0  0  0 -1  0  0  0  0]
>>>  [ 0  0  0  1  0  0  0  0]
>>>  [ 0  0  0  0  0  0  0  0]
>>>  [ 1  1  1  0  1  1  1  1]
>>>  [ 4  2  3  6  5  3  2  4]]
```


## Trained model
The trained model is available in the `trained_models` folder.
It is a ResNet + CBAM model with 20 residual blocks and 256 filters (See `model.py`).
I have trained it for 40 epochs with 2843611 T80 data (`lr = 0.001` for 30 epochs and `lr=0.0001` for 10 epochs). Each epochs took around 2500 seconds on a single RTX2070.

The final loss is as follows:
```
Policy Loss: 2.08 | Policy Loss (MSE): 0.000109 | Winner Loss: 0.804 | MLH Loss: 1.61 | Total Weighted Loss: 4.5
```
The policy MSE loss is for reference only. It is not used to calculate the total loss.

## Training
To train your own model, you first need to download the training data. Any data should be fine as long as it is in the v6 format of the Leela Chess Zero data. 
Once they are downloaded, unzip them and put them together in a folder. (All the files should end up with the `.gz` extension)
Then, make sure to edit the `config.yaml` file.

Now, you can start training by running the `training.py` script.
```
python3 training.py
```
A log file will be created in the `chessSL_training_logs` folder.