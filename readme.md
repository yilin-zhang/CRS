# Chord Recommendation System

## Preparation
The only thing that should be set up first is the machine-learning-based chord recommendation. The dataset we use is [McGill Billboard dataset](https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1).

The current chord prediction is using third-order Markov chain.
You can adjust the order by changing the variable `MARKOV_ORDER` in `configs.py`.

**If this is the first time running the program, `setup.py` should be executed first.** The program will automatically download and split the McGill Billboard dataset, and create trained machine learning model files. **Execute `python setup.py rnn` for training RNN model, `python setup.py markov` for training Markov chains.** The model files will be placed in `cache/` directory. Run `python setup.py clean` to remove the `cache/` directory.

**For now the trained RNN model files are not properly named, so the program won't work unless you change the value of `RNN_MODEL_PATH` in `configs.py` to the RNN model you are going to use.**

## Running
Run `main.py`, which is the entrance to the whole program. The system is using 5 as step size for chord recommendation. Onset detection is disabled by default.

## Testing and Evaluation
### Chord Detection
Run `chroma_chord_detection.py`.

Change the file name in main in function argument.

### Chord Recommendation
Run `chord_recommendation.py` to get prediction and evaluation.
