# Chord Detection and Recommendation

## Preparation
The only thing that should be set up first is the machine-learning-based chord recommendation.

The dataset we use is [McGill Billboard dataset](https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1). After downloading it, create two directories `mcgill-train/` and `mcgill-test/` in the current project directory. Split the dataset manually, put all the training data into `mcgill-train/`, all the testing data into `mcgill-test/`.

The current chord prediction is using third-order Markov chain.
You can adjust the order by changing the variable `MARKOV_ORDER` in `configs.py`.

If this is the first time running the program, run `setup.py` first. The program will create trained machine learning model files. Execute `python setup.py rnn` for training RNN model, `python setup.py markov` for training Markov chains. The model files will be placed in `cache/` directory. Run `python setup.py clean` to remove the `cache/` directory.

**For now the trained RNN model files are not properly named, so the program won't work unless you change the value of `RNN_MODEL_PATH` in `configs.py` to the RNN model you are going to use.**

## Running
Run `main.py`, which is the entrance to the whole program.

## Testing and Evaluation
### Chord Detection
Run `chroma_chord_detection.py`.

Change the file name in main in function argument.

### Chord Recommendation
Run `chord_recommendation.py` to get prediction and evaluation.

# References

## Chord Detection

1. Automatic Chord Recognition from Audio Using Enhanced Pitch Class Profile, https://pdfs.semanticscholar.org/30a9/0af7c214f423743472e0c82f2b5332ccb55f.pdf


2. Realtime Chord Recognition of Musical Sound: a System Using Common Lisp Music,
https://quod.lib.umich.edu/i/icmc/bbp2372.1999.446/1

## Chord Recommendation

1. https://link.springer.com/article/10.1007/s11042-016-3984-z (Possibly, not completely sure. Just skimmed through it)
