# Chord Detection and Prediction

## Chord Detection
Run `chroma_chord_detection.py`.

Change the file name in main in function argument.


## Chord Recommendation

The dataset we use is [McGill Billboard dataset](https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1). After downloading it, create two directories `mcgill-train` and `mcgill-test` in the current project directory. Split the dataset manually, put all the training data into `mcgill-train`, all the testing data into `mcgill-test`.

The current chord prediction is using third-order Markov chain.
You can adjust the order by changing the variable `MARKOV_ORDER` in `configs.py`.

Run `chord_recommendation.py`.

# References

## Chord Detection

1. Automatic Chord Recognition from Audio Using Enhanced Pitch Class Profile, https://pdfs.semanticscholar.org/30a9/0af7c214f423743472e0c82f2b5332ccb55f.pdf


2. 

## Chord Recommendation

1. 
