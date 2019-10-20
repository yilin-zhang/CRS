# Chord Detection and Prediction

## Chord Detection
Run `chromagram.py`.

## Chord Prediction
Run `chord_prediction.py`.

You should unzip put the [McGill Billboard dataset](https://www.dropbox.com/s/2lvny9ves8kns4o/billboard-2.0-salami_chords.tar.gz?dl=1) in the project directory.
The name of the dataset directory should be `McGill-Billboard`.

The current chord prediction is using third-order Markov chain.
You can adjust the order by changing the variable `MARKOV_ORDER` in `configs.py`.
