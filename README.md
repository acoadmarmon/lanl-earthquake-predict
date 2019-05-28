# lanl-earthquake-predict
My solution to the lanl-earthquake kaggle competition. I transferred this code from my private gitlab repo where I initially worked on it.

## Technology:
I used Tensorflow and Keras to build my model, and Google Cloud's ML Engine to train. Much of the GCP training boilerplate code comes from https://github.com/GoogleCloudPlatform/training-data-analyst/

## Methodology:
I tried many different approaches to train a model to predict when an earthquake will occur given sesmic signals, including:
- Manual Feature Extraction -> Random Forest
- 1D CNN
- 1D CNN feature extractor -> LSTM prediction
- Convolutional LSTM (CNN applied to each step in the LSTM to create a low dimensional feature representation that is passed forward to future LSTM steps)
- Chunked Manual Feature Extraction (100, 1000, 10000 signal example chunks) -> LSTM


## Results:
TBD
