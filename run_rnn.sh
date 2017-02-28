#!/bin/bash

FEATURES_FOLDER='features/'
FEATURES_PREFIX='trial1'

echo "run.sh >> EXTRACTING FEATURES"
python extract_features.py $FEATURES_PREFIX $FEATURES_FOLDER -y

OUTPUT_PREFIX=$FEATURES_PREFIX
#OUTPUT_PREFIX='trial2'

echo "run.sh >> FITTING AND RUNNING MODEL"
python main_fit_predict_rnn.py $FEATURES_PREFIX $OUTPUT_PREFIX -X 70
#python main_fit_predict_from_model.py $FEATURES_PREFIX $OUTPUT_PREFIX -X 70
