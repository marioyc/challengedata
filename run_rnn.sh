#!/bin/bash

FEATURES_FOLDER='features/'
FEATURES_PREFIX='trial1'

# if the features have already been extracted, these two lines can be commented out
echo "run.sh >> EXTRACTING FEATURES"
#python extract_features.py $FEATURES_PREFIX $FEATURES_FOLDER

OUTPUT_PATH='results/'$FEATURES_PREFIX'.csv'

echo "run.sh >> FITTING AND RUNNING MODEL"
python main_fit_predict_rnn.py $FEATURES_FOLDER $FEATURES_PREFIX $OUTPUT_PATH -X 70
