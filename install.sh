apt-get install -y python-pip python-matplotlib python-setuptools

pip install numpy scipy

pip install nltk
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

pip install Cython
pip install word2vec

pip install -U scikit-learn

#git clone --recursive https://github.com/dmlc/xgboost
#cd xgboost
#make -j4
#cd python-package
#python setup.py install
