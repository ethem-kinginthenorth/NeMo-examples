
# NeMo examples
This small project has some NeMo examples that covers some of the features including mixed precision training. It also features profiling. I split this project into two folds: 
1) end-to-end ASR with NeMo:
    - [1_ASR_turotial_using_NeMo](notebooks/1_ASR_tutorial_using_NeMo.ipynb): This is the main ASR tutorial starting with introducing the data and the model to be used. Then it explains how to utilize NeMo modules to build training and test DAGs. Next, it further trains an existing model and evaluates it
    - [1_ASR_tutorial_using_NeMo_mp.ipynb](notebooks/1_ASR_tutorial_using_NeMo_mp.ipynb): This is almost the same notebook with the previous one. Except with two additions. One of them is, it has a data augmentation step in between feature extraction (from audio file to melspectrum) and encoder. The other one is enabling mixed precision training. 
2) punctuation and capitalization on top of BERT with NeMo.
    - [PunctuationWithBERT.ipynb](notebooks/PunctuationWithBERT.ipynb): This is another notebook to showcase NeMo. Here, I build a puntuation and capitalization classifications on top of a BERT model.
    - [punctuation_pyprof_o0.py](notebooks/punctuation_pyprof_o0.py): I incepted profiler commands in the code and implemented a custom call back function to profile just one step rather than the entire epoch. Further, this notebook also uses fp32.
    - [punctuation_pyprof_o1.py](notebooks/punctuation_pyprof_01.py): The same as above except I use mixed precision training here.
    - [run_pyprof.sh](notebooks/run_pyprof.sh): This file runs pyprof creates several files. At the completion of this call, you can type `head -n 10 out_o0_b512.sorted` to see the top time consuimg kernel operations for fp32 training and `head -n 10 out_o1_b512.sorted` to see top time consuming kernel operations for mixed precision training.
    - [run_dlprof.sh](notebooks/run_dlprof.sh): This file runs dlprof on top of pyprof and at the completion of this call we should be able to visualize profiling results in tensorboard if you are using the container stemmed from the `dockerfile.nemo_dlprof` file.

These notebooks are derived from the NeMo [repo](https://github.com/NVIDIA/NeMo)


