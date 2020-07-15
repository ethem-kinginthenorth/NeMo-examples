# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Tutorial on how to use this script to solve NER task could be found here:
https://nvidia.github.io/NeMo/nlp/intro.html#named-entity-recognition
"""

import math
import numpy as np
import os
import warnings
warnings.simplefilter("ignore")
import nemo
from nemo import logging
from nemo.utils.lr_policies import WarmupAnnealing

import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.data import NemoBertTokenizer
from nemo.collections.nlp.nm.trainables import TokenClassifier
from nemo.backends.pytorch.common.losses import CrossEntropyLossNM, LossAggregatorNM
from nemo.collections.nlp.callbacks.punctuation_capitalization_callback import eval_iter_callback, eval_epochs_done_callback
from nemo.collections.nlp.data.datasets.datasets_utils import calc_class_weights

import torch
import torch.cuda.profiler as profiler
import pyprof
import time
import sys
import datetime

class CustomSimpleLossLoggerCallback(nemo.core.SimpleLossLoggerCallback):
    """
    For callback documentation: please see
    https://nvidia.github.io/NeMo/tutorials/callbacks.html
    """

    def __init__(
        self, tensors, print_func=None, get_tb_values=None, log_to_tb_func=None, step_freq=25, tb_writer=None,
    ):

        super().__init__( tensors)
        if not isinstance(tensors, list):
            tensors = [tensors]
        self._tensors = tensors
        self._print_func = print_func
        self._get_tb_values = get_tb_values
        self._log_to_tb_func = log_to_tb_func
        self._step_freq = step_freq
        self._swriter = tb_writer
        self._start_time = None
        self._last_epoch_start = None
        self._last_iter_start = None

    @property
    def tensors(self):
        return self._tensors

    def on_action_start(self):
        if self.global_rank is None or self.global_rank == 0:
            logging.info("Starting .....")
            self._start_time = time.time()

    def on_action_end(self):
        if self.global_rank is None or self.global_rank == 0:
            if self._swriter is not None:
                self._swriter.close()
            delta = datetime.timedelta(seconds=(time.time() - self._start_time))
            logging.info("Done in %s", delta)

    def on_epoch_start(self):
        if self.global_rank is None or self.global_rank == 0:
            logging.info(f"Starting epoch {self.epoch_num}")
            self._last_epoch_start = time.time()
    def on_epoch_end(self):
        if self.global_rank is None or self.global_rank == 0:
            step = self.step

            delta = datetime.timedelta(seconds=(time.time() - self._last_epoch_start))
            logging.info(f"Finished epoch {self.epoch_num} in {delta}")


    def on_iteration_start(self):
        if self.step == 4:
            profiler.start()
            logging.info(f"********************Starting profiler at step: "+str(self.step))

        if self.global_rank is None or self.global_rank == 0:
            self._last_iter_start = time.time()

    def on_iteration_end(self):

        if self.step == 4:
            profiler.stop()
            logging.info(f"********************Stopping profiler at step: "+str(self.step))

        if self.global_rank is None or self.global_rank == 0:
            step = self.step
            run_time = time.time() - self._last_iter_start
            logging.info(f"Step {self.step} time: {run_time} seconds")




DATA_DIR = "/workspace/data"
WORK_DIR = "/workspace/working"

# See the list of available pre-trained models by calling
# the nemo_nlp.nm.trainables.get_bert_models_list()
PRETRAINED_BERT_MODEL = "bert-base-uncased"

# model parameters
BATCHES_PER_STEP = 1
BATCH_SIZE = 512
CLASSIFICATION_DROPOUT = 0.1
MAX_SEQ_LENGTH = 64
NUM_EPOCHS = 1
LEARNING_RATE = 0.00002
LR_WARMUP_PROPORTION = 0.1
OPTIMIZER = "adam"
STEP_FREQ = 200 # determines how often loss will be printed and checkpoint saved
PUNCT_NUM_FC_LAYERS = 3
NUM_SAMPLES = 5000

# Instantiate neural factory with supported backend
nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,

    # If you're training with multiple GPUs, you should handle this value with
    # something like argparse. See examples/nlp/token_classification.py for an example.
    local_rank=None,

    # If you're training with mixed precision, this should be set to mxprO1 or mxprO2.
    # See https://nvidia.github.io/apex/amp.html#opt-levels for more details.
    #optimization_level="O0",
    optimization_level="O1",
    
    # Define path to the directory you want to store your results
    log_dir=WORK_DIR,

    # If you're training with multiple GPUs, this should be set to
    # nemo.core.DeviceType.AllGpu
    placement=nemo.core.DeviceType.GPU)
	
# If you're using a standard BERT model, you should do it like this. To see the full
# list of BERT/ALBERT/RoBERTa model names, call nemo_nlp.nm.trainables.get_bert_models_list()

tokenizer = NemoBertTokenizer(pretrained_model=PRETRAINED_BERT_MODEL)
bert_model = nemo_nlp.nm.trainables.get_huggingface_model(pretrained_model_name=PRETRAINED_BERT_MODEL)

train_data_layer = nemo_nlp.nm.data_layers.PunctuationCapitalizationDataLayer(
     tokenizer=tokenizer,
     text_file=os.path.join(DATA_DIR, 'text_train.txt'),
     label_file=os.path.join(DATA_DIR, 'labels_train.txt'),
     max_seq_length=MAX_SEQ_LENGTH,
     batch_size=BATCH_SIZE)

punct_label_ids = train_data_layer.dataset.punct_label_ids
capit_label_ids = train_data_layer.dataset.capit_label_ids


# Define classifier for Punctuation and Capitalization tasks
punct_classifier = TokenClassifier(
    hidden_size=bert_model.hidden_size,
    num_classes=len(punct_label_ids),
    dropout=CLASSIFICATION_DROPOUT,
    num_layers=PUNCT_NUM_FC_LAYERS,
    name='Punctuation')

capit_classifier = TokenClassifier(
    hidden_size=bert_model.hidden_size,
    num_classes=len(capit_label_ids),
    dropout=CLASSIFICATION_DROPOUT,
    name='Capitalization')


# If you don't want to use weighted loss for Punctuation task, use class_weights=None
punct_label_freqs = train_data_layer.dataset.punct_label_frequencies
class_weights = calc_class_weights(punct_label_freqs)

# define loss
punct_loss = CrossEntropyLossNM(logits_ndim=3, weight=class_weights)
capit_loss = CrossEntropyLossNM(logits_ndim=3)
task_loss = LossAggregatorNM(num_inputs=2)

input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels = train_data_layer()

hidden_states = bert_model(
    input_ids=input_ids,
    token_type_ids=input_type_ids,
    attention_mask=input_mask)

punct_logits = punct_classifier(hidden_states=hidden_states)
capit_logits = capit_classifier(hidden_states=hidden_states)

punct_loss = punct_loss(
    logits=punct_logits,
    labels=punct_labels,
    loss_mask=loss_mask)

capit_loss = capit_loss(
    logits=capit_logits,
    labels=capit_labels,
    loss_mask=loss_mask)

task_loss = task_loss(
    loss_1=punct_loss,
    loss_2=capit_loss)
	
	
callback_train = CustomSimpleLossLoggerCallback(
    tensors=[task_loss, punct_loss, capit_loss, punct_logits, capit_logits],
    print_func=lambda x: logging.info("Loss: {:.3f}".format(x[0].item())),
    step_freq=STEP_FREQ)

train_data_size = len(train_data_layer)

# If you're training on multiple GPUs, this should be
# train_data_size / (batch_size * batches_per_step * num_gpus)
steps_per_epoch = int(train_data_size / (BATCHES_PER_STEP * BATCH_SIZE))
print ('Number of steps per epoch: ', steps_per_epoch)

# Callback to evaluate the model
#callback_eval = nemo.core.EvaluatorCallback(
#    eval_tensors=[eval_punct_logits,
#    eval_capit_logits,
#    eval_punct_labels,
#    eval_capit_labels,
#    eval_subtokens_mask],
#    user_iter_callback=lambda x, y: eval_iter_callback(x, y),
#    user_epochs_done_callback=lambda x: eval_epochs_done_callback(x,
#                                                      punct_label_ids,
#                                                      capit_label_ids),
#    eval_step=steps_per_epoch)

# Callback to store checkpoints
#ckpt_callback = nemo.core.CheckpointCallback(
#    folder=nf.checkpoint_dir,
#    step_freq=STEP_FREQ)
	
	
lr_policy = WarmupAnnealing(NUM_EPOCHS * steps_per_epoch,
                            warmup_ratio=LR_WARMUP_PROPORTION)

pyprof.init()

with torch.autograd.profiler.emit_nvtx():

    nf.train(tensors_to_optimize=[task_loss],
         callbacks=[callback_train],
         lr_policy=lr_policy,
         batches_per_step=BATCHES_PER_STEP,
         optimizer=OPTIMIZER,
         optimization_params={"num_epochs": NUM_EPOCHS,
                              "lr": LEARNING_RATE})


