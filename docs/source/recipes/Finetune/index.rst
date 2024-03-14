Fine-tune a pre-trained model
=============================

After pre-training on public available datasets, the ASR model is already capable of
performing general speech recognition with relatively high accuracy. However, the accuracy
could be still low on certain domains that are quite different from the original training
set. In this case, we can fine-tune the model with a small amount of additional labelled
data to improve the performance on new domains.


.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   from_supervised/finetune_zipformer
   adapter/finetune_adapter
