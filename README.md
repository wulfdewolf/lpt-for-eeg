# Pretrained transformers as EEG decoders

This repository contains a framework that allows for running a number of EEG classification tasks, each time using two types of models.
The first type of models consists of traditional EEG decoding models, while the second type of models are huge NLP models.
Research by [Lu et al.](https://arxiv.org/abs/2103.05247) has shown that these huge NLP models, or _Pretrained Transformers (PT)_, that have been trained on natural language can be generalized to other modalities with minimal finetuning.

The framework in this repository, which is the code companion to a master's thesis at the [VUB](https://www.vub.be/), was created to verify if said PT have any adaptability in the EEG decoding field.
