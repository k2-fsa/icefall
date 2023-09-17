# Introduction

This is a weakly supervised ASR recipe for the LibriSpeech (clean 100 hours) dataset. We training a
conformer model using BTC/OTC with transcripts with synthetic errors. In this README, we will describe
the task and the BTC/OTC training process.

## Task
We propose BTC/OTC to directly train an ASR system leveraging weak supervision, i.e., speech with non-verbatim transcripts.
This is achieved by using a special token to model uncertainties (i.e., substitution errors, insertion errors, and deletion errors) 
within the WFST framework during training.
