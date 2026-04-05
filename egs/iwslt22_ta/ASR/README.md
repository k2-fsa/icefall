# IWSLT_Ta

The IWSLT Tunisian dataset is a 3-way parallel dataset consisting of approximately 160 hours
and 200,000 lines of aligned audio, Tunisian transcripts, and English translations. This dataset
comprises conversational telephone speech recorded at a sampling rate of 8kHz. The train, dev,
and test1 splits of the iwslt2022 shared task correspond to catalog number LDC2022E01. Please
note that access to this data requires an LDC subscription from your institution.To obtain this
dataset, you should download the predefined splits by running the following command:
git clone https://github.com/kevinduh/iwslt22-dialect.git. For more detailed information about
the shared task, please refer to the task paper available at this link:
https://aclanthology.org/2022.iwslt-1.10/.

## Stateless Pruned Transducer Performance Record (after 20 epochs)

|    Decoding method                 |     dev WER     |    test WER    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 47.6      | 51.2       | --epoch 20, --avg 10  |

## Zipformer Performance Record (after 20 epochs)

|    Decoding method                 |     dev WER     |    test WER    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 40.8      | 44.4        | --epoch 20, --avg 10  |


See [RESULTS](RESULTS.md) for details.
