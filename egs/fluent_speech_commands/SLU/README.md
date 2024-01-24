## Fluent Speech Commands recipe

This is a recipe for the Fluent Speech Commands dataset, a speech dataset which transcribes short utterances (such as "turn the lights on in the kitchen") into action frames (such as {"action": "activate", "object": "lights", "location": "kitchen"}). The training set contains 23,132 utterances, whereas the test set contains 3793 utterances. 

Dataset Paper link: <https://paperswithcode.com/dataset/fluent-speech-commands>

cd icefall/egs/fluent_speech_commands/
Training: python transducer/train.py
Decoding: python transducer/decode.py