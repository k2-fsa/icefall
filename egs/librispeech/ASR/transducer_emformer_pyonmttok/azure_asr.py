import azure.cognitiveservices.speech as speechsdk
import os

global last
global all_txt
global all_txt_post_processing

last = ""
all_txt = ""
all_txt_post_processing = ""

speech_key, service_region = (
    os.environ.get("AZURE_SPEECH_KEY"),
    "francecentral",
)


def get_stream_azure():
    """gives an example how to use a push audio stream to recognize speech from a custom audio
    source"""
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=service_region,
        speech_recognition_language="fr-FR",
    )

    # setup the audio stream
    stream_azure = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream_azure)

    # instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Connect callbacks to the events fired by the speech recognizer
    global last
    global all_txt
    global all_txt_post_processing

    def te(evt):
        try:
            global last
            last = evt.result.text
            print("last", evt)
        except Exception as e:
            print(e)

    speech_recognizer.recognizing.connect(lambda evt: te(evt))

    def te_recognized(evt):
        global last
        global all_txt
        global all_txt_post_processing
        all_txt += " " + last
        all_txt_post_processing += " " + evt.result.text
        last = ""
        # print(last)
        print("RECOGNIZED: {}".format(evt))

    speech_recognizer.recognized.connect(lambda evt: te_recognized(evt))
    speech_recognizer.session_started.connect(
        lambda evt: print("SESSION STARTED: {}".format(evt))
    )
    speech_recognizer.session_stopped.connect(
        lambda evt: print("SESSION STOPPED {}".format(evt))
    )
    speech_recognizer.canceled.connect(
        lambda evt: print("CANCELED {}".format(evt))
    )

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    return stream_azure, speech_recognizer
