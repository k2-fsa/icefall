// see https://mdn.github.io/web-dictaphone/scripts/app.js
// and https://gist.github.com/meziantou/edb7217fddfbb70e899e

const record = document.getElementById('record');
const stop = document.getElementById('stop');
const soundClips = document.getElementById('sound-clips');
const canvas = document.getElementById('canvas');

soundClips.innerHTML = "hello";

stop.disabled = true;

let audioCtx;
const canvasCtx = canvas.getContext("2d");

let sampleRate;

if (navigator.mediaDevices.getUserMedia) {
  console.log('getUserMedia supported.');

  // see https://w3c.github.io/mediacapture-main/#dom-mediadevices-getusermedia
  const constraints = {
    // does not work
    // audio : {sampleRate : 16000, sampleSize : 16, channelCount : 1}
    audio : true,
  };
  let chunks = [];

  let onSuccess = function(stream) {
    var settings = stream.getAudioTracks()[0].getSettings();
    sampleRate = settings.sampleRate;
    console.log(settings);
    console.log('sample rate ' + settings.sampleRate);
    console.log('channel count ' + settings.channelCount);
    console.log('sample size ' + settings.sampleSize);
    const mediaRecorder = new MediaRecorder(stream);
    console.log('mime type ' + mediaRecorder.mimeType);
    console.log('audio bits per second ' + mediaRecorder.audioBitsPerSecond);
    console.log(mediaRecorder)

    visualize(stream);

    record.onclick = function() {
      mediaRecorder.start(10); // 10ms period to send data
      console.log(mediaRecorder.state);
      console.log("recorder started");
      record.style.background = "red";

      stop.disabled = false;
      record.disabled = true;
    };

    stop.onclick = function() {
      mediaRecorder.stop();
      console.log(mediaRecorder.state);
      console.log("recorder stopped");
      record.style.background = "";
      record.style.color = "";
      // mediaRecorder.requestData();

      stop.disabled = true;
      record.disabled = false;
    };

    mediaRecorder.onstop = function(e) {
      console.log("data available after MediaRecorder.stop() called.");

      const clipName =
          prompt('Enter a name for your sound clip?', 'My unnamed clip');

      const clipContainer = document.createElement('article');
      const clipLabel = document.createElement('p');
      const audio = document.createElement('audio');
      const deleteButton = document.createElement('button');

      clipContainer.classList.add('clip');
      audio.setAttribute('controls', '');
      deleteButton.textContent = 'Delete';
      deleteButton.className = 'delete';

      if (clipName === null) {
        clipLabel.textContent = 'My unnamed clip';
      } else {
        clipLabel.textContent = clipName;
      }

      clipContainer.appendChild(audio);
      clipContainer.appendChild(clipLabel);
      clipContainer.appendChild(deleteButton);
      soundClips.appendChild(clipContainer);

      audio.controls = true;
      const blob = new Blob(chunks, {'type' : 'audio/ogg; codecs=opus'});
      chunks = [];
      const audioURL = window.URL.createObjectURL(blob);
      audio.src = audioURL;
      console.log("recorder stopped");

      deleteButton.onclick =
          function(e) {
        let evtTgt = e.target;
        evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode);
      }

          clipLabel.onclick = function() {
        const existingName = clipLabel.textContent;
        const newClipName = prompt('Enter a new name for your sound clip?');
        if (newClipName === null) {
          clipLabel.textContent = existingName;
        } else {
          clipLabel.textContent = newClipName;
        }
      }
    };

    mediaRecorder.ondataavailable = function(e) {
      console.log('size ' + e.data.size);
      console.log(e.data);
      chunks.push(e.data);
    }
  };

  let onError = function(
      err) { console.log('The following error occured: ' + err); };

  navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);

} else {
  console.log('getUserMedia not supported on your browser!');
}

function visualize(stream) {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }

  const source = audioCtx.createMediaStreamSource(stream);

  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  source.connect(analyser);
  // analyser.connect(audioCtx.destination);

  draw()

  function draw() {
    const WIDTH = canvas.width
    const HEIGHT = canvas.height;

    requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgb(200, 200, 200)';
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

    canvasCtx.beginPath();

    let sliceWidth = WIDTH * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {

      let v = dataArray[i] / 128.0;
      let y = v * HEIGHT / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }

      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }
}

window.onresize = function() { canvas.width = mainSection.offsetWidth; };

window.onresize();
