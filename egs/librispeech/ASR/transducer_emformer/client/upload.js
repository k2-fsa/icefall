/**
References
https://developer.mozilla.org/en-US/docs/Web/API/FileList
https://developer.mozilla.org/en-US/docs/Web/API/FileReader
https://javascript.info/arraybuffer-binary-arrays
https://developer.mozilla.org/zh-CN/docs/Web/API/WebSocket
https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/send
*/

var socket;
function initWebSocket() {
  socket = new WebSocket("ws://localhost:6008/");

  // Connection opened
  socket.addEventListener(
      'open',
      function(event) { document.getElementById('file').disabled = false; });

  // Connection closed
  socket.addEventListener('close', function(event) {
    document.getElementById('file').disabled = true;
    initWebSocket();
  });

  // Listen for messages
  socket.addEventListener('message', function(event) {
    document.getElementById('results').innerHTML = event.data;
    console.log('Received message: ', event.data);
  });
}

function onFileChange() {
  var files = document.getElementById("file").files;

  if (files.length == 0) {
    console.log('No file selected');
    return;
  }

  console.log('files: ' + files);

  const file = files[0];
  console.log(file);
  console.log('file.name ' + file.name);
  console.log('file.type ' + file.type);
  console.log('file.size ' + file.size);

  let reader = new FileReader();
  reader.onload = function() {
    let view = new Uint8Array(reader.result);
    console.log('bytes: ' + view.byteLength);
    // we assume the input file is a wav file.
    // TODO: add some checks here.
    let body = view.subarray(44);
    socket.send(body);
    socket.send(JSON.stringify({'eof' : 1}));
  };

  reader.readAsArrayBuffer(file);
}
