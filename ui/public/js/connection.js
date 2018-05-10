// make connection
var socket = io.connect('http://localhost:8080');

// query DOM
var liveFeed = document.getElementById('live-feed'),
  startBtn = document.forms[0],
  onBtn = document.getElementById('on-btn'),
  offBtn = document.getElementById('off-btn'),
  accuractionRate = document.forms[1],
  moveFront = document.getElementById('move-front'),
  moveBack = document.getElementById('move-back'),
  moveLeft = document.getElementById('move-left'),
  moveRight = document.getElementById('move-right');

console.log(startBtn.on);
// emit events
startBtn.on.addEventListener('click', function() {
  console.log('sent value (on)', this.value);
  socket.emit('startBtn', {
    switch: this.value
  });
  console.log('data sent to server (startBtn)');
});
startBtn.off.addEventListener('click', function() {
  console.log('sent valu (off)', this.value);
  socket.emit('startBtn', {
    switch: this.value
  });
  console.log('data sent to server (startBtn)');
});
accuractionRate.rate.addEventListener('click', function() {
  socket.emit('rate', {
    rate: 1
  });
  console.log('data sent to server (rate)');
});
moveFront.addEventListener('click', function() {
  socket.emit('move', {
    move: 1
  });
  console.log('data sent to server (moveFront)');
});
moveFront.addEventListener('click', function() {
  socket.emit('move', {
    move: 1
  });
  console.log('data sent to server (moveFront)');
});
moveBack.addEventListener('click', function() {
  socket.emit('move', {
    move: 2
  });
  console.log('data sent to server (moveBack)');
});
moveLeft.addEventListener('click', function() {
  socket.emit('move', {
    move: 3
  });
  console.log('data sent to server (moveLeft)');
});
moveRight.addEventListener('click', function() {
  socket.emit('move', {
    move: 4
  });
  console.log('data sent to server (moveRight)');
});

socket.on('update', function(data) {
  // set on and off button
  //console.log('data reseved from server', data.switch);
  onBtn.checked = false;
  offBtn.checked = false;
  if (data.switch == 1) offBtn.checked = true;
  else if (data.switch == 0) onBtn.checked = true;

  // set accuractionRate

  accuractionRate.value = data.rate;
});
