// server
'use strict';
var fs = require('fs'),
  express = require('express'),
  bodyParser = require('body-parser'),
  http_module = require('http'),
  socket = require('socket.io'),
  path = require('path'),
  app = express(),
  http = http_module.Server(app);
app.set('port', process.env.PORT || 8080);

var mode = ['switch', 'rate', 'move'];

// reading json files
//var currentPath = process.cwd();
//var dataFolder = currentPath + '/data/';
//var aboutJSON = JSON.parse(fs.readFileSync(dataFolder + 'about.json', 'utf8'));
//var blogJSON = JSON.parse(fs.readFileSync(dataFolder + 'blog.json', 'utf8'));
//var labJSON = JSON.parse(fs.readFileSync(dataFolder + 'lab.json', 'utf8'));
//var cvJSON = JSON.parse(fs.readFileSync(dataFolder + 'cv.json', 'utf8'));

// View Engine
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

// Set Static Fold
app.use('/', express.static(__dirname + '/public/'));

// Define routes
app.get('/', function(req, res) {
  res.render('viewer');
});
app.get('/host', function(req, res) {
  res.render('host');
});
app.get('*', function(req, res) {
  res.render('404');
});

// load page
var server = http.listen(app.get('port'), () => {
  console.info('==> 🌎  Go to http://localhost:%s', app.get('port'));
});

// start holders
var message = {
  display: 0,
  switch: 1,
  rate: 3,
  move: 0
};

// web socket
var io = socket(server);
io.on('connection', function(socket) {
  console.log('made socket connection');

  io.sockets.emit('update', message);
  console.log('data sent to client (Fresh)');

  socket.on('stream', function(data) {
    socket.broadcast.emit('stream', data);
  });

  socket.on('startBtn', function(data) {
    if (data.switch == 1) console.log('OFF');
    else if (data.switch == 0) console.log('ON');
    message.switch = data.switch;
    var spawn = require('child_process').spawn;
    var process = spawn('python', [
      './controller.py', // program name
      mode[0],
      data.switch // action
    ]);
    process.stdout.on('data', function(data) {
      console.log(data.toString()); // get the print in the python program
    });
    io.sockets.emit('update', message);
    console.log('data sent to client (startBtn)');
  });

  socket.on('rate', function(data) {
    console.log('rate =', data.rate);
    message.rate = data.rate;
    var spawn = require('child_process').spawn;
    var process = spawn('python', [
      './controller.py', // program name
      mode[1],
      data.rate // action
    ]);
    process.stdout.on('data', function(data) {
      console.log(data.toString()); // get the print in the python program
    });
    io.sockets.emit('update', message);
    console.log('data sent to client (rate)');
  });

  socket.on('move', function(data) {
    if (data.move == 1) console.log('FORWARD', data.move);
    else if (data.move == 2) console.log('BACKWARD', data.move);
    else if (data.move == 3) console.log('LEFT', data.move);
    else if (data.move == 4) console.log('RIGHT', data.move);
    message.move = data.move;
    var spawn = require('child_process').spawn;
    var process = spawn('python', [
      './controller.py', // program name
      mode[2],
      data.move // action
    ]);
    process.stdout.on('data', function(data) {
      console.log(data.toString()); // get the print in the python program
    });
    io.sockets.emit('update', message);
    console.log('data sent to client (move)');
  });
});
