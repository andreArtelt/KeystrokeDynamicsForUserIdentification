'use strict';

function ClientAPI(serverUrl) {
  this.URL = serverUrl;
  this.wsHandler = undefined;
  this.ready = false;

  this.open = function() {
    this.wsHandler = new WebSocket(this.URL);

    this.wsHandler.onopen = function() {
      this.ready = true
    };
    this.wsHandler.onerror = function(err) {
      console.log(err);
      alert("ERROR: Can not connect to server!\nSee console for details.");
    };
    this.wsHandler.onclose = function () {};
  };

  this.close = function() {
    if(this.wsHandler != undefined) {
      this.wsHandler.close();
      this.ready = false;
    }
  };

  this.send = function(data) {
    if(this.wsHandler.readyState != 1) {
      this.ready = false;
      return false;
    }

    this.wsHandler.send(data);

    return true;
  };
}
