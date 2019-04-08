'use strict';

function Logger(api) {
  this.API = api;

  this.EventTypes = {
    "ONKEYDOWN": 0,
    "ONKEYUP": 1,
    "ONMOUSEMOVE": 2,
    "ONMOUSEDOWN": 3,
    "ONMOUSEUP": 4,
    "ONMOUSECLICK": 5,
    "ONWHEEL": 6
  };

  this.register = function(doc) {
    doc.addEventListener("mousemove", this.onmousemove.bind(this), true);
    doc.addEventListener("wheel", this.onwheel.bind(this), true);
    doc.addEventListener("mousedown", this.onmousedown.bind(this), true);
    doc.addEventListener("mouseup", this.onmouseup.bind(this), true);
    doc.addEventListener("mouseclick", this.onmouseclick.bind(this), true);
    doc.addEventListener("keydown", this.onkeydown.bind(this), true);
    doc.addEventListener("keyup", this.onkeyup.bind(this), true);
  };

  this.add = function(type, time, value) {
    this.API.send(JSON.stringify({"et": type, "t": time, "v": JSON.stringify(value)}));
  };

  this.getCurrentTime = function() {
    return new Date().getTime();
  };

  this.onkeydown = function(event) {
    this.add(this.EventTypes.ONKEYDOWN, this.getCurrentTime(), event.key);
  };

  this.onkeyup = function(event) {
    this.add(this.EventTypes.ONKEYUP, this.getCurrentTime(), event.key);
  };

  this.onmousemove = function(event) {
    this.add(this.EventTypes.ONMOUSEMOVE, this.getCurrentTime(), {"x": event.pageX, "y": event.pageY});
  };

  this.onmousedown = function(event) {
    this.add(this.EventTypes.ONMOUSEDOWN, this.getCurrentTime(), {"x": event.clientX, "y": event.clientY, "b": event.buttons});
  };

  this.onmouseup = function(event) {
    this.add(this.EventTypes.ONMOUSEUP, this.getCurrentTime(), {"x": event.clientX, "y": event.clientY, "b": event.buttons});
  };

  this.onmouseclick = function(event) {
    this.add(this.EventTypes.ONMOUSECLICK, this.getCurrentTime(), {"x": event.clientX, "y": event.clientY, "b": event.buttons});
  };

  this.onwheel = function(event) {
    this.add(this.EventTypes.ONWHEEL, this.getCurrentTime(), {"x": event.clientX, "y": event.clientY, "dy": event.deltaY,
                                                              "dx": event.deltaX, "dz": event.deltaZ, "m": event.deltaMode});
  };
}
