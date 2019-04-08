'use strict';

var serverUrl = "ws://localhost:8080/api";
var iframeDoc = "frame";

// Init
document.addEventListener("DOMContentLoaded", function(e) {
	// Init connection
	var api = new ClientAPI(serverUrl);
	api.open();

	window.onbeforeunload = _ => api.close();

	// Init logging
	new Logger(api).register(document);
});
