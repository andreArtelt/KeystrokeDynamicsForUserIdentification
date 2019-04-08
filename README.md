# README

## About

Code for the *master project* "ISY-Project: Machine Learning for user identification on websites based on their mouse and keyboard usage", carried out in the summer term 2018 as part of the *degree program "Intelligent Systems (M.Sc.)"* at the university of Bielefeld (faculty of technology).

**Team members**
- André Artelt
- Jonathan Jakob
- Valerie Vaquet

## Conference

This project was presented as a poster at [Interdisciplinary College 2019]([http://www.interdisciplinary-college.de/). Details can be found in `abstract.pdf` and `poster.pdf`.

## Description

### Official project proposal

The goal of the project is to investigate in how far it is possible to identify the user
of a website based on tracked signals from keyboard and mouse only. If so, such technologies
could be used to increase the safety of web usage (e.g. match user login to signal profile, identify
possible hazardeous anonymous users in terms of their signal fingerprint). The problem itself
is a classical problem within machine learning, where the task is to design a test environment, gather
test data, design a suitable feature representation, identify suitable ML pipelines for a valid classification,and evaluate the results in realistic scenarios.


## Requirements

- python 2.7
- pip
- chromium-webbrowser

## How to use

### Setup your machine

Given that your system fulfills the requirements from above, all you have to do is installing the python packages (dependencies). You can do so by running `pip install -r REQUIREMENTS.txt` from the root directory.<br>
**Note:** You might want to setup/use a virtual environment in order to avoid conflicts with your local configuration!

## Server

The server (located in **BackEnd**) is written in python.

You can start the server by running `python main.py` from within the *BackEnd* directory.<br>
The script hosts a websocket server and writes all incoming data into a **sqlite database**.

If you do not specify any arguments, the server is going to listen on **port 8080** and write all incoming messages/events into the database **storage.db**.<br>
You can change the port by editing the global variable *port* in main.py.<br>
If you want to specify/change the file name of the database, you can do so by providing it as an argument to main.py (e.g. `python main.py ../Data/User01.db` for using the database in ../Data/User01.db).

### Database design

The received data (**json**) along with a timestamp and a type identifier (**integer**) is stored in a table having the following structure<br>

| type:INTEGER | time:INTEGER | value:TEXT  | server_time:INTEGER |
| ------------ |--------------| ------------|---------------------|
| 2            | 1512293837195| "{\"x\":948,\"y\":7}" | 1512293837199 |
| ...          | ....         |  ...                  | ...           |

where *type* describes the type of the event (e.g. ONKEYDOWN), *time* is the timestamp (ms since Jan 1, 1970, 00:00:00.000 GMT) of the recorded event in the client, *server_time* is the timestamp of the time where the event was received in the server and *value* contains the event data as a json string.<br>
All attributes (columns) are received from the client and no addtitional computation is needed (the server simply stores all received data).

## Website for data acquisition

The website, used for collecting data, (located in **Website**) is written in *HTML* & *JavaScript* and is displayed/executed by opening *Start.htm* in a modern webbrowser (we use *chromium-webbrowser*). The user interface of the website is written in german because the study was conducted with native German speakers.

In order to "generate/collect" data from the user, the websites asks the user to type some text and click on some buttons (including some mouse movements).

**ATTENTION:** You have to **start the server first**, before running/starting the website!

The logic/code for recording the data is in *Website/data-recorder* whereby all important settings are in the file *main.js*.<br>
The url of our server (storing the collected data) can be set in `serverUrl` (the default is `ws://localhost:8080/api`).

The website records and sends all mouse and keyboard events to the server, which stores them in a database (see section *Sever*).<br>
The data itself is send as a json object, which looks like
`{"et": type, "t": time, "v": jsonStringValue}`
where `et` specifies the type of the event, `t` is the timestamp (ms since 01.01.1970 00:00:00) and `v` contains the event data (encoded as json).

Mapping between events and ids:

| type         | id |
| ------------ |----|
| ONKEYDOWN    | 0  |
| ONKEYUP      | 1  |
| ONMOUSEMOVE  | 2  |
| ONMOUSEDOWN  | 3  |
| ONMOUSEUP    | 4  |
| ONMOUSECLICK | 5  |
| ONWHEEL      | 6  |

<br>
How does the data look like for different types of events?

| type         | data |
| ------------ |-----------|
| ONKEYDOWN    | event.key |
| ONKEYUP      | event.key |
| ONMOUSEMOVE  | {"x": event.pageX, "y": event.pageY} |
| ONMOUSEDOWN  |{"x": event.clientX, "y": event.clientY, "b": event.buttons} |
| ONMOUSEUP    | {"x": event.clientX, "y": event.clientY, "b": event.buttons} |
| ONMOUSECLICK | {"x": event.clientX, "y": event.clientY, "b": event.buttons} |
| ONWHEEL      | {"x": event.clientX, "y": event.clientY, "dy": event.deltaY, "dx": event.deltaX, "dz": event.deltaZ, "m": event.deltaMode} |


## Results
See `poster.pdf` and `abstract.pdf`.
