import os
import base64

import face_recognition
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc 
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask, Response, redirect
import cv2
import sqlalchemy as db

from app import app, server

class VideoCamera(object):
    def __init__(self, resize_ratio=0.25, distance_threshold=0.5):
        # Distance threshold set 0.5 after testing (still slightly risky if lookalike person exist)
        self.video = cv2.VideoCapture(0)
        self.process_this_frame = True
        self.resize_ratio = resize_ratio
        self.distance_threshold = distance_threshold
        self.match_found = False
        self.match_found_image = None
        self.match_found_names = []

    def __del__(self):
        self.video.release()

    def __convert_and_resize(self, frame):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_ratio, fy=self.resize_ratio)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        return rgb_small_frame

    def __face_matching(self, rgb_small_frame):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_deteced_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings=known_face_encodings,
                                                     face_encoding_to_check=face_encoding,
                                                     tolerance=self.distance_threshold)
            name = "Unknown"
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name is not "Unknown":
                    self.match_found=True
                    self.match_found_name = name
            face_deteced_names.append(name)
        return face_locations, face_deteced_names

    def __draw_bounding_boxes(self, frame, face_locations, face_names):
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= int(1 // self.resize_ratio)
            right *= int(1 // self.resize_ratio)
            bottom *= int(1 // self.resize_ratio)
            left *= int(1 // self.resize_ratio)
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return frame

    def process_frame(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        # Preprocessing (reduce size for faster processing and changing from BGR to RGB channels)
        rgb_small_frame = self.__convert_and_resize(frame)
        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Generate face matches
            face_locations, face_names = self.__face_matching(rgb_small_frame)
            # Display the results
            frame = self.__draw_bounding_boxes(frame, face_locations, face_names)
        ret, jpeg = cv2.imencode('.jpg', frame)
        # Constantly alternate frames to improve
        self.process_this_frame = not self.process_this_frame
        # If match found, store match
        if self.match_found and self.match_found_image is None:
            self.match_found_image = jpeg.tobytes()
        return jpeg.tobytes()

    def restart(self):
        self.video = cv2.VideoCapture(0)
        self.match_found = False
        self.match_found_image = None
        self.match_found_name = None

def gen(camera):
    while True:
        # if camera.match_found:
        #     camera.__del__()
        frame = camera.process_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

camera = VideoCamera()

def load_encodings(data_folder):
    face_encoding_list = []
    face_names_list = []
    for image_file in os.listdir(data_folder):
        face_names_list.append(image_file.split(".")[0])
        face_image = face_recognition.load_image_file(os.path.join(data_folder, image_file))
        face_encoding_list.append(face_recognition.face_encodings(face_image)[0])
    return face_names_list, face_encoding_list

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("data/Obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("data/Biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

jm_image = face_recognition.load_image_file("data/Jian Ming.jpeg")
jm_face_encoding = face_recognition.face_encodings(jm_image)[0]

de_bin_image = face_recognition.load_image_file("data/De Bin.jpeg")
de_bin_face_encoding = face_recognition.face_encodings(de_bin_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    jm_face_encoding,
    de_bin_face_encoding
]

known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Jian Ming",
    "De Bin"
]

initial_layout = html.Div(id="initial-layout", children=[
    html.H3(id="intro-header", children="Welcome to the face recognition demo!"),
    html.Br(),
    html.P(id="intro-statement", children="The purpose of this demo is to illustrate how facial recognition works as well as its weaknesses."),
    html.P(id="intro-begin-statement", children="To begin the demo, please click on the button below."),
    html.Br(),
    html.Div(id="intro-button-container", className="container", children=
        dbc.Button(id="intro-button", color="primary", className="mr-1", children="Begin Demo", n_clicks=0)
    )
])
video_stream_layout = html.Div(id="video-stream-layout", children=[
    html.Img(id="vid-stream", src="/")
])
detected_layout = html.Div(id="detected-layout", children=[
    html.H2(id="detected-header"),
    html.Div(className="container", children=[ 
        html.Img(id="detected-face"),
        html.Img(id="database-face"),
    ]),
    html.Br(),
    html.Div(id="detected-button-container", className="container", children=
        dbc.Button(id="return-start", color="primary", className="mr-1", children="Return to start", n_clicks=0)
    )
])
layout = html.Div([
    dcc.Interval(id="checker", interval=1000, n_intervals=0, disabled=True),
    dbc.Card(
        [
            dbc.CardHeader("Face Recognition Demo", className="card-title"),
            dbc.CardBody(
                [
                    html.Div(id="card-content", className="container", children=[
                        initial_layout,
                        video_stream_layout,
                        detected_layout
                    ])
                ]
            ),
        ]
    )
])


@server.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.callback(
    [Output(component_id='initial-layout', component_property='style'),
     Output(component_id='video-stream-layout', component_property='style'),
     Output(component_id='detected-layout', component_property='style'),
     Output(component_id='checker', component_property='disabled'),
     Output(component_id='vid-stream', component_property='src'),
     Output(component_id='detected-header', component_property='children'),
     Output(component_id='detected-face', component_property='src'),
     Output(component_id='database-face', component_property='src')],
    [Input(component_id='intro-button', component_property='n_clicks'),
     Input(component_id='checker', component_property='n_intervals'),
     Input(component_id='return-start', component_property='n_clicks')],
    [State(component_id='checker', component_property='disabled')]
)
def layout_update(intro_click, interval, return_click, checker_state):
    if intro_click is None and interval is None and return_click is None:
        raise PreventUpdate
    ctx = dash.callback_context
    trigger_component = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_component == "intro-button":
        return ({'display': 'none'}, {'display': 'block'}, {'display': 'none'}, False, "/video_feed", "", "", "")
    elif trigger_component == "checker":
        if not checker_state:
            if camera.match_found:
                detected_header = "User detected: {}".format(camera.match_found_name)
                detected_image = "data:image/jpeg;base64,{}".format(base64.b64encode(camera.match_found_image).decode())
                database_image = "data:image/jpeg;base64,{}".format(base64.b64encode(camera.match_found_image).decode())
                return ({'display': 'none'}, {'display': 'none'}, {'display': 'block'}, True, "/", detected_header, detected_image, database_image)
            else:
                return ({'display': 'none'}, {'display': 'block'}, {'display': 'none'}, False, "/video_feed", "", "", "")
        else:
            raise PreventUpdate
    elif trigger_component == "return-start":
        camera.restart()
        return ({'display': 'block'}, {'display': 'none'}, {'display': 'none'}, True, "/video_feed", "", "", "")
    else:
        raise PreventUpdate