import os
import time
import base64
import logging

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

class VideoCamera(object):
    def __init__(self, resize_ratio=1, distance_threshold=0.5):
        # Distance threshold set 0.5 after testing (still slightly risky if lookalike person exist)
        self.video = cv2.VideoCapture(99)
        self.eye_model = cv2.CascadeClassifier('models/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
        self.process_this_frame = True
        self.resize_ratio = resize_ratio
        self.distance_threshold = distance_threshold
        self.match_found = False
        self.match_found_image = None
        self.match_found_names = []
        # For eye anti-spoofing method
        self.current_match_name = None
        self.blink_counter = 0
        self.blink_target = 0
        self.blink_timer = 0
        self.blink_timer_start = 0
        self.eyes_close = True

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
        face_detected_distances = []
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
            face_deteced_names.append(name)
            face_detected_distances.append(face_distances[best_match_index])
        return face_locations, face_deteced_names, face_detected_distances
    
    def __eye_matching(self, rgb_small_frame, face_locations):
        # Convert to grayscale
        gray_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_RGB2GRAY)
        single_face_location = face_locations[0]
        face_image = gray_frame[single_face_location[0]:single_face_location[2], single_face_location[3]: single_face_location[1]]
        eyes = self.eye_model.detectMultiScale(face_image, 1.1, 3)
        return eyes

    def __draw_bounding_boxes(self, frame, face_locations, face_names, face_distances):
        # Display the results
        for (top, right, bottom, left), name, distance in zip(face_locations, face_names, face_distances):
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
            cv2.putText(frame, name + " " + str(round(distance, 2)), (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)
        return frame
    
    def __draw_eye_keypoints(self, frame, eyes, face_locations):
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        single_face_location = face_locations[0]
        # Centers must be (x, y) coordinates. note that eyes location are in (x, y, w, h) format instead
        eye_1_center_x = eye_1[0] + (eye_1[2] / 2) 
        eye_1_center_y = eye_1[1] + (eye_1[3] / 2)
        eye_2_center_x = eye_2[0] + (eye_2[2] / 2)
        eye_2_center_y = eye_2[1] + (eye_2[3] / 2)
        # Must remember to rescale back to full small image frame
        eye_1_center = [int(eye_1_center_x + single_face_location[3]), int(eye_1_center_y + single_face_location[0])]
        eye_2_center = [int(eye_2_center_x + single_face_location[3]), int(eye_2_center_y + single_face_location[0])]
        # Scale back to original image size
        eye_1_center[0] = int(eye_1_center[0] // self.resize_ratio)
        eye_2_center[0] = int(eye_2_center[0] // self.resize_ratio)
        eye_1_center[1] = int(eye_1_center[1] // self.resize_ratio)
        eye_2_center[1] = int(eye_2_center[1] // self.resize_ratio)
        # Draw eyes
        cv2.circle(frame, tuple(eye_1_center), 5, (0, 255, 0), 2)
        cv2.circle(frame, tuple(eye_2_center), 5, (0, 255, 0), 2)
        return frame

    def __draw_blink_counter(self, frame):
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Face detected! Please blink {} times".format(self.blink_target), (10,40), font, 0.8, (0,255,0), 2)
        cv2.putText(frame, "Blinked: {}".format(self.blink_counter), (10,80), font, 0.8, (0,255,0), 2)
        return frame

    def process_frame(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        # Preprocessing (reduce size for faster processing and changing from BGR to RGB channels)
        rgb_small_frame = self.__convert_and_resize(frame)
        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Generate face matches
            face_locations, face_names, face_distances = self.__face_matching(rgb_small_frame)
            if len(face_names) == 1:
                # Draw number of times to blink text. If matched do not need to draw
                if self.blink_counter != self.blink_target:
                    frame = self.__draw_blink_counter(frame)
                # Determine if match found after blink test (this is placed above to allow the frame to write the last blinked image)
                # If not, the last actual blinked image shown will be the second last blinked image
                if self.blink_counter == self.blink_target:
                    self.match_found = True
                    self.match_found_name = self.current_match_name
                # For first match
                if self.current_match_name is None:
                    self.current_match_name = face_names[0]
                    self.blink_counter = 0
                    self.blink_target = np.random.randint(1, 5)
                    self.blink_timer = np.random.randint(5, 10)
                    self.blink_timer_start = time.time()
                    self.eyes_close = False
                # Check if eyes match
                eyes = self.__eye_matching(rgb_small_frame, face_locations)
                # Check if it is the same user as before
                if self.current_match_name == face_names[0]:
                    # If user eyes is opened
                    if(len(eyes)>=2):
                        frame = self.__draw_eye_keypoints(frame, eyes, face_locations)
                        # Prevent duplicate scenarios
                        if self.eyes_close is True:
                            self.blink_counter += 1
                        self.eyes_close = False
                    # If user eyes is closed
                    else:
                        self.eyes_close = True
                # Different user found
                else:
                    self.current_match_name = None
                    self.blink_counter = 0
                    self.blink_target = np.random.randint(1, 5) 
                    self.blink_timer = np.random.randint(5, 10)
                    self.blink_timer_start = 0
                    self.eyes_close = False
            # More than 1 face detected
            else:
                self.current_match_name = None
                self.blink_counter = 0
                self.blink_target = np.random.randint(1, 5)
                self.blink_timer = np.random.randint(5, 10)
                self.eyes_close = False
            # Display the results
            frame = self.__draw_bounding_boxes(frame, face_locations, face_names, face_distances)
        ret, jpeg = cv2.imencode('.jpg', frame)
        # Constantly alternate frames to improve
        self.process_this_frame = not self.process_this_frame
        # If match found, store match
        if self.match_found and self.match_found_image is None:
            self.match_found_image = jpeg.tobytes()
        return jpeg.tobytes()

    def start_video(self):
        self.video = cv2.VideoCapture(0)
        server.logger.info("Video Started")
    
    def stop_video(self):
        self.video = cv2.VideoCapture(99)
        server.logger.info("Video Stopped")
    
    def reset_status(self):
        self.match_found = False
        self.match_found_image = None
        self.match_found_name = None
        # Reset for demo purposes
        self.blink_counter = 0
        self.blink_target = np.random.randint(1, 5)
        self.blink_timer = np.random.randint(5, 10)
        self.blink_timer_start = 0
        self.eyes_close = False
        self.current_match_name = None

def gen(camera):
    while True:
        # if camera.match_found:
        #     camera.__del__()
        frame = camera.process_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


### Global Variables ###
DATA_FOLDER = "data"
DB_NAME = 'sqlite:///demo.db'
TABLE_NAME = 'users'
INTERVAL_STARTUP = 5
global camera
### Define app server ###
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.DARKLY])
### Set logging level ###
server.logger.setLevel(logging.INFO)
### Create DB if needed ###
server.logger.info("Connecting to database")
engine = db.create_engine(DB_NAME)
connection = engine.connect()
metadata = db.MetaData()
server.logger.info("Creating users table if needed")
face_table = db.Table(
   TABLE_NAME, metadata, 
   db.Column('id', db.Integer, primary_key = True), 
   db.Column('name', db.String), 
   db.Column('image_filepath', db.String),
   db.Column('face_encodings', db.String)
)
metadata.create_all(engine)
### Extract filenames for all users to determine if there is a need to update db ###
server.logger.info("Updating database based on images present in {} folder specified".format(DATA_FOLDER))
users_table = db.Table(TABLE_NAME, metadata, autoload=True, autoload_with=engine)
image_filepath_query = db.select([users_table.c.image_filepath])
image_filepath_result = connection.execute(image_filepath_query).fetchall()
### Check if file exists, if not update db ###
for filename in os.listdir(DATA_FOLDER):
    image_filepath = os.path.join(DATA_FOLDER, filename)
    if image_filepath not in image_filepath_result:
        face_image = face_recognition.load_image_file(image_filepath)
        face_encoding = face_recognition.face_encodings(face_image)[0]
        face_encoding_string = list(map(lambda x : str(x), face_encoding))
        insert_query = users_table.insert().values(
            name=filename.split('.')[0],
            image_filepath=image_filepath,
            face_encodings=",".join(face_encoding_string)
        )
        connection.execute(insert_query)
### Query to load data for app ###
server.logger.info("Extracting all users information from database for application")
users_query = db.select([users_table])
users_result = connection.execute(users_query).fetchall()
known_face_encodings = [np.array(list(map(lambda y: float(y), x[-1].split(',')))) for x in users_result]
known_face_names = [x[1] for x in users_result]
known_image_filepaths = [x[2] for x in users_result]
### Define camera ###
camera = VideoCamera()
server.logger.info("Application Ready!")
### Define layouts ###
initial_layout = html.Div(id="initial-layout", children=[
    html.H3(id="intro-header", children="Welcome to the facial recognition demo with anti-spoof!"),
    html.Br(),
    html.P(id="intro-statement", children="The purpose of this demo is to illustrate a method to counter spoofing attempts to the system through the use of blink detection counter."),
    html.P(id="intro-begin-statement", children="To begin the demo, please click on the button below."),
    html.Br(),
    html.Br(),
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
        dbc.Row([
            html.Div(id="detected-face-container", children=[
                html.Img(id="detected-face"),
                html.H5(id="detected-face-text", children="Detected Face")
            ]),
            html.Div(id="database-face-container", children=[
                html.Img(id="database-face"),
                html.H5(id="database-face-text", children="Database Face")
            ])
        ])
    ]),
    html.Br(),
    html.Br(),
    html.Div(id="detected-button-container", className="container", children=
        dbc.Button(id="return-start", color="primary", className="mr-1", children="Return to start", n_clicks=0)
    )
])
### Specify initial layout ###
app.layout = html.Div(id="main-body", children=[
    dbc.NavbarSimple(
        children=[
            dbc.DropdownMenu(
                children=[
                    dbc.NavItem(dbc.NavLink("Basic", id="basic-link", href="http://localhost:8050", n_clicks=0)),
                    dbc.NavItem(dbc.NavLink("Anti Spoofing", id="anti-spoofing-link", href="/", n_clicks=0))
                ],
                nav=True,
                in_navbar=True,
                label="Select demo",
                id="dropdown-menu"
            )
        ],
        brand="Group 5 Demo",
        color="dark",
        dark=True,
    ),
    html.Br(),
    html.Br(),
    dcc.Interval(id="checker", interval=1000, n_intervals=0, disabled=True),
    html.Div(id='page-content', className='container d-flex align-items-center justify-content-center', children=[
        dbc.Card(
            [
                dbc.CardHeader("Face Recognition With Anti-Spoofing Demo", className="card-title"),
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
])

### Callbacks and routes ###
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
        camera.reset_status()
        camera.start_video()
        return ({'display': 'none'}, {'display': 'block'}, {'display': 'none'}, False, "/video_feed", "", "", "")
    elif trigger_component == "checker":
        if not checker_state:
            if camera.match_found:
                detected_header = "User detected: {}".format(camera.match_found_name)
                detected_image = "data:image/jpeg;base64,{}".format(base64.b64encode(camera.match_found_image).decode())
                database_image_index = known_face_names.index(camera.match_found_name)
                database_image = "data:image/jpeg;base64,{}".format(base64.b64encode(open(known_image_filepaths[database_image_index], 'rb').read()).decode())
                camera.reset_status()
                return ({'display': 'none'}, {'display': 'none'}, {'display': 'block'}, True, "/", detected_header, detected_image, database_image)
            else:
                return ({'display': 'none'}, {'display': 'block'}, {'display': 'none'}, False, "/video_feed", "", "", "")
        else:
            raise PreventUpdate
    elif trigger_component == "return-start":
        return ({'display': 'block'}, {'display': 'none'}, {'display': 'none'}, True, "/", "", "", "")
    else:
        raise PreventUpdate


@app.callback(
    Output(component_id='basic-link', component_property='href'),
    [Input(component_id='basic-link', component_property='n_clicks')],
    [State(component_id='basic-link', component_property='href')]
)
def destroy_camera(n_clicks, href):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    camera.stop_video()
    return href

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8040, debug=True, dev_tools_ui=False, dev_tools_props_check=False)