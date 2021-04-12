import os
import base64

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc 
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import cv2
import sqlalchemy as db

from app import app, server

DATA_FOLER = "data"

layout = html.Div(id="database-insert-layout-container", children=[
    html.H1("Add new face to database"),
    html.Br(),
    dcc.Upload(
        id='upload-face-image',
        children=html.Div(id='upload-face-image-container', children=[
            html.A('Drag and Drop or Click Here to Upload File ')
        ]),
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload', children=[
        dbc.Alert(id="success-upload-alert", children="Successfully update database!", color="surccess", is_open=False, fade=True, dismissable=True),
        dbc.Alert(id="fail-upload-alert", children="Failed to update database", color="danger", is_open=False, fade=True, dismissable=True),
    ]),
])


@app.callback(
    [Output('success-upload-alert', 'is_open'),
     Output('fail-upload-alert', 'is_open')],
    [Input('upload-face-image', 'contents')],
    [State('upload-face-image', 'filename'),
     State('upload-face-image', 'last_modified')])
def update_output(contents, names, dates):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        print(os.getcwd())
        return (True, True)
    else:
        raise PreventUpdate