"""
Main module of flask API.
"""
# Third party modules
import os
import glob
import io
import base64
from typing import Any
from functools import wraps
import pandas as pd
from flask import (
    Flask, request,
    json, make_response, Response
)
from flask_cors import CORS, cross_origin
from generator import conversation, initialize_llmchain_after_refresh, initialize_database
from asgiref.wsgi import WsgiToAsgi
# Module
app = Flask(__name__)
cors = CORS(app)
asgi_app = WsgiToAsgi(app)
api_cors = {
  "origins": ["*"],
  "methods": ["OPTIONS", "GET", "POST"],
  "allow_headers": ["Content-Type"]
}
#app.config['PROPAGATE_EXCEPTIONS'] = True
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok = True)

def validate_request(request_api: Any)-> bool:
    """
    method will take a json request and perform all validations if the any error 
    found then return error response with status code if data is correct then 
    return data in a list.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    bool
        return True or False.

    """

    if "data" in request_api.files:
        return True
    if "data" not in request_api.files:
        return False
def validate_text_request(request_api: Any)-> bool:
    """
    method will take a json request and perform all validations if the any error 
    found then return error response with status code if data is correct then 
    return data in a list.

    Parameters
    ----------
    request_api: Request
        contain the request data in file format.

    Return
    ------
    bool
        return True or False.

    """
    data = request_api.get_json()
    if "data" in data:
        if data["data"] == '':
            return False
        return True
    if "data" not in data:
        return False
def get_textdata(data: json)-> str:
    """
    method will take a json data and return text_data.

    Parameters:
    ----------
    data: json
        json data send in request.

    Return:
    ------
    text_data: str
        return text_data as string.

    """
    text_data = data["data"]
    return text_data

def get_data(request_api: Any)-> str:
    """
    method will take request and get data from request then return thhe data.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    image_file: str
        return the data file as string.

    """
    data = request_api.files["data"]
    return data
def make_bad_params_value_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'data key error',
        'category' : 'Bad Params',}),
        400)
    return result
def make_file_save_error_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'File not save sucesfully',
        'category' : 'Bad Params error',}),
        400)
    return result

def make_file_save_error_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'File not save sucesfully',
        'category' : 'Bad Params error',}),
        400)
    return result
def make_invalid_extension_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'Invalid Extension',
        'category' : 'Params error',}),
        400)
    return result
def validate_extension(request_api: Any)-> bool:
    """
    method will take image and check its extension is .mp4, .avi, .FLV.

    Parameters
    ----------
    video: Any
        api request send by user.

    Return
    ------
    bool
        return the true or false video is has valid extension or not.

    """
    data_f = request_api.files["data"]
    data_list = data_f.filename.split(".")
    data_extension = data_list[len(data_list)-1]
    data_extensions = ['pdf']
    if data_extension in data_extensions:
        return True
    return False

def save_file(request_api: Any)-> str:
    """
    method will take request and save file from request in specified folder.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.
    Return:
    ------
    save_file_path: str
        file path save on our local sever.
    """
    data_f = request_api.files["data"]
    save_file_path = f"{upload_folder}/{data_f.filename}"
    data_f.save(save_file_path)
    return True
@app.route('/upload_file', methods = ['POST'])
@cross_origin(**api_cors)
def upload_file():
    """
    method will take the file as input and save file on local server.

    Parameters:
    ----------
    None

    Return:
    ------
    str
        return the reponse.

    """
    try:
        if validate_request(request):
            if validate_extension(request):
                if save_file(request):
                    response = initialize_database(upload_folder, chunk_size = 1000, chunk_overlap = 100)
                    output = {
                        'collections_name': response,
                        'message' : "embedding created sucesfully"
                        }
                    # output_dict["output"] = output
                    return Response(
                        json.dumps(output),
                        mimetype = 'application/json'
                        )
                return make_file_save_error_response()
            return make_invalid_extension_response()
        make_bad_params_value_response()
    except Exception as exception:
        result = make_response(json.dumps(
                    {'message'  : str(exception),
                    'category' : 'Internal server error',}),
                    500)
        return result
    
@app.route('/query', methods = ['POST'])
@cross_origin(**api_cors)
def generate_response():
    """
    method will take the text prompt as input and return the generated reponse.

    Parameters:
    ----------
    None

    Return:
    ------
    str
        return the reponse.

    """
    try:
        if validate_text_request(request):
            query = request.get_json()
            text_data = get_textdata(query)
            output, chat_history = conversation(text_data)
            output = { 
                "answer" : output,
                "chat_history" : chat_history
            }
            return Response(
                json.dumps(output),
                mimetype = 'application/json'
                )
        return make_bad_params_value_response()
    except Exception as exception:
        result = make_response(json.dumps(
                    {'message'  : str(exception),
                    'category' : 'Internal server error',}),
                    500)
        return result
@app.route('/refresh', methods = ['POST'])
@cross_origin(**api_cors)
def refresh_history():
    """
    method will take the trigger text to refresh the llm chain memory..

    Parameters:
    ----------
    None

    Return:
    ------
    str
        return the success response.

    """
    try:
        if validate_text_request(request):
            query = request.get_json()
            text_data = get_textdata(query)
            output = initialize_llmchain_after_refresh()
            output = { 
                "Mesage" : output
            }
            return Response(
                json.dumps(output),
                mimetype = 'application/json'
                )
        return make_bad_params_value_response()
    except Exception as exception:
        result = make_response(json.dumps(
                    {'message'  : str(exception),
                    'category' : 'Internal server error',}),
                    500)
        return result
if __name__=='__main__':
    app.run(debug = True, host = "0.0.0.0")