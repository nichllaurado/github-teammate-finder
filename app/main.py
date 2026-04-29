import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS 
from dotenv import load_dotenv


load_dotenv();

app = Flask(__name__)
CORS(app)

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
