# app.py
from flask import Flask, render_template
import json

app = Flask(__name__)

# Load configuration from config.json
def load_config():
    with open('config.json') as config_file:
        config = json.load(config_file)
    return config

config = load_config()
print("config: ", config)

@app.route('/')
def index():
    # Inject the API host into the HTML template
    api_host = config.get('API_HOST', 'http://localhost:8080')
    return render_template('index.html', api_host=api_host)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
