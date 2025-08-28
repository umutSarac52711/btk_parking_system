from flask import Flask
from flask_cors import CORS
from flasgger import Swagger # <-- IMPORT THIS
from .routes import api
from . import database

app = Flask(__name__)
CORS(app)
swagger = Swagger(app) # <-- INITIALIZE IT

app.register_blueprint(api)

if __name__ == '__main__':
    database.create_tables()
    app.run(debug=True, port=5000)