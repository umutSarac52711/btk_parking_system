# app.py
from flask import Flask
from flask_cors import CORS
from .routes import api  # Import our Blueprint from routes.py
from . import database # Import the database module

app = Flask(__name__)
CORS(app)

# Tell Flask to use the routes defined in our 'api' Blueprint
app.register_blueprint(api)

# This block runs only when you execute 'python app.py' directly
if __name__ == '__main__':
    # First, ensure the database and tables exist
    database.create_tables()
    # Then, run the app
    app.run(debug=True, port=5000)