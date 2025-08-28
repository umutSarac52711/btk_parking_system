# routes.py
from flask import Blueprint, jsonify
from . import database  # Import from the database.py file in the same folder

# A Blueprint is Flask's way of organizing routes.
api = Blueprint('api', __name__)

@api.route('/api/parked_cars', methods=['GET'])
def parked_cars():
    cars = database.get_currently_parked_vehicles()
    return jsonify(cars)

# ... You will add more routes here, like '/api/checkin' ...