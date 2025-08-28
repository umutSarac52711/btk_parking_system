# database.py
import psycopg2
import psycopg2.extras

# --- DATABASE CONFIGURATION ---
DB_NAME = "parking_db"
DB_USER = "postgres"
DB_PASS = "123"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    return conn

# Function to create the tables
def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Execute the SQL to create your schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id SERIAL PRIMARY KEY,
            plate_number TEXT UNIQUE NOT NULL,
            owner_name TEXT,
            phone_number TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_log (
            id SERIAL PRIMARY KEY,
            vehicle_id INTEGER REFERENCES vehicles(id),
            entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            exit_time TIMESTAMPTZ,
            is_parked BOOLEAN DEFAULT TRUE,
            total_fee NUMERIC
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("Tables created successfully.")

# Add a function to get parked cars (example)
def get_currently_parked_vehicles():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
        SELECT v.plate_number, p.entry_time 
        FROM parking_log p
        JOIN vehicles v ON p.vehicle_id = v.id
        WHERE p.is_parked = TRUE
        ORDER BY p.entry_time DESC;
    """)
    vehicles = cursor.fetchall()
    cursor.close()
    conn.close()
    return vehicles

# ... You will add more functions here, like 'check_in_vehicle(plate_number)' ...