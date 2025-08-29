# backend/database.py
import psycopg2
import psycopg2.extras
from datetime import datetime

# --- DATABASE CONFIGURATION ---
DB_NAME = "parking_db"
DB_USER = "postgres"
DB_PASS = "123" # Make sure this is correct
DB_HOST = "localhost"
DB_PORT = "5432"

# A fixed rate for calculating fees, e.g., $3 per hour
PARKING_RATE_PER_HOUR = 3.0

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id SERIAL PRIMARY KEY,
            plate_number TEXT UNIQUE NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_log (
            id SERIAL PRIMARY KEY,
            vehicle_id INTEGER NOT NULL REFERENCES vehicles(id),
            entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            exit_time TIMESTAMPTZ,
            is_parked BOOLEAN DEFAULT TRUE,
            total_fee NUMERIC(10, 2)
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("Tables verified/created successfully.")

def check_in_vehicle(plate_number):
    """Handles the entire check-in logic."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Step 1: Find vehicle by plate number. If it doesn't exist, create it.
        cursor.execute("SELECT id FROM vehicles WHERE plate_number = %s;", (plate_number,))
        vehicle = cursor.fetchone()
        
        if vehicle:
            vehicle_id = vehicle['id']
        else:
            # RETURNING id is an efficient way to get the new ID after an insert
            cursor.execute("INSERT INTO vehicles (plate_number) VALUES (%s) RETURNING id;", (plate_number,))
            vehicle_id = cursor.fetchone()['id']
        

        # Step 2: Verification - If there is a parking log entry that has the same exact license plate and does not have an exit date, that means the car is still parked. 
        # This is an invalid or a repeated entry
        cursor.execute("""
            SELECT id FROM parking_log
            WHERE vehicle_id = %s AND is_parked = TRUE;
        """, (vehicle_id,))
        active_log = cursor.fetchone()

        if active_log:
            ## If the difference in entry time is minimal (less than 40 seconds), it is a re-entry. Safe to ignore
            if (datetime.now() - active_log['entry_time']).total_seconds() < 40:
                print(f"Vehicle {plate_number} is already parked.")
                return {"error": "Vehicle is already parked."}
            else:
                print(f"Vehicle {plate_number} attempted to re-check-in without checking out.")
                return {"error": "Vehicle is already parked. Please check out before checking in again."}
            
        # Step 3: Fetch the latest parking log. If it was made within the last 6 seconds, it may be a bugged rapid entry.
        # Cooldown of 6 seconds is arbitrary for now.
        cursor.execute("""
            SELECT id FROM parking_log
            WHERE is_parked = TRUE
            ORDER BY entry_time DESC LIMIT 1;
        """)
        latest_log = cursor.fetchone()

        if latest_log:
            if (datetime.now() - latest_log['entry_time']).total_seconds() < 6:
                print(f"Vehicle {plate_number} is entering too quickly.")
                return {"error": "Vehicle is entering too quickly."}

        # Finally: Create a new parking log entry
        cursor.execute(
            "INSERT INTO parking_log (vehicle_id) VALUES (%s) RETURNING id, entry_time;",
            (vehicle_id,)
        )
        new_log = cursor.fetchone()
        conn.commit()
        return {"message": "Check-in successful", "log_id": new_log['id'], "entry_time": new_log['entry_time']}
    except Exception as e:
        conn.rollback()
        print(f"Database error during check-in: {e}, {e.__class__}")
        return None
    finally:
        cursor.close()
        conn.close()

def check_out_vehicle(plate_number):
    """Handles the check-out logic and fee calculation."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Find the active parking log entry for this plate
        cursor.execute("""
            SELECT p.id, p.entry_time FROM parking_log p
            JOIN vehicles v ON p.vehicle_id = v.id
            WHERE v.plate_number = %s AND p.is_parked = TRUE
            ORDER BY p.entry_time DESC LIMIT 1;
        """, (plate_number,))
        log_entry = cursor.fetchone()

        if not log_entry:
            return {"error": "No active parking session found for this vehicle."}

        # Calculate duration and fee
        entry_time = log_entry['entry_time']
        exit_time = datetime.now(entry_time.tzinfo) # Use the same timezone
        duration_seconds = (exit_time - entry_time).total_seconds()
        duration_hours = duration_seconds / 3600
        fee = round(duration_hours * PARKING_RATE_PER_HOUR, 2)

        # Update the log entry
        cursor.execute("""
            UPDATE parking_log 
            SET exit_time = %s, is_parked = FALSE, total_fee = %s
            WHERE id = %s
            RETURNING *;
        """, (exit_time, fee, log_entry['id']))
        updated_log = cursor.fetchone()
        
        conn.commit()
        return {"message": "Check-out successful", "details": updated_log}
    except Exception as e:
        conn.rollback()
        print(f"Database error during check-out: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_currently_parked_vehicles():
    """Retrieves a list of all vehicles currently marked as parked."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
        SELECT v.plate_number, p.entry_time 
        FROM parking_log p
        JOIN vehicles v ON p.vehicle_id = v.id
        WHERE p.is_parked = TRUE
        ORDER BY p.entry_time ASC;
    """)
    vehicles = cursor.fetchall()
    cursor.close()
    conn.close()
    return vehicles

def get_parking_history():
    """Retrieves all parking log entries."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute("""
        SELECT v.plate_number, p.entry_time, p.exit_time, p.total_fee
        FROM parking_log p
        JOIN vehicles v ON p.vehicle_id = v.id
        ORDER BY p.entry_time DESC;
    """)
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history