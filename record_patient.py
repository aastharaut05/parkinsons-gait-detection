import serial
import time
import os
from datetime import datetime  # <-- Added to generate unique timestamps

# --- CONFIGURATION ---
SERIAL_PORT = 'COM8'  # Change this if your ESP32 is on a different port
BAUD_RATE = 115200
RECORDING_TIME = 30   # How many seconds to record the walking data
FOLDER_NAME = 'Patient_Records' 
# ---------------------

# 1. Create the folder automatically if it does not exist yet
if not os.path.exists(FOLDER_NAME):

    os.makedirs(FOLDER_NAME)
    print(f"📁 Created new directory: {FOLDER_NAME}")

# 2. Generate a unique filename using the current date and time
# It will look like: patient_20260309_153045.csv
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f'patient_{timestamp}.csv'

# 3. Combine the folder name and file name to make the full path
file_path = os.path.join(FOLDER_NAME, OUTPUT_FILE)

try:
    # Connect to the ESP32
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print(f"✅ Successfully connected to {SERIAL_PORT}")
    print(f"📄 Creating new file: {OUTPUT_FILE}")
    print("🚶‍♂️ Start walking! Recording data...")
    
    # Open the CSV file INSIDE the new folder and write the header
    with open(file_path, 'w') as f:
        f.write("Timestamp,Ax,Ay,Az,Gx,Gy,Gz\n")
        
        start_time = time.time()
        
        # Record for the exact number of seconds specified
        while (time.time() - start_time) < RECORDING_TIME:
            if ser.in_waiting > 0:
                # Read the data, decode it, and clean up any whitespace
                line = ser.readline().decode('utf-8').strip()
                
                # Ensure it's a valid data line containing commas
                if ',' in line:
                    f.write(line + '\n')
                    print(f"Recording: {line}")
                    
    ser.close()
    print("\n✅ Data collection complete!")
    print(f"📁 File saved automatically as: {file_path}")

except serial.SerialException:
    print(f"❌ ERROR: Could not connect to {SERIAL_PORT}.")
    print("Make sure your ESP32 is plugged in and the Arduino Serial Monitor is CLOSED.") 
