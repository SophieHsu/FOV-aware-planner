#!/bin/bash

# Define the command to start your program
# program_command="python overcooked_ai_py/steak_api_test.py -l steak_mid_2 -v 0"
# program_command="python overcooked_ai_py/steak_api_test.py -l steak_mid_2 -v 1 -k 1"
# program_command="python overcooked_ai_py/steak_api_test.py -l steak_mid_2 -v 1"

# program_command="python overcooked_ai_py/steak_api_test.py -l steak_None_3 -v 0"
# program_command="python overcooked_ai_py/steak_api_test.py -l steak_none_3 -v 1 -k 1"
# program_command="python overcooked_ai_py/steak_api_test.py -l steak_None_3 -v 1"

# program_command="python overcooked_ai_py/steak_api_test.py -l steak_side_2 -v 0"
# program_command="python overcooked_ai_py/steak_api_test.py -l steak_side_2 -v 1"

# Define the number of restart attempts before giving up
max_attempts=5

# Define the delay (in seconds) between restarts
restart_delay=4

# Counter to keep track of restart attempts
attempts=0

# Infinite loop to continuously restart the program
while true; do
    # Run the program in the background
    $program_command &
    
    # Get the process ID (PID) of the program
    program_pid=$!
    
    # Wait for the program to finish
    wait $program_pid

    # Check the exit status of the program
    exit_status=$?
    
    if [ $exit_status -eq 0 ]; then
        echo "Program exited successfully."
        break
    else
        echo "Program exited with an error. Restarting..."
        attempts=$((attempts + 1))
        
        if [ $attempts -ge $max_attempts ]; then
            echo "Max restart attempts reached. Exiting."
            break
        fi
        
        sleep $restart_delay
    fi
done