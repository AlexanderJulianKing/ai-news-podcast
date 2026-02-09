#!/bin/bash

source "$(dirname "$0")/venv/bin/activate"

while true; do
    # Get the current hour and minute
    current_time=$(date +'%H:%M')

    # Wait until it's between 4:00 and 5:00 AM, but only run once per day
    current_hour=$(date +'%H')
    today=$(date +'%Y_%m_%d')
    echo "Current time: $current_time. Waiting for 4:00 AM..."
    if [[ "$current_hour" != "04" ]] || [[ "$today" == "$last_ran" ]]; then
        sleep 60  # Check every minute
        continue
    fi
    last_ran="$today"

    echo "Hello from the Bash script!"
    echo 'Running program!'
    current_date=$(date +'%Y_%m_%d')
    echo $current_date
    python3 main.py
    folder="output_audio"
    other_folder="output_scripts"
    segment_folder="segment_audio"

    mp3_file="${folder}/${current_date}.mp3"
    if [ ! -f "$mp3_file" ]; then
        echo "lol it did not work. deleting scripts and segments"
        find "$other_folder" -name "*${current_date}*.txt" ! -name "*overview.txt" -exec rm {} \;
        find "$segment_folder" -name "*${current_date}*.mp3" -exec rm {} \;
        python3 main.py
        if [ ! -f "$mp3_file" ]; then
            echo "lol it did not work. deleting scripts and segments AGAIN"
            find "$other_folder" -name "*${current_date}*.txt" ! -name "*overview.txt" -exec rm {} \;
            find "$segment_folder" -name "*${current_date}*.mp3" -exec rm {} \;
            python3 main.py
        else
            echo all good
        fi
    else
        echo all good
    fi

    # Get today's date and format it as YYYY_MM_DD
    date=$(date +'%Y_%m_%d')

    # Create the filename using the formatted date and folder path
    filename="output_audio/${date}.mp3"
    echo $filename
    # Upload the file using curl
    #curl -X PUT -T $filename -u Alexander.julian.king@gmail.com:hubviz-pipqe2-gaxvEs "https://webdav.blubrry.com/media/1474721/"
    python3 moviemaker.py
    echo made_movie
    #echo uploading_podcast
    #python3 blubrry_api.py
    echo uploading_movie
    python3 uploader2.py --file="output_video.mp4" --title="Summer vacation in California" --description="Had fun surfing in Santa Cruz" --keywords="surfing,Santa Cruz" --category="25" --privacyStatus="public"

    # python3 blubrry_api.py

    #source_directory="/Users/alexanderking/Desktop/newscaster3.5"
    #destination_directory="/Users/alexanderking/Library/Mobile Documents/com~apple~CloudDocs/messenger_folder"
    #cp -R "$source_directory"/* "$destination_directory"
    echo 'Done for the day!'

    # Calculate the time until 4 AM the next day
    next_run=$(date -d 'tomorrow 4:00' +%s)
    current_time=$(date +%s)
    sleep_time=$((next_run - current_time))
    
    echo "Sleeping for $sleep_time seconds until 4:00 AM tomorrow..."
    sleep $sleep_time
done
