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
    rm -f output_video.mp4
    python3 moviemaker.py
    if [ -f "output_video.mp4" ]; then
        echo made_movie
        echo uploading_movie
        upload_marker="output_scripts/${date}_UPLOAD_COMPLETE.flag"
        upload_succeeded=false
        if [ -s "$upload_marker" ]; then
            echo "YouTube upload already verified by $upload_marker; skipping duplicate upload"
            upload_succeeded=true
        else
            for upload_attempt in 1 2 3; do
                echo "YouTube upload attempt ${upload_attempt}/3"
                if python3 uploader2.py --file="output_video.mp4" --category="25" --privacyStatus="public"; then
                    if [ -s "$upload_marker" ]; then
                        echo "YouTube upload verified by $upload_marker"
                        upload_succeeded=true
                        break
                    fi
                    echo "Uploader exited successfully but did not write $upload_marker"
                else
                    echo "YouTube upload attempt ${upload_attempt}/3 failed"
                fi

                if [ "$upload_attempt" -lt 3 ]; then
                    upload_backoff=$((upload_attempt * 60))
                    echo "Starting a fresh upload session in ${upload_backoff} seconds..."
                    sleep "$upload_backoff"
                fi
            done
        fi

        if [ "$upload_succeeded" != true ]; then
            failure_time=$(date --iso-8601=seconds)
            failure_message="[$failure_time] YouTube upload failed after 3 fresh sessions for $date"
            echo "$failure_message" | tee -a logs/upload_failures.log
        fi
    else
        echo "moviemaker failed — skipping upload to avoid uploading stale video"
    fi

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
