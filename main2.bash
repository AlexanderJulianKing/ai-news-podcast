#!/bin/bash

source "$(dirname "$0")/venv/bin/activate"

# Two shows: the morning edition every day at 4:00, and the evening edition on
# weekdays at 3:45 pm for the drive home (2026-10-09). newscaster/schedule.py
# decides which is due; a show runs once per day, even if it fails.
run_edition() {
    edition="$1"
    export NEWSCASTER_EDITION="$edition"
    day=$(date +'%Y_%m_%d')
    if [ "$edition" = "evening" ]; then
        key="${day}_pm"
        other="__no_other_edition__"
    else
        key="$day"
        other="*${day}_pm*"      # never touch the evening show's files
    fi
    mkdir -p logs
    echo "$key" > "logs/${key}_STARTED.flag"

    echo "Hello from the Bash script!"
    echo "Running the $edition edition: $key"
    python3 main.py
    folder="output_audio"
    other_folder="output_scripts"
    segment_folder="segment_audio"

    mp3_file="${folder}/${key}.mp3"
    if [ ! -f "$mp3_file" ]; then
        echo "lol it did not work. deleting scripts and segments"
        find "$other_folder" -name "*${key}*.txt" ! -name "$other" ! -name "*overview.txt" -exec rm {} \;
        find "$segment_folder" -name "*${key}*.mp3" ! -name "$other" -exec rm {} \;
        python3 main.py
        if [ ! -f "$mp3_file" ]; then
            echo "lol it did not work. deleting scripts and segments AGAIN"
            find "$other_folder" -name "*${key}*.txt" ! -name "$other" ! -name "*overview.txt" -exec rm {} \;
            find "$segment_folder" -name "*${key}*.mp3" ! -name "$other" -exec rm {} \;
            python3 main.py
        else
            echo all good
        fi
    else
        echo all good
    fi

    echo "$mp3_file"
    rm -f output_video.mp4
    python3 moviemaker.py
    if [ -f "output_video.mp4" ]; then
        echo made_movie
        echo uploading_movie
        upload_marker="output_scripts/${key}_UPLOAD_COMPLETE.flag"
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
            failure_message="[$failure_time] YouTube upload failed after 3 fresh sessions for $key"
            echo "$failure_message" | tee -a logs/upload_failures.log
        fi
    else
        echo "moviemaker failed — skipping upload to avoid uploading stale video"
    fi
    echo "Done with the $edition edition ($key)."
}

echo "Waiting for the next show (morning 4:00 daily, evening 3:45 pm on weekdays)..."
while true; do
    due=$(python3 -m newscaster.schedule)
    if [ -n "$due" ]; then
        run_edition "$due"
        echo "Waiting for the next show (morning 4:00 daily, evening 3:45 pm on weekdays)..."
    fi
    sleep 60
done
