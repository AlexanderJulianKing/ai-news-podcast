import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)


def print_and_write(*args):
    current_date = datetime.now()
    file_name = current_date.strftime("logs/log_%y_%m_%d.txt")
    current_time = current_date.strftime("%H:%M:%S")

    with open(file_name, 'a', encoding='utf-8') as file:
        for arg in args:
            log_entry = f"{current_time} - {arg}"
            file.write(log_entry + '\n')
            try:
                print(log_entry)
            except:
                print('looooooooooooooooool i have a hard time printing this line')
