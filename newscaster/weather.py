from datetime import datetime

import requests

import newscaster.config as _config


def get_daily_temp():
    url = f"http://api.openweathermap.org/data/2.5/forecast?id=5387877&appid={_config.OPENWEATHERMAP_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(data)
        daily_hi_temps = {}
        daily_low_temps = {}
        for item in data['list']:
            date_str = item['dt_txt'][:10]
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if date not in daily_hi_temps:
                daily_hi_temps[date] = []
                daily_low_temps[date] = []
            daily_hi_temps[date].append(item['main']['temp_max'])
            daily_low_temps[date].append(item['main']['temp_min'])

        for date in daily_hi_temps.keys():
            hi_temp = max(daily_hi_temps[date])
            low_temp = min(daily_low_temps[date])
            hi_temp_f = round((hi_temp - 273.15) * 9 / 5 + 32, 1)
            low_temp_f = round((low_temp - 273.15) * 9 / 5 + 32, 1)
            weather_string = f"the high in Riverside on {date.strftime('%B %d, %Y')} is {hi_temp_f} degrees and the low will be {low_temp_f}."
            return weather_string
    else:
        return 'the weather is not availible.'
