import csv
from flask import render_template

# ...existing code...

@app.route('/marketdemand/<crop_name>')
def market_demand(crop_name):
    market_data = []
    with open('marketdemand.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['crop'] == crop_name:
                market_data.append({
                    'region': row['region'],
                    'demand': row['demand'],
                    'price': row['price']
                })
    return render_template('marketdemand.html', crop_name=crop_name, market_data=market_data)

# ...existing code...