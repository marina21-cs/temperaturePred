"""
Temperature Prediction Web Application
=======================================
A Flask web application to display January 2026 temperature predictions
for Philippine cities with interactive visualizations.
"""

from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import os

app = Flask(__name__)

# Load data
def load_data():
    """Load all prediction data"""
    data = {}
    
    # Load city summary
    if os.path.exists('January2026_CitySummary.csv'):
        data['city_summary'] = pd.read_csv('January2026_CitySummary.csv')
    
    # Load daily predictions
    if os.path.exists('January2026_DailyPredictions.csv'):
        data['daily_predictions'] = pd.read_csv('January2026_DailyPredictions.csv')
    
    # Load hourly predictions
    if os.path.exists('January2026_HourlyPredictions.csv'):
        data['hourly_predictions'] = pd.read_csv('January2026_HourlyPredictions.csv')
    
    # Load accuracy report
    if os.path.exists('ModelAccuracyReport.csv'):
        data['accuracy'] = pd.read_csv('ModelAccuracyReport.csv')
    
    return data

@app.route('/')
def home():
    """Main dashboard page"""
    data = load_data()
    
    # Get summary statistics
    if 'city_summary' in data:
        df = data['city_summary']
        stats = {
            'total_cities': len(df),
            'avg_temp': round(df['january_avg_temp'].mean(), 1),
            'hottest_city': df.loc[df['january_avg_temp'].idxmax(), 'city_name'],
            'hottest_temp': round(df['january_avg_temp'].max(), 1),
            'coolest_city': df.loc[df['january_avg_temp'].idxmin(), 'city_name'],
            'coolest_temp': round(df['january_avg_temp'].min(), 1),
        }
    else:
        stats = {}
    
    # Get accuracy info
    if 'accuracy' in data:
        accuracy = data['accuracy'].iloc[0].to_dict()
    else:
        accuracy = {}
    
    return render_template('index.html', stats=stats, accuracy=accuracy)

@app.route('/api/cities')
def get_cities():
    """API endpoint to get all city data"""
    data = load_data()
    if 'city_summary' in data:
        return jsonify(data['city_summary'].to_dict(orient='records'))
    return jsonify([])

@app.route('/api/city/<city_name>')
def get_city_data(city_name):
    """API endpoint to get specific city daily predictions"""
    data = load_data()
    if 'daily_predictions' in data:
        city_data = data['daily_predictions'][
            data['daily_predictions']['city_name'] == city_name
        ]
        return jsonify(city_data.to_dict(orient='records'))
    return jsonify([])

@app.route('/api/stats')
def get_stats():
    """API endpoint to get overall statistics"""
    data = load_data()
    if 'city_summary' in data:
        df = data['city_summary']
        return jsonify({
            'total_cities': len(df),
            'avg_temp': round(df['january_avg_temp'].mean(), 2),
            'min_temp': round(df['january_min_temp'].min(), 2),
            'max_temp': round(df['january_max_temp'].max(), 2),
            'avg_humidity': round(df['january_avg_humidity'].mean(), 2)
        })
    return jsonify({})

@app.route('/images/<filename>')
def get_image(filename):
    """Serve generated images"""
    filepath = os.path.join(os.getcwd(), filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    return "Image not found", 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download CSV files"""
    filepath = os.path.join(os.getcwd(), filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
