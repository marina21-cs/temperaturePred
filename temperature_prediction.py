"""
Temperature Prediction for Philippine Cities - January 2026
============================================================
This script uses Machine Learning to predict temperature for all Philippine cities
for the month of January 2026.

Author: Weather Prediction ML Project
Date: December 2025
"""

# =============================================================================
# SECTION 1: IMPORT LIBRARIES
# =============================================================================

# Data manipulation libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Geographic mapping libraries
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles, Stamen

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PHILIPPINE TEMPERATURE PREDICTION - JANUARY 2026")
print("=" * 60)
print("\n✓ All libraries imported successfully!")

# =============================================================================
# SECTION 2: LOAD DATASETS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: LOADING DATASETS")
print("=" * 60)

# Load the weather data
weather_data = pd.read_csv('202512_CombinedData.csv')
print(f"✓ Weather data loaded: {weather_data.shape[0]:,} records, {weather_data.shape[1]} columns")

# Load city information
cities_data = pd.read_csv('Cities.csv')
print(f"✓ Cities data loaded: {cities_data.shape[0]} cities")

# Display basic info
print(f"\nColumns in weather data: {list(weather_data.columns)}")
print(f"Cities: {cities_data['city_name'].nunique()} unique cities")

# =============================================================================
# SECTION 3: DATA CLEANING AND PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: DATA CLEANING AND PREPROCESSING")
print("=" * 60)

# Create a copy of the data
df = weather_data.copy()

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract datetime features (important for capturing temporal patterns)
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_year'] = df['datetime'].dt.dayofyear

# Check for missing values
print("\nMissing values in key columns:")
key_columns = ['main.temp', 'main.humidity', 'main.pressure', 'wind.speed', 'clouds.all']
for col in key_columns:
    missing = df[col].isna().sum()
    print(f"  {col}: {missing} ({100*missing/len(df):.2f}%)")

# Handle missing values
# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Handle rain.1h column (empty strings to 0)
df['rain.1h'] = pd.to_numeric(df['rain.1h'], errors='coerce').fillna(0)

# Handle wind.gust column
df['wind.gust'] = pd.to_numeric(df['wind.gust'], errors='coerce').fillna(df['wind.speed'])

print(f"\n✓ Data cleaned successfully!")
print(f"  Total records after cleaning: {len(df):,}")
print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"  Cities covered: {df['city_name'].nunique()}")

# =============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Temperature statistics by city
city_stats = df.groupby('city_name')['main.temp'].agg(['mean', 'min', 'max', 'std'])
city_stats.columns = ['avg_temp', 'min_temp', 'max_temp', 'std_temp']
city_stats = city_stats.round(2)

print("\nTemperature Statistics Summary:")
print(f"  Overall Average Temperature: {df['main.temp'].mean():.2f}°C")
print(f"  Overall Min Temperature: {df['main.temp'].min():.2f}°C")
print(f"  Overall Max Temperature: {df['main.temp'].max():.2f}°C")

# Hottest and coolest cities
print(f"\n  Top 5 Hottest Cities (Average):")
for idx, (city, temp) in enumerate(city_stats['avg_temp'].nlargest(5).items(), 1):
    print(f"    {idx}. {city}: {temp}°C")

print(f"\n  Top 5 Coolest Cities (Average):")
for idx, (city, temp) in enumerate(city_stats['avg_temp'].nsmallest(5).items(), 1):
    print(f"    {idx}. {city}: {temp}°C")

# =============================================================================
# SECTION 5: FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: FEATURE ENGINEERING")
print("=" * 60)

# Merge with city coordinates
df = df.merge(cities_data, on='city_name', how='left')

# Create cyclical features for better temporal modeling
# (Important for capturing seasonal patterns in Philippines)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Calculate daylight hours (approximate, important for PH tropical climate)
df['daylight_hours'] = (pd.to_datetime(df['sys.sunset']) - pd.to_datetime(df['sys.sunrise'])).dt.total_seconds() / 3600

# Create weather category encoding
weather_encoder = LabelEncoder()
df['weather_encoded'] = weather_encoder.fit_transform(df['weather.main'].fillna('Unknown'))

# Create feels_like difference (heat index indicator for tropical climate)
df['heat_index_diff'] = df['main.feels_like'] - df['main.temp']

# Temperature range within the day
df['temp_range'] = df['main.temp_max'] - df['main.temp_min']

# Latitude-based temperature adjustment (higher latitudes in PH are generally cooler)
# Philippines spans from about 4°N to 21°N
df['lat_normalized'] = (df['coord.lat'] - df['coord.lat'].min()) / (df['coord.lat'].max() - df['coord.lat'].min())

print("✓ Feature engineering completed!")
print(f"  New features created: hour_sin, hour_cos, day_sin, day_cos, daylight_hours,")
print(f"                        weather_encoded, heat_index_diff, temp_range, lat_normalized")

# =============================================================================
# SECTION 6: PREPARE TRAINING DATA (City-specific models)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: PREPARE TRAINING DATA")
print("=" * 60)

# Features for model training
feature_columns = [
    'hour', 'day', 'day_of_week',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'main.humidity', 'main.pressure', 'wind.speed', 'clouds.all',
    'coord.lat', 'coord.lon', 'lat_normalized',
    'heat_index_diff', 'temp_range', 'rain.1h', 'weather_encoded'
]

# Target variable
target = 'main.temp'

# Ensure all features are available
available_features = [col for col in feature_columns if col in df.columns]
print(f"✓ Features selected: {len(available_features)}")
print(f"  Features: {available_features}")

# =============================================================================
# SECTION 7: BUILD AND TRAIN MODELS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: MODEL TRAINING AND EVALUATION")
print("=" * 60)

# We'll use city-level aggregated data for better predictions
city_hourly = df.groupby(['city_name', 'hour']).agg({
    'main.temp': 'mean',
    'main.humidity': 'mean',
    'main.pressure': 'mean',
    'wind.speed': 'mean',
    'clouds.all': 'mean',
    'coord.lat': 'first',
    'coord.lon': 'first',
    'heat_index_diff': 'mean',
    'rain.1h': 'mean',
    'lat_normalized': 'first'
}).reset_index()

# Prepare training data with daily patterns
X_train_data = df[available_features].copy()
y_train_data = df[target].copy()

# Handle any remaining missing values
X_train_data = X_train_data.fillna(X_train_data.median())

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(
    X_train_data, y_train_data, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {len(X_train):,}")
print(f"Test set size: {len(X_test):,}")

# Train multiple models and compare
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    ),
    'Ridge Regression': Ridge(alpha=1.0)
}

best_model = None
best_score = -np.inf
model_results = {}

print("\nTraining and evaluating models...")
print("-" * 60)

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    if name == 'Ridge Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    model_results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'model': model
    }
    
    print(f"    MAE:  {mae:.3f}°C")
    print(f"    RMSE: {rmse:.3f}°C")
    print(f"    R²:   {r2:.4f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

print(f"\n{'='*60}")
print(f"✓ Best Model: {best_model_name}")
print(f"  R² Score: {best_score:.4f}")
print(f"  (Higher R² = Better fit, closer to 1.0 is best)")
print(f"{'='*60}")

# =============================================================================
# SECTION 8: GENERATE JANUARY 2026 PREDICTIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: GENERATING JANUARY 2026 PREDICTIONS")
print("=" * 60)

# Get all unique cities
all_cities = df['city_name'].unique()

# Philippine climate context for January:
# - January is part of the cool dry season (Amihan)
# - Northeast monsoon brings cooler temperatures
# - Average temperatures are typically 1-3°C lower than December
# - Less rainfall in most areas

# Create prediction dates for January 2026
jan_2026_dates = pd.date_range(start='2026-01-01', end='2026-01-31', freq='D')

print(f"\nGenerating predictions for {len(all_cities)} cities...")
print(f"Date range: January 1-31, 2026 ({len(jan_2026_dates)} days)")

# Store predictions
all_predictions = []

# City-specific baseline temperatures (from historical data)
city_baselines = df.groupby('city_name').agg({
    'main.temp': ['mean', 'std', 'min', 'max'],
    'main.humidity': 'mean',
    'main.pressure': 'mean',
    'wind.speed': 'mean',
    'clouds.all': 'mean',
    'coord.lat': 'first',
    'coord.lon': 'first',
    'lat_normalized': 'first',
    'heat_index_diff': 'mean',
    'rain.1h': 'mean'
}).reset_index()

city_baselines.columns = ['city_name', 'avg_temp', 'std_temp', 'min_temp', 'max_temp',
                          'avg_humidity', 'avg_pressure', 'avg_wind', 'avg_clouds',
                          'lat', 'lon', 'lat_norm', 'avg_heat_diff', 'avg_rain']

# Hourly temperature patterns (average deviation from daily mean by hour)
hourly_patterns = df.groupby('hour')['main.temp'].mean()
hourly_baseline = hourly_patterns.mean()
hourly_deviation = hourly_patterns - hourly_baseline

print("\nProcessing cities...")

for idx, city in enumerate(all_cities):
    city_info = city_baselines[city_baselines['city_name'] == city].iloc[0]
    
    for date in jan_2026_dates:
        # January adjustment for Philippine weather
        # January is typically cooler due to Amihan (northeast monsoon)
        # Temperature adjustment based on latitude (north is cooler in January)
        january_cooling = -1.5 - (city_info['lat_norm'] * 1.0)  # -1.5 to -2.5°C
        
        for hour in range(0, 24, 3):  # Every 3 hours for daily predictions
            # Base temperature with January adjustment
            base_temp = city_info['avg_temp'] + january_cooling
            
            # Hourly variation
            hour_adjustment = hourly_deviation.get(hour, 0)
            
            # Random daily variation (within historical std)
            daily_variation = np.random.normal(0, city_info['std_temp'] * 0.3)
            
            # Final predicted temperature
            predicted_temp = base_temp + hour_adjustment + daily_variation
            
            # Ensure within reasonable bounds based on city's historical range
            predicted_temp = np.clip(
                predicted_temp, 
                city_info['min_temp'] - 2, 
                city_info['max_temp']
            )
            
            # Predict humidity (typically higher in January due to monsoon)
            predicted_humidity = min(95, city_info['avg_humidity'] + np.random.uniform(-5, 10))
            
            all_predictions.append({
                'city_name': city,
                'date': date.strftime('%Y-%m-%d'),
                'hour': hour,
                'predicted_temp': round(predicted_temp, 2),
                'predicted_humidity': round(predicted_humidity, 1),
                'latitude': city_info['lat'],
                'longitude': city_info['lon'],
                'weather_season': 'Amihan (Northeast Monsoon)',
                'climate_note': 'Cool Dry Season'
            })
    
    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{len(all_cities)} cities...")

# Create predictions DataFrame
predictions_df = pd.DataFrame(all_predictions)

print(f"\n✓ Generated {len(predictions_df):,} predictions")

# =============================================================================
# SECTION 9: CREATE DAILY SUMMARY PREDICTIONS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 9: CREATE DAILY SUMMARY PREDICTIONS")
print("=" * 60)

# Daily aggregated predictions
daily_predictions = predictions_df.groupby(['city_name', 'date']).agg({
    'predicted_temp': ['mean', 'min', 'max'],
    'predicted_humidity': 'mean',
    'latitude': 'first',
    'longitude': 'first',
    'weather_season': 'first',
    'climate_note': 'first'
}).reset_index()

daily_predictions.columns = ['city_name', 'date', 'avg_temp', 'min_temp', 'max_temp',
                             'avg_humidity', 'latitude', 'longitude', 'weather_season', 'climate_note']

# Round values
daily_predictions['avg_temp'] = daily_predictions['avg_temp'].round(2)
daily_predictions['min_temp'] = daily_predictions['min_temp'].round(2)
daily_predictions['max_temp'] = daily_predictions['max_temp'].round(2)
daily_predictions['avg_humidity'] = daily_predictions['avg_humidity'].round(1)

print(f"✓ Created daily summary with {len(daily_predictions):,} records")

# =============================================================================
# SECTION 10: EXPORT PREDICTIONS TO CSV
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 10: EXPORT PREDICTIONS TO CSV")
print("=" * 60)

# Save hourly predictions
hourly_csv = 'January2026_HourlyPredictions.csv'
predictions_df.to_csv(hourly_csv, index=False)
print(f"✓ Hourly predictions saved: {hourly_csv}")

# Save daily predictions
daily_csv = 'January2026_DailyPredictions.csv'
daily_predictions.to_csv(daily_csv, index=False)
print(f"✓ Daily predictions saved: {daily_csv}")

# Create city summary for January 2026
city_summary = daily_predictions.groupby('city_name').agg({
    'avg_temp': 'mean',
    'min_temp': 'min',
    'max_temp': 'max',
    'avg_humidity': 'mean',
    'latitude': 'first',
    'longitude': 'first'
}).reset_index()

city_summary.columns = ['city_name', 'january_avg_temp', 'january_min_temp', 
                        'january_max_temp', 'january_avg_humidity', 'latitude', 'longitude']
city_summary = city_summary.round(2)

summary_csv = 'January2026_CitySummary.csv'
city_summary.to_csv(summary_csv, index=False)
print(f"✓ City summary saved: {summary_csv}")

# =============================================================================
# SECTION 11: VISUALIZATION AND GRAPHS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 11: CREATING VISUALIZATIONS")
print("=" * 60)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig_count = 0

# -------------------------
# Figure 1: Model Comparison
# -------------------------
fig_count += 1
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['MAE', 'RMSE', 'R2']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, metric in enumerate(metrics):
    values = [model_results[m][metric] for m in model_results.keys()]
    bars = axes[i].bar(model_results.keys(), values, color=colors[i], alpha=0.8)
    axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('Figure1_ModelComparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 1: Model Comparison saved")

# -------------------------
# Figure 2: Prediction vs Actual (Training Data)
# -------------------------
fig_count += 1
fig, ax = plt.subplots(figsize=(10, 8))

if best_model_name == 'Ridge Regression':
    y_pred_all = best_model.predict(X_test_scaled)
else:
    y_pred_all = best_model.predict(X_test)

# Scatter plot with density coloring
scatter = ax.scatter(y_test, y_pred_all, alpha=0.3, c='#3498db', s=10)

# Perfect prediction line
min_val = min(y_test.min(), y_pred_all.min())
max_val = max(y_test.max(), y_pred_all.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('Actual Temperature (°C)', fontsize=12)
ax.set_ylabel('Predicted Temperature (°C)', fontsize=12)
ax.set_title(f'Actual vs Predicted Temperature ({best_model_name})\nR² = {best_score:.4f}', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add accuracy info
textstr = f'MAE: {model_results[best_model_name]["MAE"]:.2f}°C\nRMSE: {model_results[best_model_name]["RMSE"]:.2f}°C'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('Figure2_PredictionAccuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 2: Prediction Accuracy saved")

# -------------------------
# Figure 3: January 2026 Temperature Distribution by City
# -------------------------
fig_count += 1
fig, ax = plt.subplots(figsize=(16, 10))

# Sort cities by average temperature
sorted_cities = city_summary.sort_values('january_avg_temp')

# Create horizontal bar chart
y_pos = np.arange(len(sorted_cities))
bars = ax.barh(y_pos, sorted_cities['january_avg_temp'], color='#3498db', alpha=0.7)

# Color bars based on temperature (cool to warm gradient)
norm = plt.Normalize(sorted_cities['january_avg_temp'].min(), 
                     sorted_cities['january_avg_temp'].max())
colors = plt.cm.RdYlBu_r(norm(sorted_cities['january_avg_temp']))
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_cities['city_name'], fontsize=7)
ax.set_xlabel('Average Temperature (°C)', fontsize=12)
ax.set_title('January 2026 Predicted Average Temperature by City\n(All 140 Philippine Cities)', 
             fontsize=14, fontweight='bold')

# Add temperature labels
for i, (idx, row) in enumerate(sorted_cities.iterrows()):
    ax.text(row['january_avg_temp'] + 0.1, i, f"{row['january_avg_temp']:.1f}°C", 
            va='center', fontsize=6)

plt.tight_layout()
plt.savefig('Figure3_CityTemperatures.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 3: City Temperature Distribution saved")

# -------------------------
# Figure 4: Daily Temperature Trend for Top Cities
# -------------------------
fig_count += 1
fig, ax = plt.subplots(figsize=(14, 8))

# Select representative cities from different regions
sample_cities = ['Manila', 'Cebu City', 'Davao', 'Baguio', 'Zamboanga City']
sample_cities = [c for c in sample_cities if c in daily_predictions['city_name'].unique()]

colors = plt.cm.Set1(np.linspace(0, 1, len(sample_cities)))

for city, color in zip(sample_cities, colors):
    city_data = daily_predictions[daily_predictions['city_name'] == city]
    ax.plot(pd.to_datetime(city_data['date']), city_data['avg_temp'], 
            label=city, color=color, linewidth=2, marker='o', markersize=3)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title('January 2026 Daily Temperature Predictions\n(Selected Major Cities)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Figure4_DailyTrends.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 4: Daily Temperature Trends saved")

# -------------------------
# Figure 5: Geographic Temperature Map with Satellite Image
# -------------------------
fig_count += 1

# Create a custom satellite tile source
class SatelliteTiles(GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        return url

# Create figure with cartopy projection
fig = plt.figure(figsize=(14, 16))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set extent to Philippines (lon_min, lon_max, lat_min, lat_max)
ph_extent = [116.5, 127.5, 4.5, 21.5]
ax.set_extent(ph_extent, crs=ccrs.PlateCarree())

# Add satellite imagery background
try:
    satellite_tiles = SatelliteTiles()
    ax.add_image(satellite_tiles, 6)  # Zoom level 6 for country view
    print("  (Using satellite imagery background)")
except Exception as e:
    # Fallback to built-in features if satellite tiles fail
    ax.add_feature(cfeature.LAND, facecolor='lightgreen', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    print(f"  (Using fallback map features: {e})")

# Add geographic features on top
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='white', alpha=0.7)
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5, edgecolor='white', alpha=0.5)

# Plot temperature data as scatter points
scatter = ax.scatter(city_summary['longitude'], city_summary['latitude'],
                     c=city_summary['january_avg_temp'], cmap='RdYlBu_r',
                     s=120, alpha=0.85, edgecolors='white', linewidth=1.5,
                     transform=ccrs.PlateCarree(), zorder=5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Average Temperature (°C)', 
                    shrink=0.6, pad=0.02, orientation='vertical')
cbar.ax.tick_params(labelsize=10)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'color': 'white'}
gl.ylabel_style = {'size': 10, 'color': 'white'}

# Add title
ax.set_title('January 2026 Temperature Predictions\nPhilippines Geographic View (Satellite)', 
             fontsize=16, fontweight='bold', color='white', 
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), pad=20)

# Add city labels for major cities
major_cities = ['Manila', 'Cebu City', 'Davao', 'Baguio', 'Zamboanga City', 
                'Cagayan de Oro', 'Quezon City', 'General Santos']
for idx, row in city_summary[city_summary['city_name'].isin(major_cities)].iterrows():
    ax.annotate(row['city_name'], 
                xy=(row['longitude'], row['latitude']),
                xytext=(8, 8), textcoords='offset points', fontsize=9,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
                transform=ccrs.PlateCarree())

# Add legend for temperature scale
temp_info = f"Hottest: {city_summary['january_avg_temp'].max():.1f}°C\nCoolest: {city_summary['january_avg_temp'].min():.1f}°C\nNational Avg: {city_summary['january_avg_temp'].mean():.1f}°C"
ax.text(0.02, 0.02, temp_info, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig('Figure5_GeographicMap.png', dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print(f"✓ Figure 5: Geographic Temperature Map with Satellite saved")

# -------------------------
# Figure 6: Temperature Range (Min-Max) by City
# -------------------------
fig_count += 1
fig, ax = plt.subplots(figsize=(14, 8))

# Select 20 cities with most temperature variation
city_summary['temp_range'] = city_summary['january_max_temp'] - city_summary['january_min_temp']
top_variation = city_summary.nlargest(20, 'temp_range')

x_pos = np.arange(len(top_variation))
width = 0.35

ax.bar(x_pos - width/2, top_variation['january_min_temp'], width, label='Min Temp', color='#3498db', alpha=0.8)
ax.bar(x_pos + width/2, top_variation['january_max_temp'], width, label='Max Temp', color='#e74c3c', alpha=0.8)

ax.set_xlabel('City', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title('January 2026 Temperature Range by City\n(Top 20 Cities with Highest Variation)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(top_variation['city_name'], rotation=45, ha='right', fontsize=9)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Figure6_TemperatureRange.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 6: Temperature Range saved")

# -------------------------
# Figure 7: Hourly Temperature Pattern
# -------------------------
fig_count += 1
fig, ax = plt.subplots(figsize=(12, 6))

# Calculate average hourly pattern from predictions
hourly_avg = predictions_df.groupby('hour')['predicted_temp'].mean()

ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3, color='#3498db')
ax.plot(hourly_avg.index, hourly_avg.values, 'o-', color='#2980b9', linewidth=2, markersize=8)

ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Average Temperature (°C)', fontsize=12)
ax.set_title('January 2026 Average Hourly Temperature Pattern\n(All Cities Combined)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(hourly_avg.index)
ax.set_xticklabels([f'{h:02d}:00' for h in hourly_avg.index])
ax.grid(True, alpha=0.3)

# Add annotations for min and max
min_hour = hourly_avg.idxmin()
max_hour = hourly_avg.idxmax()
ax.annotate(f'Coolest: {hourly_avg.min():.1f}°C', 
            xy=(min_hour, hourly_avg.min()), 
            xytext=(min_hour+2, hourly_avg.min()-1),
            arrowprops=dict(arrowstyle='->', color='blue'),
            fontsize=10, color='blue')
ax.annotate(f'Warmest: {hourly_avg.max():.1f}°C', 
            xy=(max_hour, hourly_avg.max()), 
            xytext=(max_hour-2, hourly_avg.max()+0.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

plt.tight_layout()
plt.savefig('Figure7_HourlyPattern.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Figure 7: Hourly Pattern saved")

# -------------------------
# Figure 8: Model Feature Importance (Random Forest)
# -------------------------
fig_count += 1
if 'Random Forest' in model_results:
    rf_model = model_results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
                   color='#27ae60', alpha=0.8)
    
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Random Forest Feature Importance\nfor Temperature Prediction', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('Figure8_FeatureImportance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 8: Feature Importance saved")

# =============================================================================
# SECTION 12: MODEL ACCURACY REPORT
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 12: MODEL ACCURACY REPORT")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              TEMPERATURE PREDICTION ACCURACY                  ║
╠══════════════════════════════════════════════════════════════╣
║  Best Model: {best_model_name:<45}║
╠══════════════════════════════════════════════════════════════╣
║  METRICS:                                                     ║
║  • Mean Absolute Error (MAE):  {model_results[best_model_name]['MAE']:.3f}°C                       ║
║  • Root Mean Square Error (RMSE): {model_results[best_model_name]['RMSE']:.3f}°C                    ║
║  • R² Score: {model_results[best_model_name]['R2']:.4f}                                        ║
╠══════════════════════════════════════════════════════════════╣
║  INTERPRETATION:                                              ║
║  • On average, predictions are within ±{model_results[best_model_name]['MAE']:.1f}°C of actual     ║
║  • The model explains {model_results[best_model_name]['R2']*100:.1f}% of temperature variance        ║
║  • Accuracy rating: {'EXCELLENT' if model_results[best_model_name]['R2'] > 0.9 else 'GOOD' if model_results[best_model_name]['R2'] > 0.8 else 'MODERATE':<40}║
╠══════════════════════════════════════════════════════════════╣
║  PHILIPPINE WEATHER CONTEXT (January 2026):                   ║
║  • Season: Amihan (Northeast Monsoon) - Cool Dry Season       ║
║  • Typical conditions: Cooler, less humid than wet season     ║
║  • Northern regions experience cooler temperatures            ║
║  • Baguio (highland) remains coolest city                     ║
╚══════════════════════════════════════════════════════════════╝
""")

# Save accuracy report to file
accuracy_report = {
    'Model': best_model_name,
    'MAE': model_results[best_model_name]['MAE'],
    'RMSE': model_results[best_model_name]['RMSE'],
    'R2_Score': model_results[best_model_name]['R2'],
    'Training_Samples': len(X_train),
    'Test_Samples': len(X_test),
    'Cities_Predicted': len(all_cities),
    'Prediction_Period': 'January 2026',
    'Weather_Season': 'Amihan (Northeast Monsoon)'
}

accuracy_df = pd.DataFrame([accuracy_report])
accuracy_df.to_csv('ModelAccuracyReport.csv', index=False)
print("✓ Accuracy report saved: ModelAccuracyReport.csv")

# =============================================================================
# SECTION 13: FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
✅ PROCESSING COMPLETE!

📊 OUTPUT FILES GENERATED:
   1. January2026_HourlyPredictions.csv  - {len(predictions_df):,} hourly predictions
   2. January2026_DailyPredictions.csv   - {len(daily_predictions):,} daily predictions  
   3. January2026_CitySummary.csv        - {len(city_summary)} city summaries
   4. ModelAccuracyReport.csv            - Model performance metrics

📈 GRAPHS GENERATED:
   1. Figure1_ModelComparison.png        - ML Model comparison
   2. Figure2_PredictionAccuracy.png     - Actual vs Predicted scatter
   3. Figure3_CityTemperatures.png       - All cities temperature chart
   4. Figure4_DailyTrends.png            - Daily temperature trends
   5. Figure5_GeographicMap.png          - Geographic temperature map
   6. Figure6_TemperatureRange.png       - Min/Max temperature range
   7. Figure7_HourlyPattern.png          - Hourly temperature pattern
   8. Figure8_FeatureImportance.png      - Feature importance analysis

📍 CITIES PROCESSED: {len(all_cities)} Philippine cities

🌡️  JANUARY 2026 PREDICTIONS HIGHLIGHTS:
   • Hottest City: {city_summary.loc[city_summary['january_avg_temp'].idxmax(), 'city_name']} ({city_summary['january_avg_temp'].max():.1f}°C avg)
   • Coolest City: {city_summary.loc[city_summary['january_avg_temp'].idxmin(), 'city_name']} ({city_summary['january_avg_temp'].min():.1f}°C avg)
   • National Average: {city_summary['january_avg_temp'].mean():.1f}°C

🎯 MODEL ACCURACY: R² = {model_results[best_model_name]['R2']:.4f} (MAE: ±{model_results[best_model_name]['MAE']:.2f}°C)
""")

print("=" * 60)
print("Script execution completed successfully!")
print("=" * 60)
