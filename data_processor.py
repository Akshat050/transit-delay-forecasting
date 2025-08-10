import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GTFSDataProcessor:
    """
    Processes GTFS data for transit delay prediction
    """
    
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.routes = None
        self.stops = None
        self.trips = None
        self.stop_times = None
        self.calendar = None
        self.calendar_dates = None
        
    def load_gtfs_data(self):
        """Load all GTFS files"""
        print("Loading GTFS data...")
        
        # Load core files
        self.routes = pd.read_csv(f'{self.data_dir}/routes.txt')
        self.stops = pd.read_csv(f'{self.data_dir}/stops.txt')
        self.trips = pd.read_csv(f'{self.data_dir}/trips.txt')
        self.stop_times = pd.read_csv(f'{self.data_dir}/stop_times.txt')
        self.calendar = pd.read_csv(f'{self.data_dir}/calendar.txt')
        
        # Load optional files if they exist
        try:
            self.calendar_dates = pd.read_csv(f'{self.data_dir}/calendar_dates.txt')
        except FileNotFoundError:
            self.calendar_dates = None
            
        print(f"Loaded {len(self.routes)} routes, {len(self.stops)} stops, {len(self.trips)} trips")
        
    def preprocess_stop_times(self):
        """Preprocess stop_times data"""
        print("Preprocessing stop times...")
        
        # Clean and parse GTFS times which may have leading spaces and hours >= 24
        arrival_str = self.stop_times['arrival_time'].astype(str).str.strip()
        departure_str = self.stop_times['departure_time'].astype(str).str.strip()
        
        # Use to_timedelta to handle HH:MM:SS with hours >= 24
        self.stop_times['arrival_time'] = pd.to_timedelta(arrival_str, errors='coerce')
        self.stop_times['departure_time'] = pd.to_timedelta(departure_str, errors='coerce')
        
        # Replace unparsable times with 0 duration to avoid crashes
        zero_td = pd.to_timedelta(0, unit='s')
        self.stop_times['arrival_time'] = self.stop_times['arrival_time'].fillna(zero_td)
        self.stop_times['departure_time'] = self.stop_times['departure_time'].fillna(zero_td)
        
        # Extract hour/minute components (hours here are modulo 24, which is fine for peak/off-peak)
        arr_comp = self.stop_times['arrival_time'].dt.components
        dep_comp = self.stop_times['departure_time'].dt.components
        self.stop_times['arrival_hour'] = arr_comp['hours']
        self.stop_times['arrival_minute'] = arr_comp['minutes']
        self.stop_times['departure_hour'] = dep_comp['hours']
        self.stop_times['departure_minute'] = dep_comp['minutes']
        
        # Calculate dwell time (time spent at stop) in seconds
        self.stop_times['dwell_time_seconds'] = (
            self.stop_times['departure_time'] - self.stop_times['arrival_time']
        ).dt.total_seconds().clip(lower=0)
        
        print(f"Processed {len(self.stop_times)} stop time records")
        
    def merge_trip_data(self):
        """Merge trip information with stop times"""
        print("Merging trip data...")
        
        # Merge with trips to get route information
        self.stop_times = self.stop_times.merge(
            self.trips[['trip_id', 'route_id', 'service_id', 'direction_id']], 
            on='trip_id', how='left'
        )
        
        # Merge with routes to get route metadata
        self.stop_times = self.stop_times.merge(
            self.routes[['route_id', 'route_short_name', 'route_long_name', 'route_type']], 
            on='route_id', how='left'
        )
        
        # Merge with stops to get stop coordinates
        self.stop_times = self.stop_times.merge(
            self.stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], 
            on='stop_id', how='left'
        )
        
        print(f"Merged data contains {len(self.stop_times)} records")
        
    def calculate_trip_features(self):
        """Calculate trip-level features"""
        print("Calculating trip features...")
        
        # Sort by trip_id and stop_sequence
        self.stop_times = self.stop_times.sort_values(['trip_id', 'stop_sequence'])
        
        # Calculate travel time between consecutive stops
        self.stop_times['next_arrival_time'] = self.stop_times.groupby('trip_id')['arrival_time'].shift(-1)
        self.stop_times['travel_time_seconds'] = (
            self.stop_times['next_arrival_time'] - self.stop_times['departure_time']
        ).dt.total_seconds()
        
        # Calculate distance between stops (simplified using coordinates)
        self.stop_times['next_stop_lat'] = self.stop_times.groupby('trip_id')['stop_lat'].shift(-1)
        self.stop_times['next_stop_lon'] = self.stop_times.groupby('trip_id')['stop_lon'].shift(-1)
        
        # Calculate approximate distance using haversine formula
        self.stop_times['distance_km'] = self.stop_times.apply(
            lambda row: self._haversine_distance(
                row['stop_lat'], row['stop_lon'],
                row['next_stop_lat'], row['next_stop_lon']
            ) if pd.notna(row['next_stop_lat']) else 0, axis=1
        )
        
        # Calculate speed (km/h)
        self.stop_times['speed_kmh'] = (
            self.stop_times['distance_km'] / (self.stop_times['travel_time_seconds'] / 3600)
        ).replace([np.inf, -np.inf], 0)
        
        # Trip-level aggregations
        trip_features = self.stop_times.groupby('trip_id').agg({
            'stop_sequence': 'max',
            'travel_time_seconds': 'sum',
            'distance_km': 'sum',
            'dwell_time_seconds': 'sum'
        }).reset_index()
        
        trip_features.columns = ['trip_id', 'total_stops', 'total_travel_time', 'total_distance', 'total_dwell_time']
        
        # Merge back to stop_times
        self.stop_times = self.stop_times.merge(trip_features, on='trip_id', how='left')
        
        # Headway features at stop level (scheduled) within route-stop
        print("Calculating headway features...")
        self.stop_times = self.stop_times.sort_values(['route_id', 'stop_id', 'arrival_time'])
        self.stop_times['prev_arrival_time_rs'] = self.stop_times.groupby(['route_id', 'stop_id'])['arrival_time'].shift(1)
        self.stop_times['next_arrival_time_rs'] = self.stop_times.groupby(['route_id', 'stop_id'])['arrival_time'].shift(-1)
        self.stop_times['headway_prev_seconds'] = (self.stop_times['arrival_time'] - self.stop_times['prev_arrival_time_rs']).dt.total_seconds()
        self.stop_times['headway_next_seconds'] = (self.stop_times['next_arrival_time_rs'] - self.stop_times['arrival_time']).dt.total_seconds()
        self.stop_times['headway_prev_seconds'] = self.stop_times['headway_prev_seconds'].clip(lower=0).fillna(0)
        self.stop_times['headway_next_seconds'] = self.stop_times['headway_next_seconds'].clip(lower=0).fillna(0)
        
        # Terminal proximity (stops to line ends)
        self.stop_times['stops_to_terminal'] = self.stop_times[['stop_sequence', 'total_stops']].apply(
            lambda s: min(max(int(s['stop_sequence']) - 1, 0), max(int(s['total_stops']) - int(s['stop_sequence']), 0)), axis=1
        )
        self.stop_times['terminal_proximity_norm'] = 1 - (self.stop_times['stops_to_terminal'] / self.stop_times['total_stops'].replace(0, 1))
        
        # Stop density proxy: number of visits per stop (already have stop_freq). Also compute local sequence spacing
        self.stop_times['avg_segment_km'] = (self.stop_times['total_distance'] / self.stop_times['total_stops'].replace(0, 1)).fillna(0)
        
        # Cast new numeric columns to float32 to save memory
        for col in ['travel_time_seconds','distance_km','speed_kmh','headway_prev_seconds','headway_next_seconds','stops_to_terminal','terminal_proximity_norm','avg_segment_km']:
            self.stop_times[col] = self.stop_times[col].astype('float32')
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points"""
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return 0
            
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
        
    def create_delay_simulation(self, delay_probability=0.15, max_delay_minutes=10):
        """
        Simulate delays for training the model
        Since GTFS is static, we simulate realistic delays
        """
        print("Simulating delays...")
        
        # Base delay probability
        np.random.seed(42)  # For reproducibility
        
        # Create delay indicators based on various factors
        self.stop_times['simulated_delay'] = 0
        
        # Higher delay probability during peak hours (7-9 AM, 4-6 PM)
        peak_hours = (self.stop_times['arrival_hour'].isin([7, 8, 9, 16, 17, 18]))
        
        # Higher delay probability for longer trips
        long_trips = self.stop_times['total_stops'] > self.stop_times['total_stops'].median()
        
        # Higher delay probability for certain route types (buses vs trains)
        bus_routes = self.stop_times['route_type'] == 3
        
        # Higher delay probability for stops with longer dwell times
        long_dwell = self.stop_times['dwell_time_seconds'] > self.stop_times['dwell_time_seconds'].quantile(0.75)
        
        # Calculate delay probability based on factors
        delay_prob = np.where(
            peak_hours & long_trips & bus_routes & long_dwell,
            delay_probability * 3,  # High risk
            np.where(
                (peak_hours & bus_routes) | (long_trips & long_dwell),
                delay_probability * 2,  # Medium risk
                delay_probability  # Base risk
            )
        )
        
        # Generate delays
        delays = np.random.random(len(self.stop_times)) < delay_prob
        self.stop_times['simulated_delay'] = delays.astype(int)
        
        # Generate delay durations (1-10 minutes)
        delay_durations = np.random.randint(1, max_delay_minutes + 1, len(self.stop_times))
        self.stop_times['delay_minutes'] = np.where(delays, delay_durations, 0)
        
        # Calculate actual arrival time with delays
        self.stop_times['actual_arrival_time'] = self.stop_times['arrival_time'] + \
            pd.to_timedelta(self.stop_times['delay_minutes'], unit='minutes')
        
        # Calculate delay in seconds
        self.stop_times['delay_seconds'] = (
            self.stop_times['actual_arrival_time'] - self.stop_times['arrival_time']
        ).dt.total_seconds()
        
        print(f"Simulated delays for {self.stop_times['simulated_delay'].sum()} stops")
        
    def create_features(self):
        """Create features for machine learning model"""
        print("Creating ML features...")
        
        # Time-based features
        self.stop_times['is_peak_hour'] = self.stop_times['arrival_hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
        self.stop_times['is_weekend'] = 0  # We'll need calendar data for this
        
        # Route features
        self.stop_times['is_bus'] = (self.stop_times['route_type'] == 3).astype(int)
        self.stop_times['is_train'] = (self.stop_times['route_type'] == 1).astype(int)
        
        # Stop sequence features
        self.stop_times['is_first_stop'] = (self.stop_times['stop_sequence'] == 1).astype(int)
        self.stop_times['is_last_stop'] = (self.stop_times['stop_sequence'] == self.stop_times['total_stops']).astype(int)
        self.stop_times['stop_sequence_normalized'] = self.stop_times['stop_sequence'] / self.stop_times['total_stops']
        
        # Trip features
        self.stop_times['trip_length_category'] = pd.cut(
            self.stop_times['total_stops'], 
            bins=[0, 10, 20, 50, 100], 
            labels=['short', 'medium', 'long', 'very_long']
        )
        
        # Dwell time features
        self.stop_times['long_dwell'] = (self.stop_times['dwell_time_seconds'] > 60).astype(int)
        self.stop_times['dwell_time_normalized'] = self.stop_times['dwell_time_seconds'] / 60  # Convert to minutes
        
        # Speed features
        self.stop_times['slow_speed'] = (self.stop_times['speed_kmh'] < 20).astype(int)
        self.stop_times['speed_normalized'] = self.stop_times['speed_kmh'] / 50  # Normalize to 0-1
        
        # Create stop pair features (for embedding)
        self.stop_times['stop_pair'] = self.stop_times['stop_id'].astype(str) + '_' + \
            self.stop_times.groupby('trip_id')['stop_id'].shift(-1).astype(str)
        
        # High-cardinality encodings (avoid huge one-hot): label codes + frequency
        self.stop_times['route_id_code'] = self.stop_times['route_id'].astype('category').cat.codes.astype('int32')
        self.stop_times['stop_id_code'] = self.stop_times['stop_id'].astype('category').cat.codes.astype('int32')
        route_freq = self.stop_times['route_id'].value_counts()
        stop_freq = self.stop_times['stop_id'].value_counts()
        self.stop_times['route_freq'] = self.stop_times['route_id'].map(route_freq).astype('int32')
        self.stop_times['stop_freq'] = self.stop_times['stop_id'].map(stop_freq).astype('int32')
        
        print("Feature engineering completed")
        
    def prepare_ml_dataset(self):
        """Prepare final dataset for machine learning"""
        print("Preparing ML dataset...")
        
        # Select features for ML (use encoded + frequency for high-cardinality columns)
        feature_columns = [
            'route_id_code', 'stop_id_code', 'route_freq', 'stop_freq',
            'stop_sequence', 'stop_sequence_normalized',
            'arrival_hour', 'arrival_minute', 'departure_hour', 'departure_minute',
            'dwell_time_seconds', 'dwell_time_normalized', 'travel_time_seconds',
            'distance_km', 'speed_kmh', 'speed_normalized',
            'headway_prev_seconds', 'headway_next_seconds', 'terminal_proximity_norm', 'avg_segment_km',
            'total_stops', 'total_travel_time', 'total_distance', 'total_dwell_time',
            'is_peak_hour', 'is_bus', 'is_train', 'is_first_stop', 'is_last_stop',
            'long_dwell', 'slow_speed', 'trip_length_category'
        ]
        
        # Create target variable (delay > 3 minutes)
        self.stop_times['delay_target'] = (self.stop_times['delay_minutes'] > 3).astype(int)
        
        # Prepare features and target
        X = self.stop_times[feature_columns].copy()
        y = self.stop_times['delay_target']
        
        # Handle categorical variables (only small one: trip_length_category)
        categorical_cols = ['trip_length_category']
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Fill missing values
        X_encoded = X_encoded.fillna(0)
        
        # Downsample to keep memory reasonable (stratified by target)
        max_rows = 300000
        if len(X_encoded) > max_rows:
            pos_idx = y[y == 1].index
            neg_idx = y[y == 0].index
            pos_keep = int(max_rows * (len(pos_idx) / (len(pos_idx) + len(neg_idx))))
            pos_sample = pos_idx.to_series().sample(n=min(pos_keep, len(pos_idx)), random_state=42)
            neg_sample = neg_idx.to_series().sample(n=max_rows - len(pos_sample), random_state=42)
            keep_idx = pd.Index(pos_sample.tolist() + neg_sample.tolist())
            X_encoded = X_encoded.loc[keep_idx]
            y = y.loc[keep_idx]
        
        # Reduce numeric precision to save memory
        num_cols = X_encoded.select_dtypes(include=[np.number]).columns
        X_encoded[num_cols] = X_encoded[num_cols].astype('float32')
        
        print(f"ML dataset prepared: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
        
        return X_encoded, y
        
    def process_all(self):
        """Run complete data processing pipeline"""
        self.load_gtfs_data()
        self.preprocess_stop_times()
        self.merge_trip_data()
        self.calculate_trip_features()
        self.create_delay_simulation()
        self.create_features()
        
        return self.prepare_ml_dataset() 