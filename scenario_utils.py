import pandas as pd
import numpy as np

class ScenarioApplier:
    """Apply what-if schedule modifications to processed stop_times in-memory."""

    @staticmethod
    def add_dwell_buffer(stop_times: pd.DataFrame, route_id=None, segment_mask=None, minutes=1.0) -> pd.DataFrame:
        df = stop_times.copy()
        mask = pd.Series(True, index=df.index)
        if route_id is not None:
            mask &= (df['route_id'] == route_id)
        if segment_mask is not None:
            mask &= segment_mask(df)
        df.loc[mask, 'dwell_time_seconds'] += minutes * 60.0
        # Update normalized dwell
        df['dwell_time_normalized'] = df['dwell_time_seconds'] / 60.0
        # Recompute long_dwell flag
        df['long_dwell'] = (df['dwell_time_seconds'] > 60).astype(int)
        return df

    @staticmethod
    def shift_departures(stop_times: pd.DataFrame, route_id=None, minutes=1.0) -> pd.DataFrame:
        df = stop_times.copy()
        mask = (df['route_id'] == route_id) if route_id is not None else pd.Series(True, index=df.index)
        df.loc[mask, 'departure_time'] += pd.to_timedelta(minutes, unit='m')
        # Recompute travel_time_seconds to next stop
        df = df.sort_values(['trip_id', 'stop_sequence'])
        df['next_arrival_time'] = df.groupby('trip_id')['arrival_time'].shift(-1)
        df['travel_time_seconds'] = (df['next_arrival_time'] - df['departure_time']).dt.total_seconds()
        # Update speed
        df['speed_kmh'] = (df['distance_km'] / (df['travel_time_seconds'] / 3600)).replace([np.inf, -np.inf], 0)
        df['speed_normalized'] = df['speed_kmh'] / 50.0
        return df

    @staticmethod
    def change_headway(stop_times: pd.DataFrame, route_id, factor=0.9) -> pd.DataFrame:
        """Scale effective headway by factor; here we approximate by shifting alternating trips."""
        df = stop_times.copy()
        rmask = df['route_id'] == route_id
        # Shift every other trip by a small delta to emulate frequency change
        trip_ids = df.loc[rmask, 'trip_id'].unique()
        shift_ids = set(trip_ids[::2])
        delta_min = (1 - factor) * 5  # heuristic 5-min baseline headway
        idx = rmask & df['trip_id'].isin(shift_ids)
        df.loc[idx, 'departure_time'] += pd.to_timedelta(delta_min, unit='m')
        # Recompute downstream features
        df = df.sort_values(['trip_id', 'stop_sequence'])
        df['next_arrival_time'] = df.groupby('trip_id')['arrival_time'].shift(-1)
        df['travel_time_seconds'] = (df['next_arrival_time'] - df['departure_time']).dt.total_seconds()
        df['speed_kmh'] = (df['distance_km'] / (df['travel_time_seconds'] / 3600)).replace([np.inf, -np.inf], 0)
        df['speed_normalized'] = df['speed_kmh'] / 50.0
        return df 