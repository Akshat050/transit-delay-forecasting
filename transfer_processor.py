import pandas as pd
import numpy as np

class TransferDataBuilder:
    """
    Builds transfer candidate pairs (inbound->outbound) at shared stops and labels
    success/failure using simulated delays from processed stop_times.
    """

    def __init__(self, max_pairs=200000, min_window_min=2, max_window_min=20, alight_buffer_min=0):
        self.max_pairs = max_pairs
        self.min_window = pd.to_timedelta(min_window_min, unit='m')
        self.max_window = pd.to_timedelta(max_window_min, unit='m')
        self.alight_buffer = pd.to_timedelta(alight_buffer_min, unit='m')

    def build_pairs(self, stop_times: pd.DataFrame) -> pd.DataFrame:
        # Ensure required columns exist
        required_cols = [
            'stop_id', 'trip_id', 'route_id', 'route_id_code', 'stop_id_code',
            'arrival_time', 'departure_time', 'delay_minutes', 'arrival_hour', 'is_peak_hour',
            'dwell_time_seconds', 'total_stops', 'route_type', 'route_freq', 'stop_freq'
        ]
        missing = [c for c in required_cols if c not in stop_times.columns]
        if missing:
            raise ValueError(f"Missing required columns in stop_times: {missing}")

        # Work per stop to reduce cartesian blowup
        pairs = []
        for stop_id, grp in stop_times.groupby('stop_id', sort=True):
            g = grp.sort_values(['arrival_time', 'departure_time']).copy()
            # Inbound arrivals
            arrivals = g[['trip_id', 'route_id', 'route_id_code', 'arrival_time', 'delay_minutes', 'arrival_hour', 'is_peak_hour', 'dwell_time_seconds', 'total_stops', 'route_type', 'route_freq', 'stop_freq']].copy()
            arrivals = arrivals.rename(columns={
                'trip_id': 'in_trip_id', 'route_id': 'in_route_id', 'route_id_code': 'in_route_code',
                'arrival_time': 'in_arrival_time', 'delay_minutes': 'in_delay_min', 'arrival_hour': 'in_hour',
                'is_peak_hour': 'in_peak', 'dwell_time_seconds': 'in_dwell_s', 'total_stops': 'in_total_stops',
                'route_type': 'in_route_type', 'route_freq': 'in_route_freq', 'stop_freq': 'in_stop_freq'
            })
            # Outbound departures (exclude same trip)
            departures = g[['trip_id', 'route_id', 'route_id_code', 'departure_time', 'delay_minutes', 'arrival_hour', 'dwell_time_seconds', 'total_stops', 'route_type', 'route_freq', 'stop_freq']].copy()
            departures = departures.rename(columns={
                'trip_id': 'out_trip_id', 'route_id': 'out_route_id', 'route_id_code': 'out_route_code',
                'departure_time': 'out_departure_time', 'delay_minutes': 'out_delay_min', 'arrival_hour': 'out_hour',
                'dwell_time_seconds': 'out_dwell_s', 'total_stops': 'out_total_stops', 'route_type': 'out_route_type',
                'route_freq': 'out_route_freq', 'stop_freq': 'out_stop_freq'
            })

            # Merge-asof: find next departures after each arrival
            arrivals = arrivals.sort_values('in_arrival_time')
            departures = departures.sort_values('out_departure_time')
            m = pd.merge_asof(
                arrivals,
                departures,
                left_on='in_arrival_time',
                right_on='out_departure_time',
                direction='forward',
                tolerance=self.max_window
            )
            if m is None or len(m) == 0:
                continue
            # Drop where no departure found or same trip
            m = m.dropna(subset=['out_trip_id'])
            m = m[m['out_trip_id'] != m['in_trip_id']]
            # Window bounds
            wait_td = (m['out_departure_time'] - m['in_arrival_time'])
            m = m[(wait_td >= self.min_window) & (wait_td <= self.max_window)]
            if len(m) == 0:
                continue

            # Actual arrival/departure including simulated delays
            in_actual = m['in_arrival_time'] + pd.to_timedelta(m['in_delay_min'], unit='m') + self.alight_buffer
            out_actual_dep = m['out_departure_time'] + pd.to_timedelta(m['out_delay_min'], unit='m')
            # Success if inbound reaches before outbound departs
            m['transfer_success'] = (in_actual <= out_actual_dep).astype(int)
            m['wait_minutes_sched'] = wait_td.dt.total_seconds() / 60.0
            m['wait_minutes_effective'] = (out_actual_dep - (m['in_arrival_time'] + pd.to_timedelta(m['in_delay_min'], unit='m'))).dt.total_seconds() / 60.0
            m['stop_id'] = stop_id

            # Keep only necessary columns
            keep_cols = [
                'stop_id',
                'in_trip_id', 'in_route_id', 'in_route_code', 'in_hour', 'in_peak', 'in_dwell_s', 'in_total_stops', 'in_route_type', 'in_route_freq', 'in_stop_freq',
                'out_trip_id', 'out_route_id', 'out_route_code', 'out_hour', 'out_dwell_s', 'out_total_stops', 'out_route_type', 'out_route_freq', 'out_stop_freq',
                'wait_minutes_sched', 'wait_minutes_effective', 'transfer_success'
            ]
            pairs.append(m[keep_cols])

            # Cap total pairs
            if sum(len(p) for p in pairs) >= self.max_pairs:
                break

        if not pairs:
            return pd.DataFrame()
        df = pd.concat(pairs, ignore_index=True)

        # Feature engineering
        df['same_route'] = (df['in_route_id'] == df['out_route_id']).astype(int)
        df['hour_diff'] = (df['out_hour'] - df['in_hour']).astype('int32')
        # Encodings are already integer codes + freq

        # Downcast
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].astype('float32')

        # Suggested additional minutes needed to make effective wait >= 3 min (tunable)
        target_wait = 3.0  # minutes; could be parameterized
        df['suggested_offset_min'] = (target_wait - df['wait_minutes_effective']).clip(lower=0)

        return df 