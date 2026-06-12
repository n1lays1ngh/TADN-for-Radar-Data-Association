import os
import pickle
import hashlib
import numpy as np
import pandas as pd

INPUT_CSV = "data/Track_form.csv"
OUTPUT_PKL = "data/tadn_track_formation_test.pkl"


class StructuralTrackFormationGenerator:
    def __init__(self, seed=101):
        self.R_earth = 6371000.0
        self.rng = np.random.default_rng(seed)
        self.radar_lat = 0.0
        self.radar_lon = 0.0

    def _geodetic_to_local_enu(self, lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        ref_lat_rad = np.radians(self.radar_lat)
        ref_lon_rad = np.radians(self.radar_lon)
        return self.R_earth * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad), self.R_earth * (lat_rad - ref_lat_rad)

    def _deterministic_hash(self, text_string):
        if not text_string or pd.isna(text_string): return -1.0
        return float(int(hashlib.md5(str(text_string).encode('utf-8')).hexdigest(), 16) % 100000)

    def generate_formation_set(self, csv_path, noise_std=140.0, max_lag=3.5, formation_rate=0.30):
        print(f"[INGESTION] Reading raw historical database: {csv_path}")
        if not os.path.exists(csv_path): return None

        cols = ['time', 'icao24', 'lat', 'lon', 'velocity', 'heading', 'vertrate', 'squawk', 'geoaltitude', 'onground']
        df = pd.read_csv(csv_path, usecols=cols)
        airborne_df = df[df['onground'] == False].dropna(subset=['lat', 'lon', 'velocity', 'heading'])

        self.radar_lat, self.radar_lon = airborne_df['lat'].median(), airborne_df['lon'].median()
        airborne_df = airborne_df[
            (airborne_df['lat'].between(self.radar_lat - 1.5, self.radar_lat + 1.5)) &
            (airborne_df['lon'].between(self.radar_lon - 1.5, self.radar_lon + 1.5))
            ].sort_values(by='time')

        unique_timesteps = airborne_df['time'].unique()
        test_dataset = []
        historical_registry = set()

        for t_idx, t_step in enumerate(unique_timesteps):
            if t_idx >= 360: break
            frame_df = airborne_df[airborne_df['time'] == t_step]
            if len(frame_df) < 3: continue

            detections_list, final_active_tracks, true_associations = [], [], []
            track_id_to_col_idx = {}

            # Establish fixed structural mapping columns first
            for idx, (_, row) in enumerate(frame_df.iterrows()):
                track_id_to_col_idx[row['icao24']] = idx

            for idx, (_, row) in enumerate(frame_df.iterrows()):
                icao24 = row['icao24']
                lat, lon, alt = row['lat'], row['lon'], row['geoaltitude'] if not pd.isna(row['geoaltitude']) else 0.0
                vx, vy = row['velocity'] * np.sin(np.radians(row['heading'])), row['velocity'] * np.cos(
                    np.radians(row['heading']))
                vz = row['vertrate'] if not pd.isna(row['vertrate']) else 0.0
                tx, ty = self._geodetic_to_local_enu(lat, lon)

                # Formulate real noisy observation
                feat_d = np.zeros(13)
                feat_d[0:3] = [tx + self.rng.normal(0, noise_std), ty + self.rng.normal(0, noise_std), alt]
                feat_d[4], feat_d[5], feat_d[6], feat_d[7] = 1.0, 1.0, noise_std ** 2, noise_std ** 2
                feat_d[8] = float(int(float(row['squawk'])) % 10000) if not pd.isna(row['squawk']) else 0.0
                feat_d[11], feat_d[12] = 1.0, self._deterministic_hash(icao24)
                detections_list.append(feat_d)

                # Track Erasure Boundary Logic
                is_new_track = (icao24 not in historical_registry) or (self.rng.random() < formation_rate)
                feat_t = np.zeros(11)

                if is_new_track:
                    # STRUCTURAL FIX: Keep the row dimension intact, but zero-mask its tracking features
                    feat_t[10] = -1.0  # Set identity token to invalid clutter
                    true_associations.append(len(frame_df))  # Maps directly to the true final Null Sink column
                else:
                    time_lag = self.rng.uniform(0.0, max_lag)
                    feat_t[0:3] = [tx - (vx * time_lag), ty - (vy * time_lag), alt - (vz * time_lag)]
                    feat_t[3:6] = [vx, vy, vz]
                    feat_t[6], feat_t[7], feat_t[10] = time_lag, 1.0, self._deterministic_hash(icao24)
                    true_associations.append(track_id_to_col_idx[icao24])

                final_active_tracks.append(feat_t)
                historical_registry.add(icao24)

            # Append background noise clutter
            null_sink_idx = len(final_active_tracks)
            for _ in range(3):
                feat_c = np.zeros(13)
                feat_c[0:2] = self.rng.uniform(-60000, 60000, 2)
                feat_c[4], feat_c[6], feat_c[12] = 1.0, noise_std * 5.0, -1.0
                detections_list.append(feat_c)
                true_associations.append(null_sink_idx)

            final_active_tracks.append(np.zeros(11))  # Append true Null Sink column vector at index M

            A_gt = np.zeros((len(detections_list), null_sink_idx + 1), dtype=np.float32)
            for n, m in enumerate(true_associations): A_gt[n, m] = 1.0

            test_dataset.append({
                'D_in': np.array(detections_list, dtype=np.float32),
                'T_in': np.array(final_active_tracks, dtype=np.float32),
                'A_gt': A_gt
            })

        print(f"[SUCCESS] Compiled {len(test_dataset)} structurally stable matrix entries.")
        return test_dataset


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    compiled_data = StructuralTrackFormationGenerator().generate_formation_set(INPUT_CSV)
    if compiled_data:
        with open(OUTPUT_PKL, 'wb') as f: pickle.dump(compiled_data, f)