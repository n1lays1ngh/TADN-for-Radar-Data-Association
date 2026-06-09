import os
import pickle
import hashlib
import numpy as np
import pandas as pd


class OpenSkyHistoricalTensorizer:
    def __init__(self, seed=42):
        """Initializes the historical dataset compiler."""
        self.R_earth = 6371000.0  # Earth's radius in meters
        self.rng = np.random.default_rng(seed)
        self.radar_lat = 0.0
        self.radar_lon = 0.0

    def _geodetic_to_local_enu(self, lat, lon):
        """Converts geodetic Lat/Lon rows into a flat local East-North-Up (ENU) plane."""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        ref_lat_rad = np.radians(self.radar_lat)
        ref_lon_rad = np.radians(self.radar_lon)

        dlat = lat_rad - ref_lat_rad
        dlon = lon_rad - ref_lon_rad

        x = self.R_earth * dlon * np.cos(ref_lat_rad)
        y = self.R_earth * dlat
        return x, y

    def _deterministic_hash(self, text_string):
        """Generates a stable, unique integer ID map for the Mode-S transponder hashes."""
        if not text_string or pd.isna(text_string):
            return -1.0
        return float(int(hashlib.md5(str(text_string).encode('utf-8')).hexdigest(), 16) % 100000)

    def process_historical_csv(self, file_path, sample_limit=10000, clutter_factor=3,
                               base_noise_std=135.0, max_time_lag=3.5, split_mode='all'):
        print(f"\nLoading historical OpenSky file: {file_path} ...")
        cols = ['time', 'icao24', 'lat', 'lon', 'velocity', 'heading', 'vertrate', 'squawk', 'geoaltitude', 'onground']

        # Load data and filter out grounded aircraft/missing coordinates
        df = pd.read_csv(file_path, usecols=cols)
        airborne_df = df[df['onground'] == False].dropna(subset=['lat', 'lon', 'velocity', 'heading'])

        # ── AUTODETECT & CROP SECTOR (Memory Fix) ──
        self.radar_lat = airborne_df['lat'].median()
        self.radar_lon = airborne_df['lon'].median()
        print(f"Sector Autodetected. Radar Origin set to: {self.radar_lat:.4f}, {self.radar_lon:.4f}")

        airborne_df = airborne_df[
            (airborne_df['lat'].between(self.radar_lat - 1.5, self.radar_lat + 1.5)) &
            (airborne_df['lon'].between(self.radar_lon - 1.5, self.radar_lon + 1.5))
            ]
        print(f"Sector cropped. Retained {len(airborne_df)} localized flight vectors.")

        # Sort chronologically to preserve true flight trajectories
        airborne_df = airborne_df.sort_values(by='time')
        unique_timesteps = airborne_df['time'].unique()

        # ── CURRICULUM TIME-SPLITTING ──
        n = len(unique_timesteps)
        if split_mode == 'vanilla':
            timesteps = unique_timesteps[:n // 3]
        elif split_mode == 'clutter':
            timesteps = unique_timesteps[n // 3: 2 * n // 3]
        elif split_mode == 'stress':
            timesteps = unique_timesteps[2 * n // 3:]
        else:
            timesteps = unique_timesteps

        print(f"Processing time partition: '{split_mode}' ({len(timesteps)} available frames)")

        dataset = []
        episode_count = 0

        for t_step in timesteps:
            if episode_count >= sample_limit:
                break

            frame_df = airborne_df[airborne_df['time'] == t_step]
            M = len(frame_df)

            # Skip sparsely populated frames
            if M < 2:
                continue

            detections_list = []
            tracks_list = []
            true_associations = []

            for idx, (_, row) in enumerate(frame_df.iterrows()):
                icao24, lat, lon = row['icao24'], row['lat'], row['lon']
                alt = row['geoaltitude'] if not pd.isna(row['geoaltitude']) else 0.0
                velocity, heading_rad = row['velocity'], np.radians(row['heading'])
                vz = row['vertrate'] if not pd.isna(row['vertrate']) else 0.0

                tx, ty = self._geodetic_to_local_enu(lat, lon)
                vx, vy = velocity * np.sin(heading_rad), velocity * np.cos(heading_rad)

                # --- CORRUPTION 1: Spatial Measurement Noise ---
                det_x = tx + self.rng.normal(0, base_noise_std)
                det_y = ty + self.rng.normal(0, base_noise_std)

                feat_d = np.zeros(13)
                feat_d[0:3] = [det_x, det_y, alt]
                feat_d[4], feat_d[5] = 1.0, 1.0
                feat_d[6], feat_d[7] = max(base_noise_std ** 2, 1.0), max(base_noise_std ** 2, 1.0)

                # Safe Squawk Code Parsing
                raw_squawk = row['squawk']
                feat_d[8] = float(int(float(raw_squawk)) % 10000) if not pd.isna(raw_squawk) else 0.0

                feat_d[11], feat_d[12] = 1.0, self._deterministic_hash(icao24)

                detections_list.append(feat_d)
                true_associations.append(idx)

                # --- CORRUPTION 3: Network Transmission Latency ---
                time_lag = self.rng.uniform(0.0, max_time_lag) if max_time_lag > 0 else 0.0
                feat_t = np.zeros(11)
                feat_t[0:3] = [tx - (vx * time_lag), ty - (vy * time_lag), alt - (vz * time_lag)]
                feat_t[3:6] = [vx, vy, vz]
                feat_t[6], feat_t[7], feat_t[10] = time_lag, 1.0, self._deterministic_hash(icao24)

                tracks_list.append(feat_t)

            # --- CORRUPTION 2: Dynamic Airspace Clutter ---
            for _ in range(clutter_factor):
                feat_c = np.zeros(13)
                feat_c[0:2] = self.rng.uniform(-60000, 60000, 2)
                feat_c[4], feat_c[6], feat_c[12] = 1.0, max(base_noise_std * 5.0, 10.0), -1.0

                detections_list.append(feat_c)
                true_associations.append(M)  # Map to null sink

            # Append T_null vector block onto index M
            t_null_vector = np.zeros(11)
            tracks_list.append(t_null_vector)

            # Construct PyTorch-ready Matrices
            D_in = np.array(detections_list, dtype=np.float32)
            T_in = np.array(tracks_list, dtype=np.float32)

            # Construct Assignment Label Matrix [N x M+1]
            A_gt = np.zeros((D_in.shape[0], M + 1), dtype=np.float32)
            for n, m in enumerate(true_associations):
                A_gt[n, m] = 1.0

            dataset.append({'D_in': D_in, 'T_in': T_in, 'A_gt': A_gt})
            episode_count += 1
            if episode_count % 1000 == 0:
                print(f" -> Compiled {episode_count} training matrices...")

        return dataset


# ── Processing Execution ─────────────────────────────────────
if __name__ == "__main__":
    tensorizer = OpenSkyHistoricalTensorizer()
    input_file = "data/states_2022-06-20-14.csv"
    os.makedirs('data', exist_ok=True)

    if os.path.exists(input_file):
        # Configuration Tuples: (filename, split_mode, clutter, noise, lag)
        configs = [
            ('phase1_clean', 'vanilla', 0, 1.0, 0.0),
            ('phase2_clutter', 'clutter', 4, 40.0, 0.0),
            ('phase3_stress', 'stress', 3, 140.0, 3.5)
        ]

        for name, mode, cf, bns, lag in configs:
            data = tensorizer.process_historical_csv(
                file_path=input_file,
                sample_limit=8000,
                split_mode=mode,
                clutter_factor=cf,
                base_noise_std=bns,
                max_time_lag=lag
            )

            with open(f'data/tadn_{name}.pkl', 'wb') as f:
                pickle.dump(data, f)

        print("\n[SUCCESS] Phase 1 Complete! All 3 Curriculum Datasets have been generated.")
    else:
        print(f"[ERROR] Could not find '{input_file}'. Ensure your OpenSky CSV is in the 'data' folder.")