from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/'data'; MODEL_DIR=BASE_DIR/'models'; SAMPLE_DIR=BASE_DIR/'sample'; STATIC_DIR=BASE_DIR/'static'
for p in (DATA_DIR,MODEL_DIR,SAMPLE_DIR,STATIC_DIR): p.mkdir(parents=True,exist_ok=True)
RANDOM_STATE=42; EPOCHS=30; BATCH_SIZE=128; LEARNING_RATE=1e-3
FEATURES=['hour','login_count_24h','failed_logins_24h','session_minutes','files_accessed_24h','downloads_mb_24h','uploads_mb_24h','unique_ips_24h','new_devices_7d','admin_actions_24h','process_count_24h','network_connections_24h','geo_distance_km','password_changes_30d']
