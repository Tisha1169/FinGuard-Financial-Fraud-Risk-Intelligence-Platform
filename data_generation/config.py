"""Central knobs for the synthetic data generator.

Sizing is deliberately kept modest (tens of thousands of transactions, not
millions) so the resulting dataset stays well within Neon's free-tier
storage limits once loaded.
"""
import datetime

SEED = 42

NUM_CUSTOMERS = 1000
NUM_MERCHANTS = 150
NUM_LOCATIONS = 25          # drawn from CITIES below
DEVICES_PER_CUSTOMER_MEAN = 1.3  # most customers use 1 device, some 2-3

SIMULATION_DAYS = 120
END_DATE = datetime.date(2026, 8, 21)
START_DATE = END_DATE - datetime.timedelta(days=SIMULATION_DAYS)

# Target share of transactions that are part of an injected fraud typology.
# Real-world card fraud rates are usually well under 1%; this project uses a
# higher rate than that so the labeled fraud set is large enough to train
# and evaluate a model on, and states this explicitly as a synthetic-data
# limitation (see docs/model_card.md, added in Phase 6).
TARGET_FRAUD_TXN_SHARE = 0.02

MCC_CATEGORIES = [
    ("5411", "GROCERY"),
    ("5812", "RESTAURANT"),
    ("5732", "ELECTRONICS"),
    ("4511", "AIRLINE"),
    ("5541", "GAS_STATION"),
    ("5999", "ONLINE_RETAIL"),
    ("5944", "JEWELRY"),
    ("6011", "ATM_WITHDRAWAL"),
    ("4900", "UTILITIES"),
    ("7832", "ENTERTAINMENT"),
    ("5912", "PHARMACY"),
    ("5651", "CLOTHING"),
    ("7011", "HOTEL"),
    ("5734", "SOFTWARE"),
    ("5411", "SUPERMARKET"),
]

# (city, country, lat, lon) - real coordinates so haversine distance /
# impossible-travel checks are meaningful.
CITIES = [
    ("New York", "US", 40.7128, -74.0060),
    ("Los Angeles", "US", 34.0522, -118.2437),
    ("Chicago", "US", 41.8781, -87.6298),
    ("Houston", "US", 29.7604, -95.3698),
    ("Miami", "US", 25.7617, -80.1918),
    ("San Francisco", "US", 37.7749, -122.4194),
    ("Seattle", "US", 47.6062, -122.3321),
    ("Toronto", "CA", 43.6532, -79.3832),
    ("Vancouver", "CA", 49.2827, -123.1207),
    ("London", "GB", 51.5074, -0.1278),
    ("Manchester", "GB", 53.4808, -2.2426),
    ("Paris", "FR", 48.8566, 2.3522),
    ("Berlin", "DE", 52.5200, 13.4050),
    ("Madrid", "ES", 40.4168, -3.7038),
    ("Mumbai", "IN", 19.0760, 72.8777),
    ("Delhi", "IN", 28.7041, 77.1025),
    ("Bengaluru", "IN", 12.9716, 77.5946),
    ("Singapore", "SG", 1.3521, 103.8198),
    ("Dubai", "AE", 25.2048, 55.2708),
    ("Sydney", "AU", -33.8688, 151.2093),
    ("Tokyo", "JP", 35.6762, 139.6503),
    ("Hong Kong", "HK", 22.3193, 114.1694),
    ("Sao Paulo", "BR", -23.5505, -46.6333),
    ("Johannesburg", "ZA", -26.2041, 28.0473),
    ("Lagos", "NG", 6.5244, 3.3792),
]

CHANNELS = ["CARD_PRESENT", "ECOM", "POS", "ATM", "WALLET"]

FRAUD_TYPOLOGIES = [
    "CARD_TESTING",
    "VELOCITY_ABUSE",
    "ACCOUNT_TAKEOVER",
    "UNUSUAL_AMOUNT",
    "GEOGRAPHIC_INCONSISTENCY",
    "FAILED_THEN_LARGE",
    "MERCHANT_BEHAVIOR_DEVIATION",
    "UNUSUAL_TIME_OF_DAY",
    "DEVICE_ANOMALY",
]
