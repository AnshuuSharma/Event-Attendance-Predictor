# # src/app.py
# import os
# import numpy as np
# import pandas as pd
# import streamlit as st
# import joblib
# import datetime as dt

# # ---------- Load trained pipeline ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = "attendance_predictor.pkl"
# pipeline = joblib.load(MODEL_PATH)

# st.set_page_config(page_title="Event Attendance Predictor", layout="centered")
# st.title("🎤 Event Attendance Predictor")

# st.markdown("Enter event details. The app will compute `venue_capacity_log` and pass all features to the model.")

# # ---------- Helper functions ----------
# def get_time_of_day(hour):
#     """Convert hour (0-23) to time of day"""
#     if 5 <= hour < 12:
#         return "Morning"
#     elif 12 <= hour < 17:
#         return "Afternoon"
#     elif 17 <= hour < 21:
#         return "Evening"
#     else:
#         return "Night"

# def is_weekend(date):
#     """Return 1 if weekend, 0 otherwise"""
#     return 1 if date.weekday() in (5, 6) else 0

# # ---------- Inputs ----------
# # Categorical (use exact labels seen in training/coefficients)
# event_type = st.selectbox("Event Type", ["concert", "sports", "seminar", "Stand-up", "festival"])
# city = st.selectbox("City", ["Bengaluru", "Delhi", "Hyderabad", "Indore", "Mumbai", "Pune"])

# # Venue
# venue_capacity = st.number_input("Venue Capacity", min_value=1, max_value=500000, value=5000, step=100)

# # Time
# start_time_hour = st.number_input("Start Time (Hour 0–23)", min_value=0, max_value=23, value=19, step=1)
# time_of_day = get_time_of_day(start_time_hour)

# # Pricing & Promotion
# ticket_price = st.number_input("Ticket Price", min_value=0.0, value=1500.0, step=50.0)
# promotion_days = st.number_input("Promotion Days", min_value=0, value=10, step=1)

# # Weather (to be replaced with API values later)
# weather_temp_c = st.number_input("Weather Temperature (°C)", min_value=-20.0, max_value=55.0, value=28.0, step=0.5)
# weather_is_rain = st.selectbox("Is it raining?", ["No", "Yes"])
# weather_is_rain = 1 if weather_is_rain == "Yes" else 0

# # Event context
# is_indoor = st.selectbox("Is Indoor?", ["Yes", "No"])
# is_indoor = 1 if is_indoor == "Yes" else 0

# # Event date -> auto-set weekend
# date = st.date_input("Event Date", dt.date.today())
# is_weekend_val = is_weekend(date)

# holiday = st.selectbox("Is Holiday?", ["No", "Yes"])
# holiday = 1 if holiday == "Yes" else 0

# st.divider()

# # ---------- Predict ----------
# if st.button("Predict Attendance 🎯"):
#     venue_capacity_log = float(np.log(venue_capacity))

#     features = [
#         'event_type', 'city', 'venue_capacity_log', 'is_indoor',
#         'is_weekend', 'holiday', 'start_time_hour', 'ticket_price',
#         'promotion_days', 'weather_temp_c', 'weather_is_rain', 'time_of_day'
#     ]

#     row = {
#         'event_type': event_type,
#         'city': city,
#         'venue_capacity_log': venue_capacity_log,
#         'is_indoor': is_indoor,
#         'is_weekend': is_weekend_val,
#         'holiday': holiday,
#         'start_time_hour': start_time_hour,
#         'ticket_price': ticket_price,
#         'promotion_days': promotion_days,
#         'weather_temp_c': weather_temp_c,
#         'weather_is_rain': weather_is_rain,
#         'time_of_day': time_of_day
#     }

#     input_df = pd.DataFrame([row], columns=features)
#     print(input_df.head())

#     try:
#         pred = pipeline.predict(input_df)[0]
#         st.success(f"✅ Predicted Attendance: **{int(round(pred))}**")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
#         st.caption("Tip: Make sure the model file and feature names match the training pipeline.")


# src/app.py

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# import os
# import numpy as np
# import pandas as pd
# import streamlit as st
# import joblib
# import datetime as dt

# # ---------- Load trained pipeline ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "./attendance_predictor.pkl")
# pipeline = joblib.load(MODEL_PATH)

# st.set_page_config(page_title="Event Attendance Predictor", layout="centered")
# st.title("🎤 Event Attendance Predictor")

# st.markdown("Enter event details. The app will pass all features to the Random Forest model.")

# # ---------- Helper functions ----------
# def get_time_of_day(hour):
#     """Convert hour (0-23) to time of day"""
#     if 5 <= hour < 12:
#         return "Morning"
#     elif 12 <= hour < 17:
#         return "Afternoon"
#     elif 17 <= hour < 21:
#         return "Evening"
#     else:
#         return "Night"

# def is_weekend(date):
#     """Return 1 if weekend, 0 otherwise"""
#     return 1 if date.weekday() in (5, 6) else 0

# # ---------- Inputs ----------
# event_type = st.selectbox("Event Type", ["concert", "sports", "seminar", "Stand-up", "festival"])
# city = st.selectbox("City", ["Bengaluru", "Delhi", "Hyderabad", "Indore", "Mumbai", "Pune"])

# venue_capacity = st.number_input("Venue Capacity", min_value=1, max_value=500000, value=5000, step=100)

# start_time_hour = st.number_input("Start Time (Hour 0–23)", min_value=0, max_value=23, value=19, step=1)
# time_of_day = get_time_of_day(start_time_hour)

# ticket_price = st.number_input("Ticket Price", min_value=0.0, value=1500.0, step=50.0)
# promotion_days = st.number_input("Promotion Days", min_value=0, value=10, step=1)

# weather_temp_c = st.number_input("Weather Temperature (°C)", min_value=-20.0, max_value=55.0, value=28.0, step=0.5)
# weather_is_rain = st.selectbox("Is it raining?", ["No", "Yes"])
# weather_is_rain = 1 if weather_is_rain == "Yes" else 0

# is_indoor = st.selectbox("Is Indoor?", ["Yes", "No"])
# is_indoor = 1 if is_indoor == "Yes" else 0

# date = st.date_input("Event Date", dt.date.today())
# is_weekend_val = is_weekend(date)

# holiday = st.selectbox("Is Holiday?", ["No", "Yes"])
# holiday = 1 if holiday == "Yes" else 0

# st.divider()

# # ---------- Predict ----------
# if st.button("Predict Attendance 🎯"):
#     features = [
#         'event_type', 'city', 'venue_capacity_log', 'is_indoor',
#         'is_weekend', 'holiday', 'start_time_hour', 'ticket_price',
#         'promotion_days', 'weather_temp_c', 'weather_is_rain', 'time_of_day'
#     ]

#     row = {
#         'event_type': event_type,
#         'city': city,
#         'venue_capacity_log': venue_capacity,
#         'is_indoor': is_indoor,
#         'is_weekend': is_weekend_val,
#         'holiday': holiday,
#         'start_time_hour': start_time_hour,
#         'ticket_price': ticket_price,
#         'promotion_days': promotion_days,
#         'weather_temp_c': weather_temp_c,
#         'weather_is_rain': weather_is_rain,
#         'time_of_day': time_of_day
#     }

#     input_df = pd.DataFrame([row], columns=features)
#     print(input_df.head())

#     try:
#         pred = pipeline.predict(input_df)[0]
#         # Cap prediction at venue capacity
#         pred = min(pred, venue_capacity)
#         st.success(f"✅ Predicted Attendance: **{int(round(pred))}**")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
#         st.caption("Tip: Make sure the model file and feature names match the training pipeline.")

# ============================================================================================================


# src/app.py
# src/app.py
# import os
# import numpy as np
# import pandas as pd
# import streamlit as st
# import joblib
# import datetime as dt
# import matplotlib.pyplot as plt

# # ---------- Load trained pipeline ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "./attendance_predictor.pkl")
# pipeline = joblib.load(MODEL_PATH)

# st.set_page_config(page_title="Event Attendance Predictor", layout="centered")
# st.title("🎤 Event Attendance Predictor")
# st.markdown("Enter event details. The app will pass all features to the model and predict attendance.")

# # ---------- Helper functions ----------
# def get_time_of_day(hour):
#     """Convert hour (0-23) to time of day"""
#     if 5 <= hour < 12:
#         return "Morning"
#     elif 12 <= hour < 17:
#         return "Afternoon"
#     elif 17 <= hour < 21:
#         return "Evening"
#     else:
#         return "Night"

# def is_weekend(date):
#     """Return 1 if weekend, 0 otherwise"""
#     return 1 if date.weekday() in (5, 6) else 0

# # ---------- Inputs ----------
# event_type = st.selectbox("Event Type", ["concert", "sports", "seminar", "Stand-up", "festival"])
# city = st.selectbox("City", ["Bengaluru", "Delhi", "Hyderabad", "Indore", "Mumbai", "Pune"])
# venue_capacity = st.number_input("Venue Capacity", min_value=1, max_value=500000, value=5000, step=100)
# start_time_hour = st.number_input("Start Time (Hour 0–23)", min_value=0, max_value=23, value=19, step=1)
# time_of_day = get_time_of_day(start_time_hour)
# ticket_price = st.number_input("Ticket Price", min_value=0.0, value=1500.0, step=50.0)
# promotion_days = st.number_input("Promotion Days", min_value=0, value=10, step=1)
# weather_temp_c = st.number_input("Weather Temperature (°C)", min_value=-20.0, max_value=55.0, value=28.0, step=0.5)
# weather_is_rain = st.selectbox("Is it raining?", ["No", "Yes"])
# weather_is_rain_val = 1 if weather_is_rain == "Yes" else 0
# is_indoor = st.selectbox("Is Indoor?", ["Yes", "No"])
# is_indoor_val = 1 if is_indoor == "Yes" else 0
# date = st.date_input("Event Date", dt.date.today())
# is_weekend_val = is_weekend(date)
# holiday = st.selectbox("Is Holiday?", ["No", "Yes"])
# holiday_val = 1 if holiday == "Yes" else 0

# # ---------- Prepare base input row ----------
# row = {
#     'event_type': event_type,
#     'city': city,
#     'venue_capacity': venue_capacity,  # Raw capacity for Random Forest
#     'is_indoor': is_indoor_val,
#     'is_weekend': is_weekend_val,
#     'holiday': holiday_val,
#     'start_time_hour': start_time_hour,
#     'ticket_price': ticket_price,
#     'promotion_days': promotion_days,
#     'weather_temp_c': weather_temp_c,
#     'weather_is_rain': weather_is_rain_val,
#     'time_of_day': time_of_day
# }

# st.divider()

# # ---------- Main Prediction ----------
# if st.button("Predict Attendance 🎯"):
#     input_df = pd.DataFrame([row])
#     try:
#         pred = pipeline.predict(input_df)[0]
#         # Cap prediction at venue capacity
#         pred = min(pred, venue_capacity)
#         st.success(f"✅ Predicted Attendance: **{int(round(pred))}**")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")
#         st.caption("Tip: Make sure the model file and feature names match the training pipeline.")

# # ---------- Interactive What-If Analysis ----------
# st.subheader("🌟 What-If Analysis")
# st.markdown("Adjust ticket price, promotion days, or rain status to see how attendance changes.")

# col1, col2 = st.columns(2)

# # --- Ticket price ---
# ticket_prices = st.slider("Ticket Price Range", min_value=0, max_value=5000, value=(0, 3000), step=50)
# price_values = np.arange(ticket_prices[0], ticket_prices[1]+1, 50)
# preds_price = [pipeline.predict(pd.DataFrame([{**row, 'ticket_price': p}]))[0] for p in price_values]

# with col1:
#     fig, ax = plt.subplots()
#     ax.plot(price_values, preds_price, marker='o')
#     ax.set_xlabel("Ticket Price")
#     ax.set_ylabel("Predicted Attendance")
#     ax.set_title("Attendance vs Ticket Price")
#     st.pyplot(fig)

# # --- Promotion days ---
# promotion_days_range = st.slider("Promotion Days Range", min_value=0, max_value=30, value=(0, 20))
# promo_values = np.arange(promotion_days_range[0], promotion_days_range[1]+1, 1)
# preds_promo = [pipeline.predict(pd.DataFrame([{**row, 'promotion_days': d}]))[0] for d in promo_values]

# with col2:
#     fig2, ax2 = plt.subplots()
#     ax2.plot(promo_values, preds_promo, marker='o', color='orange')
#     ax2.set_xlabel("Promotion Days")
#     ax2.set_ylabel("Predicted Attendance")
#     ax2.set_title("Attendance vs Promotion Days")
#     st.pyplot(fig2)
# # --- Rain effect ---
# # st.markdown("Toggle rain to see effect:")
# # rain_options = ["No", "Yes"]
# # rain_preds = [pipeline.predict(pd.DataFrame([{**row, 'weather_is_rain': 1 if r=="Yes" else 0}]))[0] for r in rain_options]

# # with col3:
# #     fig3, ax3 = plt.subplots()
# #     ax3.bar(rain_options, rain_preds, color=['green', 'red'])
# #     ax3.set_ylabel("Predicted Attendance")
# #     ax3.set_title("Effect of Rain on Attendance")
# #     st.pyplot(fig3)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime as dt
import matplotlib.pyplot as plt
import random

# ---------- Load trained pipeline ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "./attendance_predictor.pkl")
pipeline = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Event Attendance Predictor", layout="centered")
st.title("🎤 Event Attendance Predictor")
st.markdown("Enter event details. The app will pass all features to the model and predict attendance.")

# ---------- Helper functions ----------
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def is_weekend(date):
    return 1 if date.weekday() in (5, 6) else 0

def generate_weather(city, month):
    """Generate synthetic weather (temp °C, rain=0/1) based on city + month"""
    city = city.lower()
    month = month.lower()

    temp = random.randint(20, 30)
    rain = 0

    # North India (Delhi, Indore)
    if city in ["delhi", "indore"]:
        if month in ["december", "january"]:
            temp = random.randint(5, 15)
        elif month in ["july", "august"]:
            temp = random.randint(25, 32)
            rain = 1
    # West Coast (Mumbai, Pune)
    elif city in ["mumbai", "pune"]:
        if month in ["december", "january"]:
            temp = random.randint(18, 25)
        elif month in ["july", "august"]:
            temp = random.randint(28, 33)
            rain = 1
    # South (Bengaluru, Hyderabad)
    elif city in ["bengaluru", "hyderabad"]:
        if month in ["december", "january"]:
            temp = random.randint(15, 22)
        elif month in ["july", "august"]:
            temp = random.randint(22, 28)
            rain = 1

    return temp, rain

# ---------- Inputs ----------
col1, col2 = st.columns(2)

with col1:
    event_type = st.selectbox("Event Type", ["concert", "sports", "seminar", "Stand-up", "festival"])
    city = st.selectbox("City", ["Bengaluru", "Delhi", "Hyderabad", "Indore", "Mumbai", "Pune"])
    venue_capacity = st.number_input("Venue Capacity", min_value=1, max_value=500000, value=3000, step=100)
    ticket_price = st.number_input("Ticket Price", min_value=0.0, value=1500.0, step=50.0)

with col2:
    start_time_hour = st.number_input("Start Time (Hour 0–23)", min_value=0, max_value=23, value=19, step=1)
    promotion_days = st.number_input("Promotion Days", min_value=0, value=10, step=1)
    is_indoor = st.selectbox("Is Indoor?", ["Yes", "No"])
    is_indoor_val = 1 if is_indoor == "Yes" else 0
    date = st.date_input("Event Date", dt.date.today())

# Derived features
time_of_day = get_time_of_day(start_time_hour)
is_weekend_val = is_weekend(date)
holiday = st.selectbox("Is Holiday?", ["No", "Yes"])
holiday_val = 1 if holiday == "Yes" else 0

# Auto-generate weather from date + city
month_name = date.strftime("%B")
weather_temp_c, weather_is_rain_val = generate_weather(city, month_name)

st.info(f"🌡️ Estimated Weather: {weather_temp_c}°C | 🌧️ Rain: {'Yes' if weather_is_rain_val else 'No'}")

# ---------- Prepare base input row ----------
row = {
    'event_type': event_type,
    'city': city,
    'venue_capacity': venue_capacity,
    'is_indoor': is_indoor_val,
    'is_weekend': is_weekend_val,
    'holiday': holiday_val,
    'start_time_hour': start_time_hour,
    'ticket_price': ticket_price,
    'promotion_days': promotion_days,
    'weather_temp_c': weather_temp_c,
    'weather_is_rain': weather_is_rain_val,
    'time_of_day': time_of_day
}

st.divider()

# ---------- Main Prediction ----------
if st.button("Predict Attendance 🎯"):
    input_df = pd.DataFrame([row])
    try:
        pred = pipeline.predict(input_df)[0]
        pred = min(pred, venue_capacity)  # Cap at capacity
        st.success(f"✅ Predicted Attendance: **{int(round(pred))}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption("Tip: Make sure the model file and feature names match the training pipeline.")

# ---------- Interactive What-If Analysis ----------
st.subheader("🌟 What-If Analysis")
st.markdown("Adjust ticket price, promotion days, or rain status to see how attendance changes.")

col1, col2 = st.columns(2)

# --- Ticket price ---
ticket_prices = st.slider("Ticket Price Range", min_value=0, max_value=5000, value=(0, 3000), step=50)
price_values = np.arange(ticket_prices[0], ticket_prices[1]+1, 50)
preds_price = [pipeline.predict(pd.DataFrame([{**row, 'ticket_price': p}]))[0] for p in price_values]

with col1:
    fig, ax = plt.subplots()
    ax.plot(price_values, preds_price, marker='o')
    ax.set_xlabel("Ticket Price")
    ax.set_ylabel("Predicted Attendance")
    ax.set_title("Attendance vs Ticket Price")
    st.pyplot(fig)

# --- Promotion days ---
promotion_days_range = st.slider("Promotion Days Range", min_value=0, max_value=30, value=(0, 20))
promo_values = np.arange(promotion_days_range[0], promotion_days_range[1]+1, 1)
preds_promo = [pipeline.predict(pd.DataFrame([{**row, 'promotion_days': d}]))[0] for d in promo_values]

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(promo_values, preds_promo, marker='o', color='orange')
    ax2.set_xlabel("Promotion Days")
    ax2.set_ylabel("Predicted Attendance")
    ax2.set_title("Attendance vs Promotion Days")
    st.pyplot(fig2)
