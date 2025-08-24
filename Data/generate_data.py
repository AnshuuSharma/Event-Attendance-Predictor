# Event Attendance Synthetic Dataset Generator
import numpy as np
import pandas as pd

# Set random seed for reproducibility
rng = np.random.default_rng(42)

# Number of events to generate
N = 2500

# Define feature options
event_types = ["seminar","Stand-up","concert","sports","festival"]
cities = ["Delhi","Bengaluru","Indore","Mumbai","Pune","Hyderabad"]

def random_date():
    y = rng.integers(2024, 2026)
    m = rng.integers(1, 13)
    d = rng.integers(1, 28)  # keep simple
    return pd.Timestamp(year=int(y), month=int(m), day=int(d))

rows = []
for i in range(N):
    et = rng.choice(event_types, p=[0.25,0.2,0.2,0.15,0.2])
    city = rng.choice(cities)
    date = random_date()
    weekday = date.day_name()
    is_weekend = 1 if weekday in ["Saturday","Sunday"] else 0
    holiday = int(rng.random() < 0.07)  # ~7% holidays
    is_indoor = int(rng.random() < (0.8 if et in ["Stand-up","seminar"] else 0.4))

    capacity = int(rng.choice([150,200,300,500,800,1200,2000,3000], p=[.1,.1,.15,.2,.15,.15,.1,.05]))
    start_time = int(rng.integers(9, 22))
    duration = float(rng.choice([1.5,2,3,4], p=[.2,.4,.3,.1]))
    lead_time = int(rng.integers(3, 90))
    competing = int(rng.integers(0, 5))
    promo_days = int(rng.integers(0, 30))
    promo_channels = int(rng.integers(0, 5))
    promo_budget = float(rng.choice([0,2000,5000,10000,20000], p=[.2,.25,.25,.2,.1]))
    rep = int(rng.integers(1, 6))
    ticket_base = {"concert": [399, 1499], "sports": [199, 999],
                   "Stand-up":[99, 599], "seminar":[0, 299], "festival":[0, 399]}
    lo, hi = ticket_base[et]
    ticket_price = float(np.round(rng.uniform(lo, hi), 0))

    # weather (rough seasonal-ish)
    month = date.month
    temp = float(np.round(rng.normal(28 - abs(6 - month), 4), 1))
    rain_prob = 0.5 if month in [6,7,8,9] else 0.15
    is_rain = int(rng.random() < rain_prob)
    rain_mm = float(np.round(max(0, rng.normal(6 if is_rain else 0.5, 3)), 1))

    hist_avg = int(max(10, rng.normal(capacity*0.4, capacity*0.15)))

    #festival timing
    festival_effect = 0
    if et == "festival":
        if 17 <= start_time <= 21:
         festival_effect =  min(300, capacity - signal)  
    elif 11<= start_time < 17:
        festival_effect = -100

    
    def attendance_adjustment(temp, start_time, is_indoor):
     adjustment = 0
    
    # Outdoor events
     if is_indoor == 0:
        # High temp logic
        if temp > 35:
            if 12 <= start_time <= 16:   # afternoon
                adjustment -= 150
            else:  # other times, smaller effect
                adjustment -= 50
        
        # Low temp logic
        elif temp < 5:
            if 18 < start_time <= 22:   # night
                adjustment -= 120
            else:  # other times, smaller effect
                adjustment -= 50
        
        # Pleasant weather
        elif 20 <= temp <= 30:
            adjustment += 100
    
    # Indoor events
     else:
        if temp > 35 or temp < 5:
            adjustment += 50  # slight bonus for indoor comfort
    
     return adjustment


    # attendance signal (add effects)
    signal = (
        0.55*capacity
        + (150 if is_weekend else -50)
        + (120 if holiday and et in ["concert","festival","sports"] else (-40 if holiday and et in ["Stand-up","seminar"] else 0))
        + (0.08 * promo_budget)
        + (40 * promo_days**0.5)
        + (60 * promo_channels)
        + (120 * (rep-3))
        + (-0.6 * ticket_price if et in ["concert","sports"] else -0.3 * ticket_price)
        + (-180 if (is_rain and not is_indoor) else 0)
        + (-40 * competing)
        + (15 * lead_time**0.5)
        + (0.5 * hist_avg)
        + (max(0, capacity - signal + 100) if et == "concert" and city in ["Bengaluru","Delhi","Mumbai"] else 0)
        + (max(0, capacity - signal + 100) if event_types=="sports" and et=="Hyderabad" else 0)
        + (-100 if ((temp >= 35 or temp <= 10) and (12 <= start_time <= 21)) else 0)
        + attendance_adjustment(temp, start_time, is_indoor)
    )
 
    type_bias = {"concert": 200, "festival": 120, "sports": -60, "Stand-up": -80, "seminar": -100}[et]
    signal += type_bias
    signal += rng.normal(0, capacity*0.12)

    attendance = int(np.clip(signal, 0, capacity))

    rows.append([f"EVT{i:04d}", et, city, capacity, is_indoor,
                 date.date().isoformat(), weekday, is_weekend, holiday,
                 start_time, duration, ticket_price, promo_days, promo_channels,
                 rep, lead_time, competing, temp, is_rain, rain_mm, hist_avg,
                 promo_budget, attendance])

# Column names
cols = ["event_id","event_type","city","venue_capacity","is_indoor",
        "date","weekday","is_weekend","holiday","start_time_hour","duration_hours",
        "ticket_price","promotion_days","promo_channels_count","organizer_reputation",
        "lead_time_days","competing_events_same_day","weather_temp_c","weather_is_rain",
        "weather_rain_mm","historical_avg_attendance","promotion_budget","attendance"]

df = pd.DataFrame(rows, columns=cols)

# Save to CSV
df.to_csv("event_attendance_data.csv", index=False)

print("Dataset created successfully! Sample:")
print(df.head())
