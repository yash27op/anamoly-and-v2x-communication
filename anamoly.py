import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Streamlit Dashboard Configuration
st.set_page_config(page_title="IP Anomaly Detection and V2X Terahertz Simulation", layout="wide")

# Create Tabs
tab1, tab2 = st.tabs(["IP Anomaly Detection", "V2X Terahertz Communication Simulation"])

# Tab 1: IP Anomaly Detection
with tab1:
    st.title("IP Address Anomaly Detection Dashboard")

    # Input Section for IP Addresses and Request Frequencies
    st.sidebar.header("Input IP Address Data")
    num_ips = st.sidebar.number_input("Number of IP Addresses", min_value=1, max_value=100, value=5)

    ip_data = []
    for i in range(int(num_ips)):
        ip = st.sidebar.text_input(f"IP Address {i + 1}", f"192.168.1.{i + 1}")
        freq = st.sidebar.number_input(f"Requests per Minute for {ip}", min_value=0, value=100)
        ip_data.append((ip, freq))

    # Convert input data to DataFrame
    df = pd.DataFrame(ip_data, columns=["IP Address", "Requests per Minute"])

    # Train an Isolation Forest model
    model = IsolationForest(contamination=0.1)
    df['Anomaly'] = model.fit_predict(df[['Requests per Minute']])

    # Add a column to identify anomalies
    df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

    # Function to determine reason for anomaly
    def determine_anomaly_reason(requests_per_minute):
        if requests_per_minute > 200:
            return "High number of requests (Possible DDoS)"
        elif requests_per_minute < 50:
            return "Low number of requests (Possible bot activity)"
        else:
            return "Normal activity"

    # Function to classify IP based on requests per minute
    def classify_ip(requests_per_minute):
        if requests_per_minute > 200:
            return "Suspicious"
        elif requests_per_minute < 50:
            return "Potential Bot"
        else:
            return "Normal"

    # Apply the reasoning and classification
    df['Anomaly Reason'] = df['Requests per Minute'].apply(determine_anomaly_reason)
    df['IP Classification'] = df['Requests per Minute'].apply(classify_ip)

    # Display DataFrame
    st.write("### IP Address Activity Data")
    st.dataframe(df)

    # Visualize the data
    fig, ax = plt.subplots(figsize=(10, 6))
    normal_data = df[df['Anomaly'] == 'Normal']
    anomaly_data = df[df['Anomaly'] == 'Anomaly']

    ax.scatter(normal_data['IP Address'], normal_data['Requests per Minute'], label='Normal', color='blue')
    ax.scatter(anomaly_data['IP Address'], anomaly_data['Requests per Minute'], label='Anomaly', color='red')
    ax.set_xlabel('IP Address')
    ax.set_ylabel('Requests per Minute')
    ax.set_title('IP Address Activity with Anomalies')
    ax.legend()

    st.pyplot(fig)

    # Filter to show only anomalies
    st.write("### Detected Anomalies")
    st.dataframe(df[df['Anomaly'] == 'Anomaly'])

    # Placeholder for Terahertz Detectivity communication determination
    st.write("### Terahertz Detectivity Communication Determination")
    st.write("Advanced detectivity methods could be applied to further analyze anomalies and determine if specialized hardware or additional monitoring is needed to mitigate the risk.")

# Tab 2: V2X Terahertz Communication Simulation
with tab2:
    st.title("V2X Terahertz Communication Simulation")

    # Explanation of V2X Terahertz Communication
    st.write("""
    **V2X (Vehicle-to-Everything)** communication allows vehicles to communicate with each other, infrastructure, 
    pedestrians, and the cloud using various communication technologies. Terahertz communication is emerging as a 
    promising technology for high-speed, short-range communication in V2X, providing ultra-high bandwidth.
    """)

    st.write("### Simulation of Client-Server Terahertz Communication")

    # Parameters for the Simulation
    client_position = st.slider("Client (Vehicle) Position", min_value=0, max_value=1000, value=500)
    server_position = st.slider("Server (Base Station) Position", min_value=0, max_value=1000, value=800)
    frequency = st.slider("Frequency (THz)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # Calculate Distance
    distance = abs(server_position - client_position)
    st.write(f"**Distance between Client and Server:** {distance} meters")

    # Terahertz Communication Characteristics
    def calculate_thz_communication(distance, frequency):
        # Example calculation for signal loss (simplified model)
        # Critical range for terahertz communication is 0.1 to 10 THz
        if frequency < 0.1 or frequency > 10.0:
            return float('inf')  # Infinite path loss if out of range
        # Signal loss increases significantly with both distance and frequency
        path_loss = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20
        return path_loss

    path_loss = calculate_thz_communication(distance, frequency)
    
    # Display warning if frequency is out of the terahertz range
    if frequency < 0.1 or frequency > 10.0:
        st.warning("The selected frequency is out of the effective terahertz communication range (0.1 - 10 THz). Communication may not be reliable.")
    else:
        st.write(f"**Estimated Path Loss at {frequency} THz:** {path_loss:.2f} dB")

    # Plotting the Signal Strength over Distance
    fig, ax = plt.subplots(figsize=(10, 6))
    distances = np.linspace(0, 1000, 500)
    signal_losses = [calculate_thz_communication(d, frequency) for d in distances]
    ax.plot(distances, signal_losses, label=f'Frequency: {frequency} THz', color='purple')
    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Path Loss (dB)')
    ax.set_title('Signal Strength over Distance in Terahertz Communication')
    ax.legend()

    st.pyplot(fig)

    st.write("""
    This simulation shows how the path loss increases with distance in a terahertz communication link. 
    Communication effectiveness drops significantly if the frequency is outside the terahertz range.
    Maintaining communication within the terahertz frequency range (0.1 - 10 THz) is crucial for reliable V2X communication.
    """)
