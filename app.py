import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy import stats, interpolate
from scipy.interpolate import interp1d, make_interp_spline
from scipy.stats import ttest_ind
import seaborn as sns
from streamlit_option_menu import option_menu
from markup import app_intro, how_use_intro


PASSWORD = 'Ethan101'

def authenticate(password):
    return password == PASSWORD

def intro():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("image.jpg", use_column_width=True)
    with col2:
        st.markdown(app_intro(), unsafe_allow_html=True)
    st.markdown(how_use_intro(),unsafe_allow_html=True) 


    github_link = '[<img src="https://badgen.net/badge/icon/github?icon=github&label">](https://github.com/ethanrom)'
    huggingface_link = '[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">](https://huggingface.co/ethanrom)'

    st.write(github_link + '&nbsp;&nbsp;&nbsp;' + huggingface_link, unsafe_allow_html=True)

def tab1():
    st.subheader("Upload Excel Files")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload Patient Details Excel File:")
        patient_details_file = st.file_uploader("Choose a file", type=["xlsx"], key="patient_details")
    
    with col2:
        st.subheader("Upload Report Details Excel File:")
        report_details_file = st.file_uploader("Choose a file", type=["xlsx"], key="report_details")
    
    return patient_details_file, report_details_file

def tab2(patient_details_file, report_details_file):
    st.title("View and Filter Data")

    password_input = st.text_input('Enter Password', type='password')
    if authenticate(password_input):

        if patient_details_file:
            patient_details_df = pd.read_excel(patient_details_file)
            st.subheader("Patient Details Data:")
            st.dataframe(patient_details_df)
        
        if report_details_file:
            report_details_df = pd.read_excel(report_details_file)
            st.subheader("Report Details Data:")
            st.dataframe(report_details_df)

        if patient_details_file and report_details_file:
            with st.expander("Filter"):
                st.subheader("Filter Data:")
                selected_patient_columns = st.multiselect(
                    "Select patient columns for filtering:",
                    patient_details_df.columns
                )
                selected_report_columns = st.multiselect(
                    "Select report columns for filtering:",
                    report_details_df.columns
                )
                
                filtered_patient_details = patient_details_df[selected_patient_columns]
                filtered_report_details = report_details_df[selected_report_columns]
                
                st.subheader("Filtered Patient Details Data:")
                st.dataframe(filtered_patient_details)
                st.subheader("Filtered Report Details Data:")
                st.dataframe(filtered_report_details)


def tab3(patient_details_file, report_details_file):
    st.title("Select and View Patient Info")

    password_input = st.text_input('Enter Password', type='password')
    if authenticate(password_input):

        if patient_details_file and report_details_file:
            patient_details_df = pd.read_excel(patient_details_file)
            report_details_df = pd.read_excel(report_details_file)

            selected_patient = st.selectbox("Select a patient:", patient_details_df["LastName"].unique())

            selected_patient_info = report_details_df[report_details_df["Name"] == selected_patient]

            st.subheader("Selected Patient Info:")
            st.dataframe(selected_patient_info)

            tests = selected_patient_info["CODE"].unique()
            for test in tests:
                test_data = selected_patient_info[selected_patient_info["CODE"] == test]
                test_data = test_data.sort_values(by="Auftragsdatum")  # Sort by time in ascending order
                plt.figure(figsize=(8, 4))
                plt.plot(test_data["Auftragsdatum"], test_data["Befundtext"], marker='o')  # Use marker='o' for point markers
                plt.title(f"{test} Test Results for {selected_patient}")
                plt.xlabel("Date")
                plt.ylabel("Result")
                plt.xticks(rotation=45)
                st.pyplot(plt)

def tab4(patient_details_file, report_details_file):
    st.title("Point-to-Point Analysis")

    password_input = st.text_input('Enter Password', type='password')
    if authenticate(password_input):

        if patient_details_file and report_details_file:
            patient_details_df = pd.read_excel(patient_details_file)
            report_details_df = pd.read_excel(report_details_file)
            
            selected_patient = st.selectbox("Select a patient:", patient_details_df["LastName"].unique())
            
            selected_patient_info = report_details_df[report_details_df["Name"] == selected_patient]
            
            st.subheader("Selected Patient Info:")
            st.dataframe(selected_patient_info)
            
            selected_code = st.selectbox("Select a CODE:", selected_patient_info["CODE"].unique())
            
            selected_interval = st.selectbox("Select a time interval (hours):", [-24, 0, 24, 48, 72, 96, 120, 144, 168, 240, 360])
            last_recorded_time = selected_patient_info[selected_patient_info["CODE"] == selected_code]["Auftragsdatum"].max()
            start_time = last_recorded_time - timedelta(hours=abs(selected_interval))
            
            filtered_data = selected_patient_info[(selected_patient_info["CODE"] == selected_code) &
                                                (pd.to_datetime(selected_patient_info["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S') >= start_time)]
            
            st.subheader(f"Point-to-Point Analysis for {selected_code} in the last {abs(selected_interval)} hours:")
            st.dataframe(filtered_data)
            
            if not filtered_data.empty:
                values = filtered_data["Befundtext"].astype(float)
                mean_value = np.mean(values)
                std_dev = np.std(values)
                
                st.write(f"Mean Value: {mean_value}")
                st.write(f"Standard Deviation: {std_dev}")
                
                if abs(selected_interval) > 0:
                    interval1 = filtered_data[pd.to_datetime(filtered_data["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S') > (last_recorded_time - timedelta(hours=abs(selected_interval)))]
                    interval2 = filtered_data[pd.to_datetime(filtered_data["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S') <= last_recorded_time]
                    
                    if not interval1.empty and not interval2.empty:
                        interval1 = interval1.sort_values(by="Auftragsdatum")  # Sort interval1 by time
                        interval2 = interval2.sort_values(by="Auftragsdatum")  # Sort interval2 by time

                        _, p_value = stats.ttest_ind(interval1["Befundtext"].astype(float), interval2["Befundtext"].astype(float))
                        st.write(f"P-value for Statistical Significance Test: {p_value:.4f}")

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            plt.figure(figsize=(10, 6))
                            plt.plot(pd.to_datetime(interval1["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S'), interval1["Befundtext"].astype(float), marker='o', label=f"{abs(selected_interval)} hours ago")
                            plt.plot(pd.to_datetime(interval2["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S'), interval2["Befundtext"].astype(float), marker='x', label="Now")
                            plt.title(f"{selected_code} Point-to-Point Analysis for {selected_patient}")
                            plt.xlabel("Date")
                            plt.ylabel("Result")
                            plt.xticks(rotation=45)
                            plt.legend()
                            st.pyplot(plt)

                        st.subheader("Additional Analysis and Visualizations:")
                        
                        filtered_data = filtered_data.sort_values(by="Auftragsdatum")

                        plt.figure(figsize=(10, 6))
                        plt.hist(filtered_data["Befundtext"].astype(float), bins=20, edgecolor='k')
                        plt.title(f"{selected_code} Histogram")
                        plt.xlabel("Result")
                        plt.ylabel("Frequency")
                        st.pyplot(plt)
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(pd.to_datetime(filtered_data["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S'), filtered_data["Befundtext"].astype(float), marker='o', label=f"{selected_code}")
                        plt.title(f"{selected_code} Time Series Analysis for {selected_patient}")
                        plt.xlabel("Date")
                        plt.ylabel("Result")
                        plt.xticks(rotation=45)
                        plt.legend()
                        st.pyplot(plt)
                        
                        numeric_columns = filtered_data.select_dtypes(include=[np.number])
                        correlation_matrix = numeric_columns.corr()

                        plt.figure(figsize=(10, 6))
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                        plt.title("Correlation Heatmap")
                        st.pyplot(plt)
                        
                    with col2:    
                        summary_stats = filtered_data["Befundtext"].astype(float).describe()
                        st.write("Summary Statistics:")
                        st.write(summary_stats)

def tab5(patient_details_file, report_details_file):
    st.title("Interpolated Point-to-Point Analysis")

    password_input = st.text_input('Enter Password', type='password')
    if authenticate(password_input):

        if patient_details_file and report_details_file:
            patient_details_df = pd.read_excel(patient_details_file)
            report_details_df = pd.read_excel(report_details_file)
            
            selected_patient = st.selectbox("Select a patient:", patient_details_df["LastName"].unique())
            
            selected_patient_info = report_details_df[report_details_df["Name"] == selected_patient]
            
            st.subheader("Selected Patient Info:")
            st.dataframe(selected_patient_info)
            
            selected_code = st.selectbox("Select a CODE:", selected_patient_info["CODE"].unique())
            
            selected_interval = st.selectbox("Select a time interval (hours):", [-24, 0, 24, 48, 72, 96, 120, 144, 168, 240, 360])
            
            last_recorded_time = selected_patient_info[selected_patient_info["CODE"] == selected_code]["Auftragsdatum"].max()
            start_time = last_recorded_time - timedelta(hours=abs(selected_interval))
            
            filtered_data = selected_patient_info[(selected_patient_info["CODE"] == selected_code) &
                                                (pd.to_datetime(selected_patient_info["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S') >= start_time)]
            
            st.subheader(f"Point-to-Point Analysis for {selected_code} in the last {abs(selected_interval)} hours:")
            st.dataframe(filtered_data)
            
            if not filtered_data.empty:
                if abs(selected_interval) > 0:
                    sorted_data = filtered_data.sort_values(by='Auftragsdatum')
                    
                    timestamps = pd.to_datetime(sorted_data['Auftragsdatum'], format='%d.%m.%Y %H:%M:%S', errors='coerce').round('s')
                    values = sorted_data['Befundtext'].astype(float)
                    
                    timestamps_numeric = (timestamps - timestamps.min()).dt.total_seconds()

                    f = interpolate.interp1d(timestamps_numeric, values, kind='linear', fill_value='extrapolate')
                    interpolated_timestamps_numeric = np.arange(timestamps_numeric.min(), timestamps_numeric.max() + 1, 1)

                    interpolated_values = f(interpolated_timestamps_numeric)
                    interpolated_timestamps = timestamps.min() + pd.to_timedelta(interpolated_timestamps_numeric, unit='s')
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(timestamps, values, marker='o', label='Original Data')
                    plt.plot(interpolated_timestamps, interpolated_values, label='Interpolated Data')
                    plt.title(f"{selected_code} Interpolated Point-to-Point Analysis for {selected_patient}")
                    plt.xlabel("Date")
                    plt.ylabel("Result")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)

            if not filtered_data.empty:
                if abs(selected_interval) > 0:
                    sorted_data = filtered_data.sort_values(by='Auftragsdatum')
                    
                    timestamps = pd.to_datetime(sorted_data['Auftragsdatum'], format='%d.%m.%Y %H:%M:%S', errors='coerce').round('s')
                    values = sorted_data['Befundtext'].astype(float)
                    
                    timestamps_numeric = (timestamps - timestamps.min()).dt.total_seconds()

                    from scipy.interpolate import make_interp_spline

                    spline = make_interp_spline(timestamps_numeric, values, bc_type='natural')
                    interpolated_timestamps_numeric = np.arange(timestamps_numeric.min(), timestamps_numeric.max() + 1, 1)

                    interpolated_values_spline = spline(interpolated_timestamps_numeric)
                    interpolated_timestamps = timestamps.min() + pd.to_timedelta(interpolated_timestamps_numeric, unit='s')
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(timestamps, values, marker='o', label='Original Data')
                    plt.plot(interpolated_timestamps, interpolated_values_spline, label='Spline Interpolation')
                    plt.title(f"{selected_code} Spline Interpolated Point-to-Point Analysis for {selected_patient}")
                    plt.xlabel("Date")
                    plt.ylabel("Result")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)

            if not filtered_data.empty:
                if abs(selected_interval) > 0:
                    sorted_data = filtered_data.sort_values(by='Auftragsdatum')

                    timestamps = pd.to_datetime(sorted_data['Auftragsdatum'], format='%d.%m.%Y %H:%M:%S', errors='coerce').round('s')
                    values = sorted_data['Befundtext'].astype(float)

                    timestamps_numeric = (timestamps - timestamps.min()).dt.total_seconds()

                    # Perform linear interpolation
                    f = interp1d(timestamps_numeric, values, kind='linear', fill_value='extrapolate')
                    interpolated_timestamps_numeric = np.arange(timestamps_numeric.min(), timestamps_numeric.max() + 1, 1)
                    interpolated_values = f(interpolated_timestamps_numeric)

                    # Perform t-test to assess statistical significance
                    t_statistic, p_value = ttest_ind(values, interpolated_values)

                    # Display the results of the t-test and explanations
                    st.subheader("Statistical Significance Analysis:")

                    # Display the original data mean and standard deviation
                    original_mean = np.mean(values)
                    original_std = np.std(values)
                    st.write(f"Original Data Mean: {original_mean:.2f}")
                    st.write(f"Original Data Standard Deviation: {original_std:.2f}")

                    # Display the interpolated data mean and standard deviation
                    interpolated_mean = np.mean(interpolated_values)
                    interpolated_std = np.std(interpolated_values)
                    st.write(f"Interpolated Data Mean: {interpolated_mean:.2f}")
                    st.write(f"Interpolated Data Standard Deviation: {interpolated_std:.2f}")

                    # Display the t-statistic and p-value
                    st.write(f"t-statistic: {t_statistic:.2f}")
                    st.write(f"p-value: {p_value:.4f}")

                    # Explain the results
                    st.write("The t-statistic measures the difference between the means of the original data and interpolated data.")
                    st.write("A higher t-statistic indicates a larger difference between the means.")
                    st.write("The p-value represents the probability of observing such a difference by chance alone.")
                    st.write("A small p-value (typically less than 0.05) suggests that the difference is statistically significant,")
                    st.write("meaning that the interpolation results are significantly different from the actual recorded values.")

                    # Plot original data, linear interpolation, and spline interpolation
                    plt.figure(figsize=(10, 6))
                    plt.plot(timestamps, values, marker='o', label='Original Data')
                    plt.plot(interpolated_timestamps, interpolated_values, label='Linear Interpolation')
                    plt.title(f"{selected_code} Interpolated Point-to-Point Analysis for {selected_patient}")
                    plt.xlabel("Date")
                    plt.ylabel("Result")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)

        
def tab6(patient_details_file, report_details_file):
    st.title("Generate Patient Results File")

    password_input = st.text_input('Enter Password', type='password')
    if authenticate(password_input):

        if patient_details_file and report_details_file:
            patient_details_df = pd.read_excel(patient_details_file)
            report_details_df = pd.read_excel(report_details_file)
            
            selected_patient = st.selectbox("Select a patient:", patient_details_df["LastName"].unique())
            
            selected_patient_info = report_details_df[report_details_df["Name"] == selected_patient]
            
            st.subheader("Selected Patient Info:")
            st.dataframe(selected_patient_info)
            
            unique_codes = selected_patient_info["CODE"].unique()
            time_points = [-24, 0, 24, 48, 72] + [i for i in range(96, 241, 24)]
            result_df = pd.DataFrame({"Timestamp": selected_patient_info["Auftragsdatum"]})
            
            for code in unique_codes:
                for time_point in time_points:
                    start_time = selected_patient_info[selected_patient_info["CODE"] == code]["Auftragsdatum"].max() - timedelta(hours=abs(time_point))
                    filtered_data = selected_patient_info[(selected_patient_info["CODE"] == code) &
                                                        (pd.to_datetime(selected_patient_info["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S') >= start_time)]
                    
                    timestamps = pd.to_datetime(filtered_data['Auftragsdatum'], format='%d.%m.%Y %H:%M:%S', errors='coerce').round('s')
                    values = filtered_data['Befundtext']
                    values = pd.to_numeric(values, errors='coerce')
                    
                    timestamps_numeric = (timestamps - timestamps.min()).dt.total_seconds()

                    f = interpolate.interp1d(timestamps_numeric, values, kind='linear', fill_value='extrapolate')
                    interpolated_timestamps_numeric = np.arange(timestamps_numeric.min(), timestamps_numeric.max() + 1, 1)

                    interpolated_values = f(interpolated_timestamps_numeric)
                    interpolated_timestamps = timestamps.min() + pd.to_timedelta(interpolated_timestamps_numeric, unit='s')
                    
                    timestamp_for_time_point = start_time + timedelta(hours=time_point)
                    timestamp_for_time_point_unix = timestamp_for_time_point.timestamp()
                    
                    absolute_differences = np.abs(interpolated_timestamps_numeric - timestamp_for_time_point_unix)
                    closest_index = np.argmin(absolute_differences)
                    
                    interpolated_value_for_time_point = interpolated_values[closest_index]
                    column_name = f"{code}_{time_point}h"
                    result_df[column_name] = interpolated_value_for_time_point
                
            st.subheader("Generated Patient Results File:")
            st.dataframe(result_df)
            

            result_file_name = f"{selected_patient}_results.xlsx"
            result_df.to_excel(result_file_name, index=False)
            
            st.markdown(f"Download the results file: [**{result_file_name}**](/{result_file_name})",
                        unsafe_allow_html=True)
            
            st.success(f"Results file '{result_file_name}' has been generated and saved.")

def main():
    st.set_page_config(page_title="Patient Data App")
    st.title("Lab Parameter Analysis Tool")
    
    tabs = ["Introduction", "Upload Files", "View and Filter Data", "Select and View Patient Info", "Point-to-Point Analysis", "Interpolated Point-to-Point Analysis", "Generate Patient Results File"]
    with st.sidebar:

        selected_tab = option_menu("Select a Tab", tabs, menu_icon="cast")

    if selected_tab == "Introduction":
        intro()    
    elif selected_tab == "Upload Files":
        tab1()
    elif selected_tab == "View and Filter Data":
        patient_details_file, report_details_file = tab1()
        tab2(patient_details_file, report_details_file)
    elif selected_tab == "Select and View Patient Info":
        patient_details_file, report_details_file = tab1()
        tab3(patient_details_file, report_details_file)
    elif selected_tab == "Point-to-Point Analysis":
        patient_details_file, report_details_file = tab1()
        tab4(patient_details_file, report_details_file)
    elif selected_tab == "Interpolated Point-to-Point Analysis":
        patient_details_file, report_details_file = tab1()
        tab5(patient_details_file, report_details_file)
    elif selected_tab == "Generate Patient Results File":
        patient_details_file, report_details_file = tab1()
        tab6(patient_details_file, report_details_file)

if __name__ == "__main__":
    main()
