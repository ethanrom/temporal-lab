import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy import stats, interpolate
from scipy.interpolate import interp1d, CubicSpline
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

            info_option = st.radio("Select Info Display Option:", ["Individual Patient Info", "Grouped Patient Info"])

            if info_option == "Individual Patient Info":
                selected_patient = st.selectbox("Select a patient:", patient_details_df["LastName"].unique(), key="patient")
                selected_patient_info = report_details_df[report_details_df["Name"] == selected_patient]

                st.subheader("Selected Patient Info:")
                st.dataframe(selected_patient_info)

                reference_time = patient_details_df[patient_details_df["LastName"] == selected_patient]["PreopDiagTimingSurgSkinIncision"].max()
                st.subheader(f"Reference Time : {reference_time}")

            elif info_option == "Grouped Patient Info":
                selected_timing = st.radio("Select Disease Dissection Timing:", ["chronic", "acute"])
                selected_patients = patient_details_df[patient_details_df["DiseaseDissectionTiming"] == selected_timing]

                if not selected_patients.empty:
                    st.subheader("Selected Patients:")
                    st.dataframe(selected_patients)

                    reference_times = []
                    for selected_patient in selected_patients["LastName"].unique():
                        selected_patient_info = report_details_df[report_details_df["Name"] == selected_patient]
                        
                        reference_time = patient_details_df[patient_details_df["LastName"] == selected_patient]["PreopDiagTimingSurgSkinIncision"].max()
                        reference_times.append(reference_time)
                    
                    reference_time = min(reference_times) + (max(reference_times) - min(reference_times)) / 2
                    st.subheader(f"Reference Time : {reference_time}")

                else:
                    st.write("No patients found for the selected timing.")
                    return

            selected_code = st.selectbox("Select a CODE:", selected_patient_info["CODE"].unique())
            selected_interval = st.selectbox("Select a time interval (hours):", [-24, 0, 24, 48, 72, 96, 120, 144, 168, 240, 360])

            end_time = reference_time + timedelta(hours=abs(selected_interval))            
            start_time = reference_time
            filtered_data = selected_patient_info[(selected_patient_info["CODE"] == selected_code) &
                                                 (pd.to_datetime(selected_patient_info["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S') >= start_time) &
                                                 (pd.to_datetime(selected_patient_info["Auftragsdatum"], format='%d.%m.%Y %H:%M:%S') <= end_time)]

            st.subheader(f"Point-to-Point Analysis for {selected_code} in the last {abs(selected_interval)} hours:")
            st.dataframe(filtered_data)
            

            
            if not filtered_data.empty:
                if abs(selected_interval) > 0:
                    sorted_data = filtered_data.sort_values(by='Auftragsdatum')

                    timestamps = pd.to_datetime(sorted_data['Auftragsdatum'], format='%d.%m.%Y %H:%M:%S', errors='coerce').round('s')
                    values = sorted_data['Befundtext'].astype(float)

                    if len(timestamps) < 2:
                        st.warning("Insufficient data for interpolation. Please select a different time interval or patient.")
                    else:
                        timestamps_numeric = (timestamps - timestamps.min()).dt.total_seconds()

                        cubic_spline = CubicSpline(timestamps_numeric, values)

                        interpolated_timestamps_numeric = np.arange(timestamps_numeric.min(), timestamps_numeric.max() + 1, 1)
                        interpolated_values_spline = cubic_spline(interpolated_timestamps_numeric)

                        interpolated_timestamps = timestamps.min() + pd.to_timedelta(interpolated_timestamps_numeric, unit='s')

                        plt.figure(figsize=(10, 6))
                        plt.plot(timestamps, values, marker='o', label='Original Data')
                        plt.plot(interpolated_timestamps, interpolated_values_spline, label='Cubic Spline Interpolation')
                        plt.title(f"{selected_code} Cubic Spline Interpolated Point-to-Point Analysis for {selected_patient}")
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
                    f = interp1d(timestamps_numeric, values, kind='linear', fill_value='extrapolate')
                    interpolated_timestamps_numeric = np.arange(timestamps_numeric.min(), timestamps_numeric.max() + 1, 1)
                    interpolated_values = f(interpolated_timestamps_numeric)
                    t_statistic, p_value = ttest_ind(values, interpolated_values)

                    st.subheader("Statistical Significance Analysis:")

                    original_mean = np.mean(values)
                    original_std = np.std(values)
                    st.write(f"Original Data Mean: {original_mean:.2f}")
                    st.write(f"Original Data Standard Deviation: {original_std:.2f}")

                    interpolated_mean = np.mean(interpolated_values)
                    interpolated_std = np.std(interpolated_values)
                    
                    st.write(f"Interpolated Data Mean: {interpolated_mean:.2f}")
                    st.write(f"Interpolated Data Standard Deviation: {interpolated_std:.2f}")
                    st.write(f"t-statistic: {t_statistic:.2f}")
                    st.write(f"p-value: {p_value:.4f}")
                    st.write("The t-statistic measures the difference between the means of the original data and interpolated data.")
                    st.write("A higher t-statistic indicates a larger difference between the means.")
                    st.write("The p-value represents the probability of observing such a difference by chance alone.")
                    st.write("A small p-value (typically less than 0.05) suggests that the difference is statistically significant,")
                    st.write("meaning that the interpolation results are significantly different from the actual recorded values.")


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



def calculate_midpoint(start_time, end_time):
    half_duration = (end_time - start_time) / 2
    midpoint = start_time + half_duration
    return midpoint

def format_timedelta(delta):
    days, seconds = delta.days, delta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"

def tab7(patient_details_file, report_details_file):
    st.title("Calculate Distance of Lab Tests to Midpoint of Operation")

    password_input = st.text_input('Enter Password', type='password')
    if authenticate(password_input):

        if patient_details_file and report_details_file:
            patient_details_df = pd.read_excel(patient_details_file)
            report_details_df = pd.read_excel(report_details_file)

            selected_patient = st.selectbox("Select a patient:", patient_details_df["LastName"].unique())
            selected_patient_info = report_details_df[report_details_df["Name"] == selected_patient].copy()
            st.subheader("Selected Patient Info:")
            st.dataframe(selected_patient_info)

            operation_info = patient_details_df[patient_details_df["LastName"] == selected_patient]
            start_time_str = operation_info["PreopDiagTimingSurgSkinIncision"].values[0]
            end_time_str = operation_info["PreopDiagTimingSurgSutureEnd"].values[0]

            if isinstance(start_time_str, np.datetime64):
                start_time_str = start_time_str.astype(str)
            if isinstance(end_time_str, np.datetime64):
                end_time_str = end_time_str.astype(str)
            start_time_str = start_time_str.replace("T", " ").split(".")[0]
            end_time_str = end_time_str.replace("T", " ").split(".")[0]

            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            midpoint = calculate_midpoint(start_time, end_time)

            selected_patient_info["Auftragsdatum"] = pd.to_datetime(selected_patient_info["Auftragsdatum"], format="%d.%m.%Y %H:%M:%S")
            selected_patient_info["Distance_to_Midpoint"] = abs(selected_patient_info["Auftragsdatum"] - midpoint)

            selected_patient_info["Distance_to_Midpoint"] = selected_patient_info["Distance_to_Midpoint"].apply(format_timedelta)
            st.subheader("Lab Test Distance to Midpoint of Operation:")
            st.dataframe(selected_patient_info[["Auftragsdatum", "CODE", "Befundtext", "Distance_to_Midpoint"]])

            if not pd.api.types.is_timedelta64_dtype(selected_patient_info["Distance_to_Midpoint"]):
                selected_patient_info["Distance_to_Midpoint"] = pd.to_timedelta(selected_patient_info["Distance_to_Midpoint"])

            unique_codes = selected_patient_info["CODE"].unique()

            # Create a scatter plot for each unique CODE
            for code in unique_codes:
                code_data = selected_patient_info[selected_patient_info["CODE"] == code]

                # Ensure that "Befundtext" is numeric and sort the DataFrame by it
                code_data["Befundtext"] = pd.to_numeric(code_data["Befundtext"], errors='coerce')
                code_data = code_data.sort_values("Befundtext", ascending=True)

                plt.figure(figsize=(10, 6))
                distances_hours = code_data["Distance_to_Midpoint"].dt.total_seconds() / 3600

                # Use sorted Befundtext as labels
                befundtext = code_data["Befundtext"]

                plt.scatter(distances_hours, befundtext, marker='o', alpha=0.5)
                plt.xlabel("Distance to Midpoint (hours)")
                plt.ylabel("Befundtext")
                plt.title(f"Befundtext vs. Distance to Midpoint for CODE: {code}")
                plt.grid(True)
                plt.tight_layout()

                st.pyplot(plt)




def main():
    st.set_page_config(page_title="Patient Data App")
    st.title("Lab Parameter Analysis Tool")
    
    tabs = ["Introduction", "Upload Files", "View and Filter Data", "Select and View Patient Info", "Point-to-Point Analysis", "Interpolated Point-to-Point Analysis", "Generate Patient Results File", "Calculate Distance of Lab Tests to Midpoint of Operation"]
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
    elif selected_tab == "Calculate Distance of Lab Tests to Midpoint of Operation":
        patient_details_file, report_details_file = tab1()
        tab7(patient_details_file, report_details_file)


if __name__ == "__main__":
    main()
