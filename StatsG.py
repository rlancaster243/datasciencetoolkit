import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from ttkthemes import ThemedTk
from fpdf import FPDF
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tpot import TPOTRegressor
import plotly.express as px
import plotly.graph_objects as go
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DataHandler:
    """Class for handling data operations like loading, cleaning, and selecting columns."""

    def __init__(self):
        self.data = None
        self.selected_columns = None

    def load_data_from_string(self, data_str):
        try:
            data = list(map(float, data_str.split(',')))
            if len(data) < 2:
                raise ValueError("At least two values are required.")
            self.data = pd.DataFrame(data, columns=["Value"])
            logging.info("Data loaded successfully from string input.")
        except ValueError as e:
            logging.error(f"Invalid input data: {e}")
            raise e

    def load_data_from_file(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        try:
            if file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file type.")
            logging.info(f"Data loaded successfully from {file_path}.")
        except Exception as e:
            logging.error(f"Error loading data from file: {e}")
            raise e

    def clean_data(self, remove_duplicates=False, handle_missing="None"):
        if remove_duplicates:
            self.data = self.data.drop_duplicates()
            logging.info("Duplicates removed from data.")
        
        if handle_missing == "Drop":
            self.data = self.data.dropna()
            logging.info("Missing data rows dropped.")
        elif handle_missing == "Mean":
            self.data = self.data.fillna(self.data.mean())
            logging.info("Missing data filled with mean values.")
        elif handle_missing == "Median":
            self.data = self.data.fillna(self.data.median())
            logging.info("Missing data filled with median values.")
        elif handle_missing != "None":
            logging.warning(f"Unknown option for handling missing data: {handle_missing}")
        
        logging.info("Data cleaning applied.")

    def select_columns(self, selected_columns):
        if self.data is not None and selected_columns:
            self.selected_columns = selected_columns
            self.data = self.data[selected_columns]
        else:
            raise ValueError("Data is not loaded or no columns selected.")


class StatisticalCalculator:
    """Class for performing statistical analysis on the data."""

    def __init__(self, data):
        self.data = data

    def mean(self):
        return self.data.mean()

    def median(self):
        return self.data.median()

    def mode(self):
        try:
            return self.data.mode().iloc[0]
        except:
            return "No unique mode"

    def variance(self):
        return self.data.var()

    def standard_deviation(self):
        return self.data.std()

    def z_scores(self):
        return (self.data - self.mean()) / self.standard_deviation()

    def linear_regression(self):
        x = np.arange(len(self.data))
        y = self.data.values.flatten()
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    def decision_tree(self):
        x = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data.values.flatten()
        model = DecisionTreeRegressor()
        model.fit(x, y)
        return model

    def gradient_boosting(self):
        x = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data.values.flatten()
        model = GradientBoostingRegressor()
        model.fit(x, y)
        return model

    def time_series_forecast(self, steps=5):
        if self.data.isna().sum().sum() > 0:
            logging.warning("Data contains missing values. Results may be inaccurate.")
        model = ARIMA(self.data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def sarima_forecast(self, steps=5):
        model = SARIMAX(self.data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def lstm_forecast(self, steps=5):
        data = self.data.values.reshape(-1, 1)
        data = (data - data.mean()) / data.std()  # Standardize data
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(data[:-steps].reshape(-1, 1, 1), data[:-steps], epochs=300, verbose=0)
        predictions = model.predict(data[-steps:].reshape(-1, 1, 1))
        return predictions

    def knn(self, k=3):
        x = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data.values.flatten()
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(x, y)
        return model

    def svm(self):
        x = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data.values.flatten()
        model = SVR()
        model.fit(x, y)
        return model

    def auto_ml(self, generations=5, population_size=20):
        x = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data.values.flatten()
        model = TPOTRegressor(generations=generations, population_size=population_size, verbosity=2)
        model.fit(x, y)
        return model


class StatsApp:
    """Main application class for the Data Science Toolbox GUI."""

    def __init__(self, root):
        self.root = root
        self.root.title("Data Science Toolbox")

        self.data_handler = DataHandler()

        self.setup_tabs()
        self.create_data_tab()
        self.create_analysis_tab()
        self.create_visualization_tab()
        self.create_reporting_tab()
        self.create_export_tab()

    def setup_tabs(self):
        self.tab_control = ttk.Notebook(self.root)
        self.data_tab = ttk.Frame(self.tab_control)
        self.analysis_tab = ttk.Frame(self.tab_control)
        self.visualization_tab = ttk.Frame(self.tab_control)
        self.reporting_tab = ttk.Frame(self.tab_control)
        self.export_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.data_tab, text='Data')
        self.tab_control.add(self.analysis_tab, text='Analysis')
        self.tab_control.add(self.visualization_tab, text='Visualization')
        self.tab_control.add(self.reporting_tab, text='Reporting')
        self.tab_control.add(self.export_tab, text='Export Data')

        self.tab_control.pack(expand=1, fill='both')

        self.disable_tabs()

    def disable_tabs(self):
        for i in range(1, 5):
            self.tab_control.tab(i, state="disabled")

    def enable_tabs(self):
        for i in range(1, 5):
            self.tab_control.tab(i, state="normal")

    def create_data_tab(self):
        self.data_label = tk.Label(self.data_tab, text="Enter data (comma-separated) or upload a file:")
        self.data_label.pack(pady=10)

        self.data_entry = tk.Entry(self.data_tab, width=50)
        self.data_entry.pack()

        self.upload_button = tk.Button(self.data_tab, text="Upload File", command=self.upload_file)
        self.upload_button.pack(pady=10)

        self.clean_data_button = tk.Button(self.data_tab, text="Clean Data", command=self.clean_data)
        self.clean_data_button.pack(pady=10)

        self.load_button = tk.Button(self.data_tab, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=10)

    def create_analysis_tab(self):
        self.analysis_label = tk.Label(self.analysis_tab, text="Select statistical methods:")
        self.analysis_label.pack(pady=10)

        self.analysis_frame = ttk.Labelframe(self.analysis_tab, text='Methods')
        self.analysis_frame.pack(pady=10, padx=10, fill="both", expand="yes")

        self.mean_var = tk.IntVar()
        self.median_var = tk.IntVar()
        self.mode_var = tk.IntVar()
        self.variance_var = tk.IntVar()
        self.std_dev_var = tk.IntVar()
        self.linear_regression_var = tk.IntVar()
        self.decision_tree_var = tk.IntVar()
        self.gradient_boosting_var = tk.IntVar()
        self.time_series_forecast_var = tk.IntVar()
        self.sarima_var = tk.IntVar()
        self.lstm_var = tk.IntVar()
        self.knn_var = tk.IntVar()
        self.svm_var = tk.IntVar()
        self.auto_ml_var = tk.IntVar()

        ttk.Checkbutton(self.analysis_frame, text="Mean", variable=self.mean_var).grid(row=0, column=0, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Median", variable=self.median_var).grid(row=0, column=1, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Mode", variable=self.mode_var).grid(row=0, column=2, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Variance", variable=self.variance_var).grid(row=1, column=0, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Standard Deviation", variable=self.std_dev_var).grid(row=1, column=1, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Linear Regression", variable=self.linear_regression_var).grid(row=2, column=0, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Decision Tree", variable=self.decision_tree_var).grid(row=2, column=1, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Gradient Boosting", variable=self.gradient_boosting_var).grid(row=2, column=2, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Time Series Forecast (ARIMA)", variable=self.time_series_forecast_var).grid(row=3, column=0, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="SARIMA Forecast", variable=self.sarima_var).grid(row=3, column=1, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="LSTM Forecast", variable=self.lstm_var).grid(row=4, column=0, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="K-Nearest Neighbors", variable=self.knn_var).grid(row=4, column=1, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="Support Vector Machine", variable=self.svm_var).grid(row=4, column=2, sticky='W')
        ttk.Checkbutton(self.analysis_frame, text="AutoML", variable=self.auto_ml_var).grid(row=5, column=0, sticky='W')

        self.calculate_button = tk.Button(self.analysis_tab, text="Calculate", command=self.calculate_stats)
        self.calculate_button.pack(pady=20)

        self.result_text = tk.Text(self.analysis_tab, height=15, width=70)
        self.result_text.pack(pady=10)

    def create_visualization_tab(self):
        self.visualization_label = tk.Label(self.visualization_tab, text="Select Visualization Type:")
        self.visualization_label.pack(pady=10)

        self.visualization_type = tk.StringVar()
        self.visualization_type.set("Histogram")

        ttk.Radiobutton(self.visualization_tab, text="Histogram", variable=self.visualization_type, value="Histogram").pack(anchor='w')
        ttk.Radiobutton(self.visualization_tab, text="Boxplot", variable=self.visualization_type, value="Boxplot").pack(anchor='w')
        ttk.Radiobutton(self.visualization_tab, text="Scatter Plot", variable=self.visualization_type, value="Scatter Plot").pack(anchor='w')
        ttk.Radiobutton(self.visualization_tab, text="Correlation Matrix", variable=self.visualization_type, value="Correlation Matrix").pack(anchor='w')
        ttk.Radiobutton(self.visualization_tab, text="Heatmap", variable=self.visualization_type, value="Heatmap").pack(anchor='w')
        ttk.Radiobutton(self.visualization_tab, text="Real-time Line Plot", variable=self.visualization_type, value="Real-time Line Plot").pack(anchor='w')

        self.visualize_button = tk.Button(self.visualization_tab, text="Visualize", command=self.visualize_data)
        self.visualize_button.pack(pady=10)

    def create_reporting_tab(self):
        self.reporting_label = tk.Label(self.reporting_tab, text="Save Report:")
        self.reporting_label.pack(pady=10)

        self.save_report_button = tk.Button(self.reporting_tab, text="Save Report as PDF", command=self.save_report)
        self.save_report_button.pack(pady=10)

    def create_export_tab(self):
        self.export_label = tk.Label(self.export_tab, text="Export Cleaned Data:")
        self.export_label.pack(pady=10)

        self.export_csv_button = tk.Button(self.export_tab, text="Export as CSV", command=self.export_as_csv)
        self.export_csv_button.pack(pady=10)

        self.export_excel_button = tk.Button(self.export_tab, text="Export as Excel", command=self.export_as_excel)
        self.export_excel_button.pack(pady=10)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv")])
        if file_path:
            try:
                self.data_handler.load_data_from_file(file_path)
                self.show_column_selection()  # Show dataset structure for column selection
                self.enable_tabs()
            except ValueError as e:
                messagebox.showerror("Error", str(e))

    def show_column_selection(self):
        if self.data_handler.data is not None:
            column_selection_window = tk.Toplevel(self.root)
            column_selection_window.title("Select Columns for Analysis")

            tk.Label(column_selection_window, text="Select columns to include in analysis:").pack(pady=10)

            listbox = tk.Listbox(column_selection_window, selectmode=tk.MULTIPLE)
            for col in self.data_handler.data.columns:
                listbox.insert(tk.END, col)
            listbox.pack(pady=10, padx=10, fill='both', expand=True)

            def confirm_selection():
                selected_indices = listbox.curselection()
                selected_columns = [self.data_handler.data.columns[i] for i in selected_indices]
                try:
                    self.data_handler.select_columns(selected_columns)
                    messagebox.showinfo("Columns Selected", "Columns successfully selected for analysis.")
                    column_selection_window.destroy()
                except ValueError as e:
                    messagebox.showerror("Selection Error", str(e))

            confirm_button = tk.Button(column_selection_window, text="Confirm", command=confirm_selection)
            confirm_button.pack(pady=10)

    def clean_data(self):
        remove_duplicates = messagebox.askyesno("Remove Duplicates", "Do you want to remove duplicates?")
        handle_missing = messagebox.askquestion("Handle Missing Data", "How do you want to handle missing data?",
                                                icon='question', type='none')
        handle_missing_options = {"yes": "Drop", "no": "None"}  # Example mapping, this can be expanded
        handle_missing = handle_missing_options.get(handle_missing, "None")
        self.data_handler.clean_data(remove_duplicates=remove_duplicates, handle_missing=handle_missing)
        messagebox.showinfo("Cleaning Applied", "Data cleaning options have been applied.")

    def load_data(self):
        data_str = self.data_entry.get()
        if data_str:
            try:
                self.data_handler.load_data_from_string(data_str)
                messagebox.showinfo("Data Loaded", "Data successfully loaded.")
                self.enable_tabs()
            except ValueError as e:
                messagebox.showerror("Invalid input", str(e))
                return

    def calculate_stats(self):
        if self.data_handler.data is None:
            messagebox.showerror("No Data", "Please load data before performing analysis.")
            return

        calculator = StatisticalCalculator(self.data_handler.data)
        self.report = ""
        if self.mean_var.get():
            self.report += f"Mean: {calculator.mean()}\n"
        if self.median_var.get():
            self.report += f"Median: {calculator.median()}\n"
        if self.mode_var.get():
            self.report += f"Mode: {calculator.mode()}\n"
        if self.variance_var.get():
            self.report += f"Variance: {calculator.variance()}\n"
        if self.std_dev_var.get():
            self.report += f"Standard Deviation: {calculator.standard_deviation()}\n"
        if self.linear_regression_var.get():
            m, c = calculator.linear_regression()
            self.report += f"Linear Regression: y = {m:.2f}x + {c:.2f}\n"
        if self.decision_tree_var.get():
            model = calculator.decision_tree()
            self.report += f"Decision Tree: Model trained with depth {model.get_depth()} and {model.get_n_leaves()} leaves.\n"
        if self.gradient_boosting_var.get():
            model = calculator.gradient_boosting()
            self.report += f"Gradient Boosting Model: Trained with a gradient boosting regressor.\n"
        if self.time_series_forecast_var.get():
            forecast = calculator.time_series_forecast()
            self.report += f"Time Series Forecast (ARIMA): {forecast.tolist()}\n"
        if self.sarima_var.get():
            forecast = calculator.sarima_forecast()
            self.report += f"SARIMA Forecast: {forecast.tolist()}\n"
        if self.lstm_var.get():
            predictions = calculator.lstm_forecast()
            self.report += f"LSTM Forecast: {predictions.tolist()}\n"
        if self.knn_var.get():
            model = calculator.knn()
            self.report += f"K-Nearest Neighbors Model: Trained with k={model.n_neighbors}.\n"
        if self.svm_var.get():
            model = calculator.svm()
            self.report += f"Support Vector Machine Model: Trained with a support vector regressor.\n"
        if self.auto_ml_var.get():
            model = calculator.auto_ml()
            self.report += f"AutoML Best Pipeline: {model.fitted_pipeline_}\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, self.report)

    def visualize_data(self):
        if not hasattr(self.data_handler, 'data') or self.data_handler.data.empty:
            messagebox.showerror("No data", "Please load data before attempting to visualize.")
            return

        viz_type = self.visualization_type.get()

        if viz_type == "Histogram":
            fig = px.histogram(self.data_handler.data, nbins=20, title="Histogram")
        elif viz_type == "Boxplot":
            fig = px.box(self.data_handler.data, title="Boxplot")
        elif viz_type == "Scatter Plot":
            fig = px.scatter(self.data_handler.data.reset_index(), x="index", y=self.data_handler.data.columns[0], title="Scatter Plot")
        elif viz_type == "Correlation Matrix":
            if self.data_handler.data.shape[1] < 2:
                messagebox.showerror("Insufficient Data", "Correlation Matrix requires at least two columns.")
                return
            corr = self.data_handler.data.corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        elif viz_type == "Heatmap":
            fig = px.imshow(self.data_handler.data, text_auto=True, title="Heatmap")
        elif viz_type == "Real-time Line Plot":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data_handler.data.index, y=self.data_handler.data.iloc[:, 0], mode='lines'))
            fig.update_layout(title="Real-time Line Plot", xaxis_title="Time", yaxis_title="Value")

        fig.show()

    def save_report(self):
        if hasattr(self, 'report') and self.report:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if file_path:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for line in self.report.split('\n'):
                    pdf.cell(200, 10, txt=line, ln=True)
                pdf.output(file_path)
                messagebox.showinfo("Report Saved", f"Report saved as {file_path}")
                logging.info(f"Report saved to {file_path}.")
        else:
            messagebox.showerror("No Report", "Please calculate the statistics first.")

    def export_as_csv(self):
        if hasattr(self.data_handler, 'data') and not self.data_handler.data.empty:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.data_handler.data.to_csv(file_path, index=False)
                messagebox.showinfo("Data Exported", f"Data exported as {file_path}")
                logging.info(f"Data exported to {file_path}.")
        else:
            messagebox.showerror("No Data", "Please load and clean data before attempting to export.")

    def export_as_excel(self):
        if hasattr(self.data_handler, 'data') and not self.data_handler.data.empty:
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if file_path:
                self.data_handler.data.to_excel(file_path, index=False)
                messagebox.showinfo("Data Exported", f"Data exported as {file_path}")
                logging.info(f"Data exported to {file_path}.")
        else:
            messagebox.showerror("No Data", "Please load and clean data before attempting to export.")


if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    app = StatsApp(root)
    root.mainloop()
