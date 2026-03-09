# NumPy Analytics Toolkit

A modular data analytics dashboard built using **NumPy** and **Streamlit**.
This project demonstrates how core data analysis operations can be implemented using **NumPy from scratch** and exposed through a simple interactive web interface.

The application allows users to upload CSV datasets, perform statistical analysis, apply feature scaling, compute correlations, and detect outliers.

---

## Features

* Upload CSV datasets
* Automatic statistical summary
* Min-Max feature scaling
* Standard scaling (Z-score normalization)
* Correlation matrix computation
* Outlier detection using Z-score
* Interactive web interface built with Streamlit

---

## Tech Stack

* Python
* NumPy
* Streamlit
* Git & GitHub

---

## Project Structure

```
numpy_analytics_project
│
├── data/
│
├── src/
│   ├── __init__.py
│   ├── loader.py          # Load CSV dataset
│   ├── statistics.py      # Statistical calculations
│   ├── scaler.py          # Feature scaling methods
│   └── analyzer.py        # Correlation and outlier detection
│
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Modules Overview

### DatasetLoader

Handles CSV dataset loading and converts it into a NumPy array.

### StatisticsEngine

Computes descriptive statistics including:

* Mean
* Median
* Standard Deviation
* Variance
* Minimum
* Maximum
* Sum

### DataScaler

Implements feature scaling techniques:

* Min-Max Scaling
* Standard Scaling (Z-score)

### DataAnalyzer

Performs analytical operations such as:

* Correlation Matrix
* Outlier Detection

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Aditya-analytics/numpy-data-analytics-engine
cd numpy_analytic_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload a CSV dataset to start analyzing.

---

## Example Dataset

Example columns supported:

```
Age, Salary, Score
```

All values should be numeric since the analytics engine is built directly on NumPy arrays.

---

## Learning Objectives

This project was built to practice:

* NumPy-based data analysis
* Modular Python project structure
* Git feature branch workflow
* Streamlit application development
* Clean code organization

---

## Future Improvements

Possible enhancements:

* Correlation heatmap visualization
* Dataset validation
* Feature selection tools
* Data distribution plots
* Export analysis results

---

## License

This project is open-source and available under the MIT License.

---

## Author
Aditya 
Developed as part of a data analytics learning project using NumPy and Streamlit.
