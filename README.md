# ScreenWise – Screen Time Prediction and Digital Wellbeing

## Overview

ScreenWise is a machine learning-based web application that estimates a user's daily screen-on time using device and usage-related information. It also provides personalized digital wellbeing suggestions to encourage healthier screen habits.

The application uses a trained machine learning model through a Flask web interface. Users can enter their details, view an estimated screen time, and explore a hypothetical screen-time reduction scenario.

## Problem Statement

Excessive screen usage can affect productivity, study routines, sleep schedules, and digital wellbeing. Users may find it difficult to understand their usage patterns and identify opportunities to develop healthier habits.

ScreenWise aims to make screen-time estimation and basic digital wellbeing guidance accessible through a simple web application.

## Objectives

- Estimate daily screen-on time using machine learning.
- Use device and usage-related features for prediction.
- Allow users to explore hypothetical screen-time reduction scenarios.
- Provide rule-based suggestions based on entered usage information.
- Present results through a simple, user-friendly web interface.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas
- **Model Loading:** Joblib
- **Frontend:** HTML and CSS
- **Dataset Format:** CSV

## Key Features

- **Screen-Time Estimation:** Predicts estimated daily screen-on time from user-provided inputs.
- **Scenario Comparison:** Compares the model's estimate before and after a hypothetical reduction in app usage.
- **Digital Wellbeing Suggestions:** Provides rule-based recommendations based on screen time, app usage, battery drain, and installed apps.
- **Input Validation:** Checks that entered values are valid and within the available dataset's ranges.
- **Responsive Interface:** Designed to make the application accessible through a web browser.

## Machine Learning Methodology

ScreenWise uses a trained Random Forest regression pipeline to estimate screen-on time.

### Input Features

- Device Model
- Operating System
- App Usage Time (minutes per day)
- Battery Drain (mAh per day)
- Number of Apps Installed
- Data Usage (MB per day)
- Age
- Gender

### Target Variable

- Screen On Time (hours per day)

### Prediction Workflow

1. The user enters device and usage-related information.
2. The Flask application validates the inputs.
3. The trained model processes the input features.
4. The application displays the estimated screen-on time.
5. A hypothetical scenario is evaluated by reducing the app-usage input.
6. Rule-based suggestions are displayed to support digital wellbeing.

## Project Structure

```text
ScreenWise/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── user_behavior_dataset.csv
├── models/
│   └── screenwise_screen_time_model.pkl
└── templates/
    └── index.html
```

The exact files in the `models/` folder may vary depending on the final model version used.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Git (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ScreenWise.git
cd ScreenWise
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Check Project Files

Make sure the dataset is available at:

```text
data/user_behavior_dataset.csv
```

Make sure the trained model is available at:

```text
models/screenwise_screen_time_model.pkl
```

The application requires both files at these locations unless the code is updated to use different paths.

### 5. Run the Application

```bash
python app.py
```

Open the local address shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## Important Limitations

- Predictions depend on the dataset and trained model.
- The application estimates screen-on time from supplied information; it does not directly read live phone usage statistics.
- The scenario comparison is hypothetical and does not guarantee that reducing app usage will reduce actual screen-on time by the same amount.
- Digital wellbeing suggestions are rule-based recommendations, not predictions from a separate machine learning model.
- Prediction quality may vary for inputs that differ from the training data.

## Future Enhancements

- Integrate phone usage statistics where platform permissions allow.
- Add visual charts to help users understand usage patterns.
- Provide optional usage history and progress tracking.
- Improve personalization using more diverse datasets.
- Evaluate the model on additional data to assess generalization.

## Conclusion

ScreenWise demonstrates how machine learning and web development can be combined to estimate screen-on time and provide accessible digital wellbeing suggestions. It serves as an educational project for exploring regression, data processing, model integration, and Flask-based application development.