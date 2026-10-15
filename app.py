from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "screenwise_screen_time_model.pkl"
DATA_PATH = BASE_DIR / "data" / "user_behavior_dataset.csv"

app = Flask(__name__)

FEATURE_COLUMNS = [
    "Device Model",
    "Operating System",
    "App Usage Time (min/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
    "Gender",
]
TARGET_COLUMN = "Screen On Time (hours/day)"

model = joblib.load(MODEL_PATH)
dataset = pd.read_csv(DATA_PATH)

# Dropdown choices and sensible bounds are taken from the supplied dataset.
device_models = sorted(dataset["Device Model"].dropna().astype(str).unique().tolist())
operating_systems = sorted(dataset["Operating System"].dropna().astype(str).unique().tolist())
genders = sorted(dataset["Gender"].dropna().astype(str).unique().tolist())

NUMERIC_COLUMNS = {
    "age": "Age",
    "app_usage": "App Usage Time (min/day)",
    "battery_drain": "Battery Drain (mAh/day)",
    "apps_installed": "Number of Apps Installed",
    "data_usage": "Data Usage (MB/day)",
}
limits = {
    key: {
        "min": int(dataset[column].min()),
        "max": int(dataset[column].max()),
    }
    for key, column in NUMERIC_COLUMNS.items()
}

DEFAULTS = {
    "device_model": str(dataset["Device Model"].mode().iloc[0]),
    "operating_system": str(dataset["Operating System"].mode().iloc[0]),
    "age": int(dataset["Age"].median()),
    "gender": str(dataset["Gender"].mode().iloc[0]),
    "app_usage": int(dataset["App Usage Time (min/day)"].median()),
    "battery_drain": int(dataset["Battery Drain (mAh/day)"].median()),
    "apps_installed": int(dataset["Number of Apps Installed"].median()),
    "data_usage": int(dataset["Data Usage (MB/day)"].median()),
}


def make_features(values):
    """Build one-row DataFrame in the exact order/names used for model training."""
    row = {
        "Device Model": values["device_model"],
        "Operating System": values["operating_system"],
        "App Usage Time (min/day)": values["app_usage"],
        "Battery Drain (mAh/day)": values["battery_drain"],
        "Number of Apps Installed": values["apps_installed"],
        "Data Usage (MB/day)": values["data_usage"],
        "Age": values["age"],
        "Gender": values["gender"],
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_screen_time(values):
    prediction = float(model.predict(make_features(values))[0])
    return max(0.0, prediction)


def generate_insights(screen_time, app_usage, battery_drain, apps_installed):
    insights = []

    # Screen-time suggestions
    if screen_time >= 8:
        insights.append(
            "Your estimated screen time is very high. "
            "Try setting app-free periods during study time and before sleep."
        )
    elif screen_time >= 6:
        insights.append(
            "Your estimated screen time is moderately high. "
            "Consider taking regular breaks and setting daily screen-time limits."
        )
    else:
        insights.append(
            "Your estimated screen time is below 6 hours. "
            "Continue maintaining a balanced routine and taking regular breaks."
        )

    # App-usage suggestions
    if app_usage >= 300:
        insights.append(
            "Your daily app usage is high. Try the 20-20-20 rule during "
            "long screen sessions: every 20 minutes, look at something "
            "20 feet away for 20 seconds."
        )
    elif app_usage >= 180:
        insights.append(
            "You spend several hours using apps daily. Consider scheduling "
            "short breaks between study, entertainment, and social media sessions."
        )
    else:
        insights.append(
            "Your recorded app usage is relatively low. Keep prioritizing "
            "important tasks and avoid unnecessary scrolling."
        )

    # Battery-drain suggestions
    if battery_drain >= 2500:
        insights.append(
            "Your recorded battery drain is high. Check your phone's "
            "battery settings to identify apps consuming the most power."
        )
    else:
        insights.append(
            "Your recorded battery drain is below the high-drain threshold. "
            "Continue monitoring battery usage if you notice unusual drain."
        )

    # Installed-app suggestions
    if apps_installed >= 50:
        insights.append(
            "You have many apps installed. Review unused apps and remove "
            "those you no longer need."
        )
    else:
        insights.append(
            "Your number of installed apps is below 50. Periodically review "
            "unused apps to keep your phone organized."
        )

    return insights

    insights = []

    if screen_time >= 7:
        insights.append(
            "Your screen time is high. Try setting app-free periods during study time."
        )

    if app_usage >= 300:
        insights.append(
            "You spend significant time using apps. Take short breaks between sessions."
        )

    if battery_drain >= 2500:
        insights.append(
            "Your battery drain is high. Review battery usage in your phone settings."
        )

    if apps_installed >= 50:
        insights.append(
            "You have many installed apps. Consider reviewing apps you rarely use."
        )

    if not insights:
        insights.append(
            "Your usage is within these suggestion thresholds. Keep maintaining a balanced routine."
        )

    return insights


def get_form_values(form):
    values = {
        "device_model": form.get("device_model", DEFAULTS["device_model"]),
        "operating_system": form.get("operating_system", DEFAULTS["operating_system"]),
        "gender": form.get("gender", DEFAULTS["gender"]),
    }
    if values["device_model"] not in device_models:
        raise ValueError("Please select a device model from the list.")
    if values["operating_system"] not in operating_systems:
        raise ValueError("Please select an operating system from the list.")
    if values["gender"] not in genders:
        raise ValueError("Please select a gender category from the list.")

    for key, column in NUMERIC_COLUMNS.items():
        try:
            number = float(form.get(key, DEFAULTS[key]))
        except (TypeError, ValueError):
            raise ValueError(f"Please enter a valid number for {key.replace('_', ' ')}.")
        low, high = limits[key]["min"], limits[key]["max"]
        if not number.is_integer():
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a whole number.")
        number = int(number)
        if number < low or number > high:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be between {low} and {high}.")
        values[key] = number

    return values


def page_context(values=None, prediction=None, scenario_prediction=None,
                 estimated_difference=0.0, reduction_minutes=30,
                 goal_estimate=None, insights=None, error=None):
    return {
        "values": values or DEFAULTS.copy(),
        "prediction": prediction,
        "scenario_prediction": scenario_prediction,
        "estimated_difference": estimated_difference,
        "reduction_minutes": reduction_minutes,
        "goal_estimate": goal_estimate,
        "insights": insights or [],
        "error": error,
        "device_models": device_models,
        "operating_systems": operating_systems,
        "genders": genders,
        "limits": limits,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", **page_context())

    values = DEFAULTS.copy()
    try:
        values = get_form_values(request.form)
        reduction_minutes = int(request.form.get("reduction_minutes", "30"))
        if reduction_minutes not in (10, 20, 30, 60):
            raise ValueError("Choose a reduction of 10, 20, 30, or 60 minutes.")

        prediction = predict_screen_time(values)

        # Model-based scenario: modify app-usage input and run the same trained model.
        scenario_values = values.copy()
        min_app_usage = limits["app_usage"]["min"]
        scenario_values["app_usage"] = max(
            min_app_usage, values["app_usage"] - reduction_minutes
        )
        scenario_prediction = predict_screen_time(scenario_values)
        estimated_difference = prediction - scenario_prediction

        # Separate hypothetical target: arithmetic only, not an ML prediction.
        goal_estimate = max(0.0, prediction - reduction_minutes / 60.0)
        insights = generate_insights(
            prediction,
            values["app_usage"],
            values["battery_drain"],
            values["apps_installed"],
        )

        return render_template(
            "index.html",
            **page_context(
                values=values,
                prediction=prediction,
                scenario_prediction=scenario_prediction,
                estimated_difference=estimated_difference,
                reduction_minutes=reduction_minutes,
                goal_estimate=goal_estimate,
                insights=insights,
            ),
        )
    except ValueError as exc:
        reduction_minutes = request.form.get("reduction_minutes", "30")
        try:
            reduction_minutes = int(reduction_minutes)
        except (TypeError, ValueError):
            reduction_minutes = 30
        return render_template(
            "index.html",
            **page_context(values=values, reduction_minutes=reduction_minutes, error=str(exc)),
        ), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            **page_context(values=values, error="The prediction could not be completed. Check that the model and dataset files are in the expected folders."),
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
