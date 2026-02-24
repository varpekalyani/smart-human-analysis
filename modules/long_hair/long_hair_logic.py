def predict_gender(age, hair_length):
    """
    Rule-based logic:
    - Age 20–30 → classify by hair length.
    - Outside 20–30 → normal detection.
    """
    if hair_length == "none":
        return "No hair detected — cannot classify gender"
    elif 20 <= age <= 30 and hair_length == "long":
        return "Predicted Gender: Female (rule-based)"
    elif 20 <= age <= 30 and hair_length == "short":
        return "Predicted Gender: Male (rule-based)"
    else:
        return f"Normal detection (Age {age} outside 20–30)"