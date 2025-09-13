def calculate_iddah(salary: float):


    # Condition: if salary > 4000
    if salary > 4000:
        return {
            "message": "Salary above $4000. Please seek legal advice (outside LAB scope)."
        }

    # Condition: if salary = 0
    if salary == 0:
        return {
            "iddah": 0,
            "lower_range": 0,
            "upper_range": 0
        }

    # Base formula
    base = 0.14 * salary + 47

    # Rounding helper
    def round_nearest_100(x):
        return int(round(x / 100.0) * 100)

    # Apply rounding
    iddah = round_nearest_100(base)

    # Calculate ranges
    lower = round_nearest_100(base - 50)
    upper = round_nearest_100(base + 150)

    # Conditions: no negative values
    if iddah < 0:
        iddah = 0
    if lower < 0:
        lower = 0

    return {
        "iddah": iddah,
        "lower_range": lower,
        "upper_range": upper
    }

# 🔹 Example usage:
print(calculate_iddah(2000))   # test with salary $2000
print(calculate_iddah(0))      # test with salary $0
print(calculate_iddah(5000))   # test with salary $5000 (above scope)
