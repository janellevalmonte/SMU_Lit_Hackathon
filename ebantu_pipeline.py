import pandas as pd
import re
import os

def parse_case_file(pdf_path):
    """
    Parses a single Syariah Court case PDF to extract relevant data.
    """
    case_data = {
        'case_id': os.path.basename(pdf_path),
        'husband_income': None,
        'nafkah_iddah': None,
        'mutaah': None,
        'is_consent_order': False,
        'is_high_income': False,
        'is_outlier': False
    }

    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text()

        # Regular expressions to find the data
        # Note: These are example patterns and will likely need to be
        # refined based on the actual document format.
        income_match = re.search(r'Husband\'s income:\s*S?\$(\d{1,3}(?:,\d{3})*)', full_text, re.IGNORECASE)
        nafkah_match = re.search(r'Nafkah Iddah awarded:\s*S?\$(\d{1,3}(?:,\d{3})*)', full_text, re.IGNORECASE)
        mutaah_match = re.search(r'Mutaah awarded:\s*S?\$(\d{1,3}(?:,\d{3})*)', full_text, re.IGNORECASE)
        consent_order_match = re.search(r'consent order', full_text, re.IGNORECASE)
        
        if income_match:
            # Clean and convert to integer
            case_data['husband_income'] = int(income_match.group(1).replace(',', ''))
        if nafkah_match:
            case_data['nafkah_iddah'] = int(nafkah_match.group(1).replace(',', ''))
        if mutaah_match:
            case_data['mutaah'] = int(mutaah_match.group(1).replace(',', ''))
        if consent_order_match:
            case_data['is_consent_order'] = True

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

    return case_data

def process_case_files(directory):
    """
    Processes all PDF files in a given directory and returns a DataFrame.
    """
    all_case_data = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            data = parse_case_file(pdf_path)
            all_case_data.append(data)

    df = pd.DataFrame(all_case_data)
    
    # Apply filtering criteria
    # Example: Exclude high-income cases (e.g., > $4,000)
    high_income_threshold = 4000 # Adjusted to match the iddah calculation
    df['is_high_income'] = df['husband_income'] > high_income_threshold
    
    # Filter out cases that don't fit the criteria
    filtered_df = df[
        (df['is_consent_order'] == False) &
        (df['is_high_income'] == False)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Outlier detection can be done here using statistical methods (e.g., Z-score)
    # This is a more complex step and would require specific logic based on the data distribution.
    # For now, we assume this is a manual review step.
    
    return filtered_df

def calculate_iddah(salary: float):
    """
    Calculates iddah based on the provided formula.
    """
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

def update_formulae(df):
    """
    Generates and updates formulae based on the filtered data and calculates
    Iddah based on the provided formula.
    """
    if df.empty:
        print("No cases to analyze after filtering.")
        return None, None, None
    
    # Apply iddah calculation to each valid case
    iddah_results = df['husband_income'].apply(
        lambda x: calculate_iddah(x) if pd.notnull(x) else {}
    )
    
    # Extract the iddah, ranges, and messages into new columns
    df['calculated_iddah'] = iddah_results.apply(lambda x: x.get('iddah', None))
    df['iddah_lower_range'] = iddah_results.apply(lambda x: x.get('lower_range', None))
    df['iddah_upper_range'] = iddah_results.apply(lambda x: x.get('upper_range', None))
    df['iddah_message'] = iddah_results.apply(lambda x: x.get('message', None))

    # Example formula update: a simple average
    avg_nafkah = df['nafkah_iddah'].mean()
    avg_mutaah = df['mutaah'].mean()
    avg_iddah = df['calculated_iddah'].mean()
    
    print(f"Updated Nafkah Iddah Formula (Average): S${avg_nafkah:.2f}")
    print(f"Updated Mutaah Formula (Average): S${avg_mutaah:.2f}")
    print(f"Calculated Iddah Formula (Average): S${avg_iddah:.2f}")

    return avg_nafkah, avg_mutaah, avg_iddah

# --- Main execution flow ---
if __name__ == "__main__":
    # Create a dummy folder with a few empty PDF files for demonstration
    dummy_dir = "syariah_cases"
    os.makedirs(dummy_dir, exist_ok=True)
    
    print(f"Place your PDF case files in the '{dummy_dir}' folder.")
    input("Press Enter to continue once you have added files...")
    
    # Process the files
    clean_data = process_case_files(dummy_dir)
    
    if not clean_data.empty:
        print("\n--- Processed Data ---")
        print(clean_data)
        
        # Generate and update formulae
        update_formulae(clean_data)
    else:
        print("\nNo valid cases were found or all were filtered out.")
