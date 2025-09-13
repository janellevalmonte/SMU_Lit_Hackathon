import pandas as pd
import re
import os
import argparse
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF using PyMuPDF.
    """
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_fields_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts key fields from raw text using heuristics and regex.
    Returns a dict with numeric values where possible.
    """
    result: Dict[str, Any] = {
        'husband_income': None,
        'nafkah_iddah': None,
        'mutaah': None,
        'wife_maintenance': None,
        'is_consent_order': False,
        'is_outlier': False,
    }

    # Normalize
    text_lower = text.lower()

    # Money amounts like S$1,500 or $1500
    money_re = re.compile(r"s?\$\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.\d{2})?", re.IGNORECASE)

    def parse_money_to_int(match_str: str) -> Optional[int]:
        try:
            return int(match_str.replace(',', ''))
        except Exception:
            return None

    def find_amount_after_keyword(keyword_regex: str, window: int = 250) -> Optional[int]:
        kw_re = re.compile(keyword_regex, re.IGNORECASE)
        for m in kw_re.finditer(text_lower):
            start = m.end()
            segment = text[start:start+window]
            m2 = money_re.search(segment)
            if m2:
                return parse_money_to_int(m2.group(1))
        return None

    def find_amount_near_money_with_context(required_terms: List[str], window_before: int = 120, window_after: int = 180) -> Optional[int]:
        for m in money_re.finditer(text):
            start = m.start()
            context = text_lower[max(0, start-window_before): start+window_after]
            if all(term in context for term in required_terms):
                value = parse_money_to_int(m.group(1))
                if value is not None:
                    return value
        return None

    # Income / salary
    income_val = find_amount_after_keyword(r"(husband\'?s\s+income|monthly\s+income|income|salary|take[-\s]?home)")

    # Nafkah iddah and mutaah
    nafkah_val = find_amount_after_keyword(r"nafkah\s+iddah")
    if nafkah_val is None:
        nafkah_val = find_amount_near_money_with_context(["nafkah", "iddah"])  # nearby context

    mutaah_val = find_amount_after_keyword(r"mutaah|mut\s*a\s*ah")
    if mutaah_val is None:
        mutaah_val = find_amount_near_money_with_context(["mutaah"])  # nearby context

    # Wife maintenance per month (heuristic)
    wife_maint_val = find_amount_near_money_with_context(["maintenance", "per month"]) or \
                     find_amount_near_money_with_context(["maintenance", "monthly"]) or \
                     find_amount_near_money_with_context(["maintenance", "for", "wife"]) or \
                     find_amount_near_money_with_context(["maintenance", "to", "her"])  # heuristic

    consent_order_match = re.search(r'consent\s+order', text_lower, re.IGNORECASE)

    result['husband_income'] = income_val
    result['nafkah_iddah'] = nafkah_val
    result['mutaah'] = mutaah_val
    result['wife_maintenance'] = wife_maint_val
    result['is_consent_order'] = bool(consent_order_match)
    return result

def parse_case_file(pdf_path: str) -> Dict[str, Any]:
    """
    Parses a single case PDF and returns extracted fields.
    """
    case_data: Dict[str, Any] = {
        'case_id': os.path.basename(pdf_path),
        'husband_income': None,
        'nafkah_iddah': None,
        'mutaah': None,
        'wife_maintenance': None,
        'is_consent_order': False,
        'is_high_income': False,
        'is_outlier': False
    }

    try:
        text = extract_text_from_pdf(pdf_path)
        extracted = extract_fields_from_text(text)
        case_data.update(extracted)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

    return case_data

def process_case_files(directory: str) -> pd.DataFrame:
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
    high_income_threshold = 4000 # Adjusted to match the iddah calculation
    df['husband_income_numeric'] = pd.to_numeric(df['husband_income'], errors='coerce')
    df['is_high_income'] = df['husband_income_numeric'] > high_income_threshold
    
    return df

def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], k: float = 1.5) -> pd.DataFrame:
    """
    Adds an 'is_outlier' flag based on IQR across provided numeric columns.
    If fewer than 4 valid points in a column, skips that column.
    """
    is_outlier_any = pd.Series(False, index=df.index)
    for col in columns:
        series = pd.to_numeric(df[col], errors='coerce')
        valid = series.dropna()
        if len(valid) < 4:
            continue
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        is_outlier_col = (series < lower) | (series > upper)
        is_outlier_col = is_outlier_col.fillna(False)
        is_outlier_any = is_outlier_any | is_outlier_col
    df['is_outlier'] = is_outlier_any
    return df

def filter_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out consent orders, high-income cases, and outliers.
    """
    if 'is_outlier' not in df.columns:
        df = df.copy()
        df['is_outlier'] = False
    filtered_df = df[
        (df['is_consent_order'] == False) &
        (df['is_high_income'] == False) &
        (df['is_outlier'] == False)
    ].copy()
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

def export_to_csv(df, output_path):
    """
    Exports the provided DataFrame to a CSV file at the given path.
    """
    # Format missing extracted fields as "Not Applicable" only for export
    export_df = df.copy()
    na_fields = [c for c in ['husband_income', 'nafkah_iddah', 'mutaah', 'wife_maintenance'] if c in export_df.columns]
    for c in na_fields:
        export_df[c] = export_df[c].where(export_df[c].notna(), "Not Applicable")
    export_df.to_csv(output_path, index=False)
    print(f"Saved CSV to: {output_path}")

def export_to_excel(raw_df: pd.DataFrame, filtered_df: pd.DataFrame, summary: Dict[str, Any], output_path: str) -> None:
    """
    Exports raw data, filtered data, and summary to an Excel workbook with multiple sheets.
    Falls back gracefully if openpyxl is not installed.
    """
    try:
        # Prepare export copies with "Not Applicable" markers for missing extracted fields
        def fmt(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            na_fields = [c for c in ['husband_income', 'nafkah_iddah', 'mutaah', 'wife_maintenance'] if c in out.columns]
            for c in na_fields:
                out[c] = out[c].where(out[c].notna(), "Not Applicable")
            return out

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            fmt(raw_df).to_excel(writer, index=False, sheet_name='raw_cases')
            fmt(filtered_df).to_excel(writer, index=False, sheet_name='filtered_cases')
            summary_df = pd.DataFrame([summary]) if summary else pd.DataFrame()
            summary_df.to_excel(writer, index=False, sheet_name='summary')
        print(f"Saved Excel to: {output_path}")
    except ModuleNotFoundError:
        print("openpyxl is not installed. Skipping Excel export. Install with: pip install openpyxl")

# --- Main execution flow ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract case fields from PDFs and export to spreadsheets.")
    parser.add_argument("--input-dir", default="syariah_cases", help="Directory containing PDF case files.")
    parser.add_argument("--out-csv-raw", default=None, help="Path to write raw extracted CSV.")
    parser.add_argument("--out-csv-filtered", default=None, help="Path to write filtered CSV.")
    parser.add_argument("--out-xlsx", default=None, help="Path to write Excel workbook.")
    parser.add_argument("--no-excel", action="store_true", help="Do not write Excel workbook.")
    args = parser.parse_args()

    input_dir = args.input_dir
    os.makedirs(input_dir, exist_ok=True)

    # Process files (raw extraction)
    raw_df = process_case_files(input_dir)

    # Outlier detection (across nafkah_iddah, mutaah, wife_maintenance)
    numeric_cols = [c for c in ['nafkah_iddah', 'mutaah', 'wife_maintenance'] if c in raw_df.columns]
    raw_df = detect_outliers_iqr(raw_df, numeric_cols)

    # Filter
    filtered_df = filter_cases(raw_df)

    # Update formulae from filtered data
    avg_nafkah, avg_mutaah, avg_iddah = update_formulae(filtered_df)
    summary = {
        'num_raw_cases': len(raw_df),
        'num_filtered_cases': len(filtered_df),
        'avg_nafkah_iddah': avg_nafkah,
        'avg_mutaah': avg_mutaah,
        'avg_calculated_iddah': avg_iddah,
    }

    # Outputs
    default_raw_csv = os.path.join(input_dir, "cases_raw.csv")
    default_filtered_csv = os.path.join(input_dir, "cases_filtered.csv")
    default_xlsx = os.path.join(input_dir, "cases.xlsx")

    raw_csv_path = args.out_csv_raw or default_raw_csv
    filtered_csv_path = args.out_csv_filtered or default_filtered_csv
    xlsx_path = args.out_xlsx or default_xlsx

    export_to_csv(raw_df, raw_csv_path)
    export_to_csv(filtered_df, filtered_csv_path)
    if not args.no_excel:
        export_to_excel(raw_df, filtered_df, summary, xlsx_path)
