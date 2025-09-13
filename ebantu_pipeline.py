import pandas as pd
import re
import os
import argparse
import json
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import fitz  # PyMuPDF
import openai  # Required for LLM extraction


# Load environment variables from .env if present
load_dotenv()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF using PyMuPDF.
    """
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_pages_from_pdf(pdf_path: str) -> List[str]:
    """
    Returns a list of page texts for targeted LLM extraction.
    """
    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages

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

def parse_case_file(pdf_path: str, llm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single case PDF and returns extracted fields.
    """
    case_data: Dict[str, Any] = {
        'case_id': os.path.basename(pdf_path),
        'husband_income': None,
        'women_salary': None,
        'marriage_length': None,
        'number_of_children': None,
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

        # Optionally enhance with LLM if fields missing
        if llm.get('use_llm') and any(case_data.get(k) in (None, "", float('nan')) for k in ['husband_income','nafkah_iddah','mutaah','wife_maintenance']):
            llm_result = llm_extract_fields(pdf_path, llm)
            case_data.update({k: v for k, v in llm_result.items() if v not in (None, "")} )
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

    return case_data

def process_case_files(directory: str, llm: Dict[str, Any]) -> pd.DataFrame:
    """
    Processes all PDF files in a given directory and returns a DataFrame.
    """
    all_case_data = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            data = parse_case_file(pdf_path, llm)
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

def _score_pages_for_extraction(pages: List[str]) -> List[Tuple[int, int]]:
    """
    Returns list of (page_index, score) based on keyword hits.
    """
    keywords = [
        'nafkah', 'iddah', 'mutaah', 'maintenance', 'income', 'salary', 'per month', 'monthly', 'award'
    ]
    scored: List[Tuple[int, int]] = []
    for idx, txt in enumerate(pages):
        tl = txt.lower()
        score = sum(tl.count(k) for k in keywords)
        scored.append((idx, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit]

def llm_extract_fields(pdf_path: str, llm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses an LLM to extract fields and provide evidence. Requires OPENAI_API_KEY when using OpenAI.
    Returns numeric values where applicable and evidence strings.
    """
    result: Dict[str, Any] = {
        'husband_income': None,
        'nafkah_iddah': None,
        'mutaah': None,
        'wife_maintenance': None,
        'husband_income_evidence': None,
        'nafkah_iddah_evidence': None,
        'mutaah_evidence': None,
        'wife_maintenance_evidence': None,
    }
    if openai is None:
        print("LLM extraction requested but openai package not installed. Skipping.")
        return result
    if not os.getenv('OPENAI_API_KEY'):
        print("LLM extraction requested but OPENAI_API_KEY is not set. Skipping.")
        return result

    pages = extract_pages_from_pdf(pdf_path)
    scored = _score_pages_for_extraction(pages)
    top = [i for i, _ in scored[:max(1, int(llm.get('llm_pages', 4)))]]

    # Build prompt with top pages
    case_id = os.path.basename(pdf_path)
    content_parts: List[str] = []
    for pi in top:
        page_label = f"Page {pi+1}"
        content_parts.append(f"{page_label}:\n{_truncate(pages[pi], 3500)}")
    joined_pages = "\n\n".join(content_parts)

    system_msg = (
        "You are a meticulous legal data extractor. Extract only the requested fields from Syariah/civil court judgments. "
        "Return strict JSON that conforms to the schema. If a field is not clearly present, use null. For each non-null field, include a short evidence quote and page number."
    )
    schema = {
        "case_id": case_id,
        "husband_income": None,
        "women_salary": None,
        "marriage_length": None,
        "number_of_children": None,
        "mutaah": None,
        "nafkaah_iddah": None
    }
    user_prompt = (
        "Extract these fields as integers or null ONLY: husband_income, women_salary, marriage_length, number_of_children, mutaah, nafkaah_iddah. "
        "Rules: 1) Return ONLY one JSON object with EXACTLY these keys; no extra keys. 2) Values must be integers or null (no strings, no nested objects). "
        "3) Remove currency symbols and decimals (round to nearest integer). 4) If not explicitly stated, use null.\n"
        "Contextual formulas (for reference only; do NOT compute, just extract stated values):\n"
        "- Iddah: 0.14 * salary + 47; lower = round_nearest_100(0.14 * salary - 3); upper = round_nearest_100(0.14 * salary + 197)\n"
        "- Mutaah per day: round(0.00096 * salary + 0.85); lower = round(0.00096 * salary - 0.85); upper = round(0.00096 * salary + 1.85)\n"
        "Schema template (follow keys and types exactly):\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Document excerpts to analyze (top {int(llm.get('llm_pages', 4))} pages by relevance):\n{joined_pages}"
    )

    try:
        # OpenRouter/OpenAI-compatible client (openai>=1.x)
        base_url = os.getenv('OPENAI_BASE_URL') or 'https://openrouter.ai/api/v1'
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY is not set')

        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # Resolve model strictly from environment/config, defaulting to DeepSeek R1 on OpenRouter
        model_name = llm.get('llm_model') or os.getenv('LLM_MODEL') or 'deepseek/deepseek-r1:free'

        # Optional OpenRouter ranking headers via env
        extra_headers = {
            'HTTP-Referer': os.getenv('OPENAI_HTTP_REFERER', ''),
            'X-Title': os.getenv('OPENAI_X_TITLE', ''),
        }
        # Remove empty headers
        extra_headers = {k: v for k, v in extra_headers.items() if v}

        print(f"Sending request to {model_name} LLM, this may take a while for LLM to respond...")

        # Retry/backoff and strict JSON parsing
        timeout_s = int(os.getenv('LLM_TIMEOUT', '45'))
        max_retries = int(os.getenv('LLM_MAX_RETRIES', '3'))

        def coerce_json(text: str) -> Dict[str, Any]:
            try:
                return json.loads(text)
            except Exception:
                pass
            if text.strip().startswith('```'):
                stripped = re.sub(r'^```[a-zA-Z0-9_\-]*\n', '', text.strip())
                stripped = re.sub(r'```\s*$', '', stripped)
                try:
                    return json.loads(stripped)
                except Exception:
                    text = stripped
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                return json.loads(candidate)
            raise ValueError('Failed to parse JSON from LLM response')

        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                chat = client.chat.completions.create(
                    extra_headers=extra_headers or None,
                    extra_body={},
                    model=model_name,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
                    timeout=timeout_s,
                )
                content = chat.choices[0].message.content
                data = coerce_json(content)
                print(f"LLM response received: {data}")
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                else:
                    raise last_err
        def to_int(v):
            if v is None or v == "":
                return None
            if isinstance(v, dict):
                for key in ("value", "amount", "number", "num", "val"):
                    if key in v:
                        return to_int(v[key])
                # Fallback: stringify and extract digits
                v = json.dumps(v)
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, (int, float)):
                try:
                    return int(v)
                except Exception:
                    return None
            if isinstance(v, str):
                m = re.search(r"[-+]?\d[\d,]*", v)
                if not m:
                    return None
                digits = m.group(0).replace(",", "")
                try:
                    return int(digits)
                except Exception:
                    return None
            return None
        
        # Map fields from LLM data to our result keys
        result['husband_income'] = to_int(data.get('husband_income'))
        result['women_salary'] = to_int(data.get('women_salary'))
        result['marriage_length'] = to_int(data.get('marriage_length'))
        result['number_of_children'] = to_int(data.get('number_of_children'))
        result['mutaah'] = to_int(data.get('mutaah'))
        result['nafkah_iddah'] = to_int(data.get('nafkaah_iddah'))

        ev = data.get('evidence') or {}
        def mk_ev(key: str) -> Optional[str]:
            e = ev.get(key) or {}
            q = e.get('quote')
            p = e.get('page')
            if q and p:
                return f"{q} (p. {p})"
            if q:
                return q
            return None
        result['husband_income_evidence'] = mk_ev('husband_income')
        result['nafkah_iddah_evidence'] = mk_ev('nafkah_iddah')
        result['mutaah_evidence'] = mk_ev('mutaah')
        result['wife_maintenance_evidence'] = mk_ev('wife_maintenance')
    except Exception as e:
        print(f"LLM extraction failed for {case_id}: {e}")
    return result

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
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL"), help="LLM model name (OpenAI). Defaults from LLM_MODEL in .env")
    parser.add_argument("--llm-pages", type=int, default=4, help="Max number of top-relevance pages to send to LLM.")
    args = parser.parse_args()

    input_dir = args.input_dir
    os.makedirs(input_dir, exist_ok=True)

    # Configure LLM behavior from .env (dotenv)
    env_use_llm = os.getenv("USE_LLM", "false").lower() == "true"
    env_llm_model = os.getenv("LLM_MODEL")
    llm_config = {
        'use_llm': env_use_llm,
        'llm_model': env_llm_model,
        'llm_pages': max(1, int(args.llm_pages)),
    }

    # Process files (raw extraction)
    raw_df = process_case_files(input_dir, llm_config)

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
