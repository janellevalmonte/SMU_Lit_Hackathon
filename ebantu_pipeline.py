import re
import os
import glob
import argparse
import pandas as pd
import pdfplumber
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------- Utilities ----------
def text_from_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text.append(p.extract_text() or "")
    return "\n".join(text)

def normalize_number(s):
    if s is None: return None
    s = s.replace(',', '').strip()
    m = re.search(r'[-+]?\d+(\.\d+)?', s)
    return float(m.group()) if m else None

# strong regex patterns with currency and commas
RE_INCOME = re.compile(r'(?:income of|husband(?:\'s)? income|monthly income|income was)\s*[^\d\w\-]*([$\u20B9€£]?\s?[\d,]+(?:\.\d+)?)', re.I)
RE_NAFKAH = re.compile(r'nafkah iddah\s*(?:awarded|granted|is|of|:)?\s*[^\d\w\-]*([$\u20B9€£]?\s?[\d,]+(?:\.\d+)?)', re.I)
RE_MUTAAH = re.compile(r'mutaah\s*(?:awarded|granted|is|of|:)?\s*[^\d\w\-]*([$\u20B9€£]?\s?[\d,]+(?:\.\d+)?)', re.I)
RE_CONSENT = re.compile(r'\bconsent order\b|\bconsent judgment\b', re.I)

def find_first(pattern, text):
    m = pattern.search(text)
    return m.group(1) if m else None

# ---------- Pipeline ----------
def process_file(path):
    if path.lower().endswith('.pdf'):
        text = text_from_pdf(path)
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    rec = {'source_file': os.path.basename(path), 'raw_text': text[:4000]}  # store sample for debugging
    inc = find_first(RE_INCOME, text)
    naf = find_first(RE_NAFKAH, text)
    mut = find_first(RE_MUTAAH, text)
    consent_flag = bool(RE_CONSENT.search(text))
    rec.update({
        'husband_income_raw': inc,
        'husband_income': normalize_number(inc),
        'nafkah_iddah_raw': naf,
        'nafkah_iddah': normalize_number(naf),
        'mutaah_raw': mut,
        'mutaah': normalize_number(mut),
        'is_consent_order': consent_flag,
    })
    return rec

def detect_outliers_iqr(series):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return lambda x: (x < low) | (x > high)

def recalibrate_formula(df, target_col='nafkah_iddah', feature_col='husband_income'):
    # Simple linear regression: target = a + b * income
    subset = df[[feature_col, target_col]].dropna()
    if len(subset) < 3:
        return None  # not enough data
    X = subset[[feature_col]].values
    y = subset[target_col].values
    model = LinearRegression().fit(X, y)
    a = float(model.intercept_)
    b = float(model.coef_[0])
    return {'intercept': a, 'slope': b, 'r2': float(model.score(X,y))}

# ---------- main ----------
def main(input_dir='samples', out_csv='cases_extracted.csv', high_income_threshold=10000):
    files = glob.glob(os.path.join(input_dir, '*'))
    records = []
    for f in files:
        try:
            records.append(process_file(f))
        except Exception as e:
            print("Error processing", f, e)
    df = pd.DataFrame(records)
    # filtering flags
    df['is_high_income'] = df['husband_income'].apply(lambda x: x is not None and x > high_income_threshold)
    # outlier detection based on nafkah and mutaah
    naf_outlier_fn = detect_outliers_iqr(df['nafkah_iddah'])
    mut_outlier_fn = detect_outliers_iqr(df['mutaah'])
    df['is_naf_outlier'] = df['nafkah_iddah'].apply(lambda x: naf_outlier_fn(x) if pd.notna(x) else False)
    df['is_mut_outlier'] = df['mutaah'].apply(lambda x: mut_outlier_fn(x) if pd.notna(x) else False)
    # pass_filter = not consent order, not high income, not outlier
    df['pass_filter'] = (~df['is_consent_order']) & (~df['is_high_income']) & (~df['is_naf_outlier']) & (~df['is_mut_outlier'])

    df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    # Recalibration examples
    naf_model = recalibrate_formula(df[df['pass_filter']], 'nafkah_iddah', 'husband_income')
    mut_model = recalibrate_formula(df[df['pass_filter']], 'mutaah', 'husband_income')
    print("NAF model:", naf_model)
    print("MUT model:", mut_model)
    return df, naf_model, mut_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='samples', help='folder with PDFs or txts')
    parser.add_argument('--out', default='cases_extracted.csv')
    parser.add_argument('--high_income', type=float, default=10000)
    args = parser.parse_args()
    main(args.input, args.out, args.high_income)
