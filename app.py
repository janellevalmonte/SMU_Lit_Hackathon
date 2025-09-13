import streamlit as st
import pandas as pd
import io
from ebantu_pipeline import process_file, recalibrate_formula  # assume same folder or install as module

st.title("eBantu: Quick Case Extractor")

uploaded = st.file_uploader("Upload judgment PDF or text", accept_multiple_files=True)
if uploaded:
    rows = []
    for f in uploaded:
        b = f.read()
        # write a temp file
        tmp = f"/tmp/{f.name}"
        with open(tmp, 'wb') as fp:
            fp.write(b)
        rows.append(process_file(tmp))
    df = pd.DataFrame(rows)
    st.write("### Extracted cases")
    st.dataframe(df[['source_file','husband_income','nafkah_iddah','mutaah','is_consent_order']])
    # filter controls
    high_income = st.number_input("High income threshold", value=10000)
    df['is_high_income'] = df['husband_income'].apply(lambda x: x is not None and x > high_income)
    df['pass_filter'] = (~df['is_consent_order']) & (~df['is_high_income'])
    st.write("### Filtered cases")
    st.dataframe(df[df['pass_filter']])
    # recalibration
    if st.button("Recalibrate formulas (linear)"):
        naf_model = recalibrate_formula(df[df['pass_filter']], 'nafkah_iddah', 'husband_income')
        mut_model = recalibrate_formula(df[df['pass_filter']], 'mutaah', 'husband_income')
        st.write("NAF model:", naf_model)
        st.write("MUT model:", mut_model)
