import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Tox21 AI Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Responsive CSS (Fixes the invisible text issue)
st.markdown("""
    <style>
    /* Clean up the main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style the metric boxes to look like modern cards */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    
    /* Add a slight hover effect to the metric cards */
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    
    /* Make the title pop */
    h1 {
        background: -webkit-linear-gradient(45deg, #1f77b4, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS (Cached for speed)
# ==========================================
@st.cache_resource
def load_models():
    try:
        return joblib.load('tox21_xgboost_advanced_models.joblib')
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the file is in the same folder.")
        return None

models = load_models()

target_cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
               'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# ==========================================
# 3. FEATURE EXTRACTION PIPELINE
# ==========================================
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_ultimate_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return None, None
    
    fp = fp_gen.GetFingerprintAsNumPy(mol)
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    features = np.append(fp, [mol_wt, logp, tpsa])
    
    props = {
        "Molecular Weight": round(mol_wt, 2),
        "LogP (Lipophilicity)": round(logp, 2),
        "TPSA (Polar Surface)": round(tpsa, 2),
        "Ring Count": Descriptors.RingCount(mol)
    }
    return features.reshape(1, -1), props

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.title("🧪 Toxicity Predictor AI")
st.markdown("""
    Welcome to the **Advanced Machine Learning Dashboard**. 
    Enter a chemical SMILES string below to calculate its physicochemical properties and predict its risk across 12 different biological pathways using our fine-tuned XGBoost ensemble.
""")

# Sidebar for Input
with st.sidebar:
    st.header("🔬 Input Molecule")
    st.markdown("Enter a **SMILES** string:")
    
    smiles_input = st.text_input("SMILES Data", value="CC(C)(C1=CC=C(O)C=C1)C2=CC=C(O)C=C2", label_visibility="collapsed")
    
    analyze_btn = st.button("Analyze Toxicity", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Example Molecules:**")
    st.info("Aspirin:\n`CC(=O)OC1=CC=CC=C1C(=O)O`")
    st.info("Caffeine:\n`CN1C=NC2=C1C(=O)N(C(=O)N2C)C`")
    st.error("DDT (Highly Toxic):\n`ClC(Cl)(Cl)C(c1ccc(Cl)cc1)c2ccc(Cl)cc2`")

# ==========================================
# 5. PREDICTION & VISUALIZATION
# ==========================================
if analyze_btn and models:
    features, props = get_ultimate_features(smiles_input)
    
    if features is None:
        st.error("Invalid SMILES string. Please check your chemical structure and try again.")
    else:
        # --- Top Row: Chemical Properties ---
        st.subheader("📊 Physicochemical Properties")
        cols = st.columns(4)
        cols[0].metric(label="Molecular Weight", value=f"{props['Molecular Weight']} g/mol")
        cols[1].metric(label="LogP", value=props['LogP (Lipophilicity)'])
        cols[2].metric(label="TPSA", value=f"{props['TPSA (Polar Surface)']} Å²")
        cols[3].metric(label="Rings", value=props['Ring Count'])
        
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        
        # --- Make Predictions ---
        st.subheader("⚠️ Toxicity Pathway Predictions")
        
        results = []
        for target in target_cols:
            prob = models[target].predict_proba(features)[0][1] * 100
            risk_category = "High Risk" if prob >= 50 else "Low Risk"
            results.append({
                "Assay": target, 
                "Risk (%)": round(prob, 2), 
                "Status": risk_category
            })
            
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values("Risk (%)", ascending=True) # Sort for better visual flow
        
        # --- Display Results ---
        col1, col2 = st.columns([1.8, 1])
        
        with col1:
            # INTERACTIVE PLOTLY CHART
            fig = px.bar(
                df_results, 
                x='Risk (%)', 
                y='Assay', 
                color='Status',
                color_discrete_map={"High Risk": "#ff4b4b", "Low Risk": "#1f77b4"},
                orientation='h',
                text='Risk (%)'
            )
            
            fig.update_layout(
                xaxis_range=[0, 100],
                xaxis_title="Probability of Toxicity (%)",
                yaxis_title="",
                legend_title=None,
                margin=dict(l=0, r=0, t=30, b=0),
                height=500,
                paper_bgcolor="rgba(0,0,0,0)", # Transparent background
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            # Add the 50% threshold line
            fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50% Threshold", annotation_position="top right")
            fig.update_traces(textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### **Analysis Summary**")
            high_risk_assays = df_results[df_results['Status'] == "High Risk"]
            
            if len(high_risk_assays) > 0:
                st.error(f"🚨 **Alert:** Flagged for {len(high_risk_assays)} toxic pathways.")
                for index, row in high_risk_assays.iterrows():
                    st.warning(f"**{row['Assay']}**: {row['Risk (%)']}%")
            else:
                st.success("✅ **Safe:** No significant toxicity detected across the 12 pathways.")
            
            with st.expander("View Raw Data Table"):
                st.dataframe(df_results.style.format({'Risk (%)': '{:.2f}%'}), use_container_width=True)