import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Tox21 AI Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1f77b4; font-family: 'Helvetica Neue', sans-serif; }
    h3 { color: #ff4b4b; }
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS (Cached for speed)
# ==========================================
@st.cache_resource
def load_models():
    try:
        # Update this path if your joblib file is named differently
        return joblib.load('tox21_xgboost_advanced_models.joblib')
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure 'tox21_xgboost_advanced_models.joblib' is in the same folder.")
        return None

models = load_models()

target_cols = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
               'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# ==========================================
# 3. FEATURE EXTRACTION PIPELINE (Fixed for 2051 features)
# ==========================================
# Initialize modern fingerprint generator
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_ultimate_features(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return None, None
    
    # Extract 2048-bit Fingerprint
    fp = fp_gen.GetFingerprintAsNumPy(mol)
    
    # Extract the EXACT 3 descriptors your loaded model expects
    mol_wt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Combine them (2048 + 3 = exactly 2051 features)
    features = np.append(fp, [mol_wt, logp, tpsa])
    
    # Store properties for UI display
    props = {
        "Molecular Weight": round(mol_wt, 2),
        "LogP (Lipophilicity)": round(logp, 2),
        "TPSA (Polar Surface)": round(tpsa, 2),
        "Ring Count": Descriptors.RingCount(mol) # Still calculate for UI, but don't send to model
    }
    return features.reshape(1, -1), props

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.title("🧪 Tox21 AI Toxicity Predictor")
st.markdown("""
    Welcome to the **Tox21 Advanced Machine Learning Dashboard**. 
    Enter a chemical SMILES string below to calculate its physicochemical properties and predict its risk across 12 different biological pathways using our fine-tuned XGBoost ensemble.
""")

# Sidebar for Input
with st.sidebar:
    st.header("🔬 Input Molecule")
    st.markdown("Enter a **SMILES** string:")
    
    # Default is Bisphenol A (BPA) - a known endocrine disruptor
    smiles_input = st.text_input("SMILES", value="CC(C)(C1=CC=C(O)C=C1)C2=CC=C(O)C=C2")
    
    analyze_btn = st.button("Analyze Toxicity", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Example Molecules:**")
    st.code("Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O", language="text")
    st.code("Caffeine: CN1C=NC2=C1C(=O)N(C(=O)N2C)C", language="text")
    st.code("DDT (Toxic): ClC(Cl)(Cl)C(c1ccc(Cl)cc1)c2ccc(Cl)cc2", language="text")

# ==========================================
# 5. PREDICTION & VISUALIZATION
# ==========================================
if analyze_btn and models:
    features, props = get_ultimate_features(smiles_input)
    
    if features is None:
        st.error("Invalid SMILES string. Please check your chemical structure and try again.")
    else:
        st.success("Molecule parsed successfully!")
        
        # --- Top Row: Chemical Properties ---
        st.subheader("📊 Physicochemical Properties")
        cols = st.columns(4)
        cols[0].metric(label="Molecular Weight", value=f"{props['Molecular Weight']} g/mol")
        cols[1].metric(label="LogP", value=props['LogP (Lipophilicity)'])
        cols[2].metric(label="TPSA", value=f"{props['TPSA (Polar Surface)']} Å²")
        cols[3].metric(label="Rings", value=props['Ring Count'])
        
        st.markdown("---")
        
        # --- Make Predictions ---
        st.subheader("⚠️ Toxicity Pathway Predictions")
        
        results = []
        for target in target_cols:
            # Predict Probability of Class 1 (Toxic)
            prob = models[target].predict_proba(features)[0][1] 
            is_toxic = "High Risk" if prob >= 0.5 else "Low Risk"
            results.append({"Assay": target, "Toxicity Probability": prob * 100, "Status": is_toxic})
            
        df_results = pd.DataFrame(results)
        
        # --- Display Results ---
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            # Dynamic Bar Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color bars red if > 50%, else blue
            colors = ['#ff4b4b' if val >= 50 else '#1f77b4' for val in df_results['Toxicity Probability']]
            
            sns.barplot(data=df_results, x='Toxicity Probability', y='Assay', palette=colors, ax=ax)
            
            ax.axvline(50, color='gray', linestyle='--', linewidth=2) # 50% Threshold line
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability of Toxicity (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Biological Target', fontsize=12, fontweight='bold')
            ax.set_title('Risk Profile across Tox21 Pathways', fontsize=14, pad=15)
            
            st.pyplot(fig)
            
        with col2:
            # Show high-risk warnings
            high_risk_assays = df_results[df_results['Toxicity Probability'] >= 50]
            
            if len(high_risk_assays) > 0:
                st.error(f"🚨 **Alert:** Flagged for {len(high_risk_assays)} toxic pathways!")
                for index, row in high_risk_assays.iterrows():
                    st.warning(f"**{row['Assay']}**: {row['Toxicity Probability']:.1f}% risk")
            else:
                st.success("✅ **Safe:** No significant toxicity detected across the 12 pathways.")
            
            # Show raw data table
            st.markdown("**Raw Data**")
            st.dataframe(df_results.style.format({'Toxicity Probability': '{:.1f}%'}))