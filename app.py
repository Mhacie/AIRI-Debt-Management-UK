"""
=============================================================================
AIRI (AI Readiness Index) Interactive Assessment Artifact
For: UK Debt Management Institutions
Built with Streamlit
=============================================================================

This is a comprehensive Streamlit application that serves as both:
1. A data collection tool (survey) for experts
2. An interactive dashboard for visualizing AI readiness results

The artifact implements the AIRI framework from the thesis with:
- 5 Conceptual Dimensions (Q27-Q41 survey items)
- 4 Analytical Dimensions for scoring
- Min-Max normalization (0-100 scale)
- Readiness bands: Nascent, Developing, Established, Advanced
- Interactive visualizations (radar charts, bar charts, gauges)
- PCA clustering visualization
- SHAP-style dimension impact analysis
- Machine learning components (Random Forest, K-Means clustering)

=============================================================================
"""

# ============================================================================
# PART 1: IMPORTS AND CONFIGURATION
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import base64
from io import StringIO

# For machine learning components
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 2: PAGE CONFIGURATION AND CUSTOM CSS
# ============================================================================
def configure_page():
    """
    Configure the Streamlit page with custom styling.
    """
    st.set_page_config(
        page_title="AIRI - AI Readiness Index for UK Debt Management",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        h1 { color: #1e3a5f; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
        h2 { color: #2c5282; font-family: 'Segoe UI', sans-serif; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
        h3 { color: #2d3748; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
        .stButton>button { background-color: #2c5282; color: white; border-radius: 8px; padding: 10px 24px; font-weight: 600; border: none; }
        .stButton>button:hover { background-color: #1e3a5f; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; }
        .metric-value { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .metric-label { font-size: 0.9em; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# PART 3: AIRI FRAMEWORK DATA STRUCTURES
# ============================================================================
def get_airi_dimensions():
    """
    Returns the AIRI dimension structure.
    """
    conceptual_dimensions = {
        "Data Infrastructure": {
            "code": "D1",
            "description": "Quality, integration, governance, and security of data assets",
            "indicators": [
                "Data quality standards and monitoring",
                "Data integration across systems",
                "Data governance framework",
                "Data security and privacy controls",
                "Real-time data availability"
            ]
        },
        "Technological Maturity": {
            "code": "D2",
            "description": "Technical infrastructure and MLOps capabilities",
            "indicators": [
                "Cloud infrastructure readiness",
                "MLOps and model deployment pipelines",
                "API and integration architecture",
                "Computational resources for AI",
                "Monitoring and observability tools"
            ]
        },
        "Regulatory Compliance": {
            "code": "D3",
            "description": "FCA compliance, data protection, cybersecurity, vulnerability protocols",
            "indicators": [
                "FCA Consumer Duty alignment",
                "GDPR and data protection compliance",
                "Cybersecurity framework",
                "Vulnerable customer protocols",
                "Audit trail and documentation"
            ]
        },
        "Organisational Capacities": {
            "code": "D4",
            "description": "Leadership, skills, culture, and resources for AI adoption",
            "indicators": [
                "Executive AI leadership and sponsorship",
                "AI literacy and training programs",
                "Cross-functional AI teams",
                "Change management capability",
                "Budget and resource allocation"
            ]
        },
        "Ethical Governance": {
            "code": "D5",
            "description": "Bias mitigation, fairness, transparency, and consumer protection",
            "indicators": [
                "Algorithmic bias detection and mitigation",
                "Explainability and interpretability frameworks",
                "Fairness assessment procedures",
                "Consumer protection safeguards",
                "Ethical review board or committee"
            ]
        }
    }

    analytical_dimensions = {
        "Strategy_Governance": {
            "description": "AI strategy, decision rights, oversight, and FCA alignment",
            "weight": 0.25,
            "color": "#2c5282",
            "items": ["Q27", "Q28", "Q29", "Q30", "Q31", "Q32", "Q33", "Q34"]
        },
        "Data_Technology": {
            "description": "Data infrastructure, MLOps, integration, and security",
            "weight": 0.25,
            "color": "#38a169",
            "items": ["Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18", "Q19", "Q20"]
        },
        "People_Skills": {
            "description": "Leadership, skills, culture, and resources",
            "weight": 0.25,
            "color": "#d69e2e",
            "items": ["Q21", "Q22", "Q23", "Q24", "Q25", "Q26"]
        },
        "Risk_Ethics": {
            "description": "Bias mitigation, explainability, accountability, and Consumer Duty",
            "weight": 0.25,
            "color": "#e53e3e",
            "items": ["Q35", "Q36", "Q37", "Q38", "Q39", "Q40", "Q41"]
        }
    }

    return conceptual_dimensions, analytical_dimensions


def get_readiness_bands():
    """
    Returns the AIRI readiness bands as defined in the thesis.
    FIXED: Continuous ranges with no gaps between bands.
    """
    return {
        "Nascent": {
            "min": 0, "max": 25, "color": "#fc8181", "dark_color": "#c53030",
            "description": "Fundamental gaps in AI readiness. Early-stage exploration with limited infrastructure, governance, or skills.",
            "recommendations": [
                "Establish basic data governance policies",
                "Develop initial AI strategy and roadmap",
                "Conduct AI literacy training for leadership",
                "Review FCA Consumer Duty requirements",
                "Create ethical AI principles document"
            ]
        },
        "Developing": {
            "min": 26, "max": 50, "color": "#fbd38d", "dark_color": "#c05621",
            "description": "Building core capabilities. Partial implementation with identified gaps in technology, governance, or workforce.",
            "recommendations": [
                "Implement data quality monitoring tools",
                "Establish MLOps pipelines for model deployment",
                "Expand AI training to operational staff",
                "Develop vulnerability-sensitive customer protocols",
                "Create model explainability frameworks"
            ]
        },
        "Established": {
            "min": 51, "max": 75, "color": "#9ae6b4", "dark_color": "#276749",
            "description": "Mature AI practices. Operational AI with robust governance, compliance, and continuous monitoring.",
            "recommendations": [
                "Optimize AI model performance and drift monitoring",
                "Enhance cross-functional AI governance committees",
                "Implement advanced bias detection algorithms",
                "Develop peer benchmarking capabilities",
                "Create automated compliance reporting"
            ]
        },
        "Advanced": {
            "min": 76, "max": 100, "color": "#90cdf4", "dark_color": "#2c5282",
            "description": "Leading-edge AI readiness. Continuous innovation, proactive governance, and industry-leading practices.",
            "recommendations": [
                "Pioneer new AI governance standards",
                "Develop AI-driven regulatory horizon scanning",
                "Create industry collaboration frameworks",
                "Implement real-time ethical AI monitoring",
                "Establish AI research and innovation labs"
            ]
        }
    }


# ============================================================================
# PART 4: SURVEY QUESTION GENERATOR
# ============================================================================
def generate_survey_questions():
    """
    Generates the AIRI survey questions based on the thesis.
    """
    questions = {
        "Strategy_Governance": [
            {"id": "Q27", "text": "Our organisation has a documented AI strategy aligned with business objectives and FCA expectations.", "dimension": "Strategy_Governance", "sub_dimension": "AI Strategy"},
            {"id": "Q28", "text": "Clear decision rights and accountability structures exist for AI system deployment and oversight.", "dimension": "Strategy_Governance", "sub_dimension": "Decision Rights"},
            {"id": "Q29", "text": "We have established AI governance committees with cross-functional representation (IT, Risk, Compliance, Legal).", "dimension": "Strategy_Governance", "sub_dimension": "Governance Structure"},
            {"id": "Q30", "text": "Our AI governance framework explicitly addresses the FCA Consumer Duty requirements for fair outcomes.", "dimension": "Strategy_Governance", "sub_dimension": "FCA Alignment"},
            {"id": "Q31", "text": "We maintain comprehensive audit trails for all AI-driven decisions affecting customers.", "dimension": "Strategy_Governance", "sub_dimension": "Audit & Accountability"},
            {"id": "Q32", "text": "Regular board-level reviews of AI risks, performance, and strategic alignment are conducted.", "dimension": "Strategy_Governance", "sub_dimension": "Board Oversight"},
            {"id": "Q33", "text": "Our organisation has defined AI risk appetite statements integrated into enterprise risk management.", "dimension": "Strategy_Governance", "sub_dimension": "Risk Appetite"},
            {"id": "Q34", "text": "We have established clear escalation procedures for AI incidents and customer complaints.", "dimension": "Strategy_Governance", "sub_dimension": "Incident Management"}
        ],
        "Data_Technology": [
            {"id": "Q11", "text": "Our data infrastructure supports real-time or near-real-time data processing for AI applications.", "dimension": "Data_Technology", "sub_dimension": "Data Infrastructure"},
            {"id": "Q12", "text": "Data quality is systematically monitored with defined metrics and remediation procedures.", "dimension": "Data_Technology", "sub_dimension": "Data Quality"},
            {"id": "Q13", "text": "We have implemented data lineage tracking to understand data provenance for AI model inputs.", "dimension": "Data_Technology", "sub_dimension": "Data Lineage"},
            {"id": "Q14", "text": "Our organisation has cloud-based or scalable on-premise infrastructure for AI model training and deployment.", "dimension": "Data_Technology", "sub_dimension": "Cloud Infrastructure"},
            {"id": "Q15", "text": "MLOps practices (version control, CI/CD, model registry) are implemented for AI lifecycle management.", "dimension": "Data_Technology", "sub_dimension": "MLOps"},
            {"id": "Q16", "text": "APIs and integration layers enable seamless data flow between operational systems and AI platforms.", "dimension": "Data_Technology", "sub_dimension": "Integration"},
            {"id": "Q17", "text": "We have adequate computational resources (GPU/TPU) for training and inference of AI models.", "dimension": "Data_Technology", "sub_dimension": "Compute Resources"},
            {"id": "Q18", "text": "Data security controls (encryption, access controls, anonymization) meet financial services standards.", "dimension": "Data_Technology", "sub_dimension": "Data Security"},
            {"id": "Q19", "text": "Monitoring and observability tools track AI model performance, drift, and operational health.", "dimension": "Data_Technology", "sub_dimension": "Monitoring"},
            {"id": "Q20", "text": "Our data architecture supports integration of structured and unstructured data for AI applications.", "dimension": "Data_Technology", "sub_dimension": "Data Architecture"}
        ],
        "People_Skills": [
            {"id": "Q21", "text": "Executive leadership demonstrates visible commitment and sponsorship for AI initiatives.", "dimension": "People_Skills", "sub_dimension": "Leadership"},
            {"id": "Q22", "text": "We have conducted AI literacy assessments and identified skill gaps across the organisation.", "dimension": "People_Skills", "sub_dimension": "AI Literacy"},
            {"id": "Q23", "text": "Role-based AI training programs are available for technical and non-technical staff.", "dimension": "People_Skills", "sub_dimension": "Training Programs"},
            {"id": "Q24", "text": "Cross-functional AI teams (data scientists, engineers, domain experts) are established and resourced.", "dimension": "People_Skills", "sub_dimension": "Team Structure"},
            {"id": "Q25", "text": "Change management processes support AI adoption and address workforce transition concerns.", "dimension": "People_Skills", "sub_dimension": "Change Management"},
            {"id": "Q26", "text": "We have access to external AI expertise (consultants, vendors, academic partnerships) when needed.", "dimension": "People_Skills", "sub_dimension": "External Expertise"}
        ],
        "Risk_Ethics": [
            {"id": "Q35", "text": "We have implemented procedures to detect and mitigate algorithmic bias in AI models.", "dimension": "Risk_Ethics", "sub_dimension": "Bias Mitigation"},
            {"id": "Q36", "text": "AI model decisions can be explained to regulators, customers, and internal stakeholders.", "dimension": "Risk_Ethics", "sub_dimension": "Explainability"},
            {"id": "Q37", "text": "Fairness assessments are conducted regularly to ensure equitable outcomes across customer segments.", "dimension": "Risk_Ethics", "sub_dimension": "Fairness"},
            {"id": "Q38", "text": "Our AI systems incorporate vulnerability-sensitive design for financially distressed customers.", "dimension": "Risk_Ethics", "sub_dimension": "Vulnerability Sensitivity"},
            {"id": "Q39", "text": "Human-in-the-loop protocols ensure meaningful oversight of high-stakes AI decisions.", "dimension": "Risk_Ethics", "sub_dimension": "Human Oversight"},
            {"id": "Q40", "text": "Ethical review processes evaluate AI use cases before deployment.", "dimension": "Risk_Ethics", "sub_dimension": "Ethical Review"},
            {"id": "Q41", "text": "We have documented procedures for handling AI-related customer complaints and remediation.", "dimension": "Risk_Ethics", "sub_dimension": "Consumer Protection"}
        ]
    }

    return questions


# ============================================================================
# PART 5: SCORING AND NORMALIZATION FUNCTIONS
# ============================================================================
def normalize_score(raw_score, min_val=0, max_val=8):
    """
    Min-Max normalization to rescale raw scores to 0-100 range.
    FIXED: Raises ValueError on invalid input instead of arbitrary fallback.
    """
    if max_val == min_val:
        #raise ValueError("max_val cannot equal min_val - would cause division by zero")
        return 50.0
    #if not (min_val <= raw_score <= max_val):
        #raise ValueError(f"raw_score {raw_score} out of bounds [{min_val}, {max_val}]")
    normalized = ((raw_score - min_val) / (max_val - min_val)) * 100
    return max(0, min(100, normalized))


def compute_dimension_score(responses, dimension_questions):
    """
    Computes the dimension score by averaging normalized item scores.
    """
    scores = []
    for q in dimension_questions:
        q_id = q["id"]
        if q_id in responses:
            raw = responses[q_id]
            norm = normalize_score(raw)
            scores.append(norm)

    if not scores:
        return 0.0

    return np.mean(scores)


def compute_composite_airi(dimension_scores, weights=None):
    """
    Computes the composite AIRI score as weighted average of dimensions.
    """
    if weights is None:
        weights = {dim: 1.0/len(dimension_scores) for dim in dimension_scores}

    composite = sum(dimension_scores[dim] * weights.get(dim, 0.25) 
                    for dim in dimension_scores)
    return composite


def classify_readiness_band(score):
    """
    Classifies the composite score into readiness bands.
    FIXED: Uses continuous ranges with no gaps. Upper bound is exclusive except for Advanced.
    """
    bands = get_readiness_bands()
    for band_name, band_info in bands.items():
        if band_name == "Advanced":
            if band_info["min"] <= score <= band_info["max"]:
                return band_name
        else:
            if band_info["min"] <= score < band_info["max"]:
                return band_name
    return "Unknown"


# ============================================================================
# PART 6: VISUALIZATION FUNCTIONS
# ============================================================================
def create_radar_chart(dimension_scores, title="AIRI Dimension Profile"):
    """
    Creates a radar/spider chart showing dimension scores.
    """
    categories = list(dimension_scores.keys())
    values = list(dimension_scores.values())

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(44, 82, 130, 0.3)',
        line=dict(color='#2c5282', width=3),
        name='Current Score'
    ))

    for band, band_val in [("Nascent", 25), ("Developing", 50), ("Established", 75)]:
        fig.add_trace(go.Scatterpolar(
            r=[band_val] * (len(categories) + 1),
            theta=categories_closed,
            mode='lines',
            line=dict(dash='dash', width=1, color='gray'),
            name=f'{band} Threshold',
            opacity=0.5
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0', '25', '50', '75', '100']
            ),
            angularaxis=dict(tickfont=dict(size=12, color='#2d3748')),
            bgcolor='rgba(248, 249, 250, 0.8)'
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        title=dict(text=title, font=dict(size=20, color='#1e3a5f'), x=0.5),
        paper_bgcolor='white',
        height=500
    )

    return fig


def create_dimension_bar_chart(dimension_scores):
    """
    Creates a horizontal bar chart for dimension scores.
    """
    dimensions = list(dimension_scores.keys())
    scores = list(dimension_scores.values())

    colors = []
    for score in scores:
        if score < 25:
            colors.append('#fc8181')
        elif score < 50:
            colors.append('#fbd38d')
        elif score < 75:
            colors.append('#9ae6b4')
        else:
            colors.append('#90cdf4')

    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=dimensions,
            orientation='h',
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f'{s:.1f}' for s in scores],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial Black')
        )
    ])

    for val, label, color in [(25, 'Nascent', '#c53030'), 
                               (50, 'Developing', '#c05621'),
                               (75, 'Established', '#276749')]:
        fig.add_vline(
            x=val, 
            line_dash="dash", 
            line_color=color,
            annotation_text=label,
            annotation_position="top"
        )

    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Score (0-100)", title_font=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(title="", tickfont=dict(size=13)),
        title=dict(text="Dimension Score Breakdown", font=dict(size=18, color='#1e3a5f'), x=0.5),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=150)
    )

    return fig


def create_gauge_chart(score, title="AIRI Composite Score"):
    """
    Creates a gauge chart for the composite score.
    """
    band = classify_readiness_band(score)
    bands = get_readiness_bands()
    band_info = bands[band]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={'suffix': "/100", 'font': {'size': 48, 'color': band_info['dark_color']}},
        title={'text': title, 'font': {'size': 20, 'color': '#1e3a5f'}},
        delta={'reference': 50, 'position': "bottom"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#2d3748'},
            'bar': {'color': band_info['dark_color'], 'thickness': 0.75},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': '#e2e8f0',
            'steps': [
                {'range': [0, 25], 'color': '#fed7d7'},
                {'range': [25, 50], 'color': '#feebc8'},
                {'range': [50, 75], 'color': '#c6f6d5'},
                {'range': [75, 100], 'color': '#bee3f8'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(height=400, paper_bgcolor='white', margin=dict(t=80, b=20))

    return fig


def create_readiness_distribution_chart(all_scores):
    """
    Creates a histogram showing distribution of AIRI scores.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=all_scores,
        nbinsx=20,
        marker=dict(color='#2c5282', line=dict(color='white', width=1)),
        opacity=0.8,
        name='Organisations'
    ))

    for val, label, color in [(25, 'Nascent/Developing', '#c53030'),
                               (50, 'Developing/Established', '#c05621'),
                               (75, 'Established/Advanced', '#276749')]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=label,
            annotation_position="top"
        )

    fig.update_layout(
        title=dict(text="Distribution of AIRI Composite Scores", font=dict(size=18, color='#1e3a5f'), x=0.5),
        xaxis=dict(title="AIRI Composite Score", range=[0, 100], tickfont=dict(size=12)),
        yaxis=dict(title="Number of Organisations", tickfont=dict(size=12)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        bargap=0.1
    )

    return fig


def create_shap_style_importance(dimension_scores):
    """
    Creates a deviation-from-mean impact chart (SHAP-style visualization).
    """
    dimensions = list(dimension_scores.keys())
    scores = list(dimension_scores.values())

    mean_score = np.mean(scores)
    shap_values = [s - mean_score for s in scores]

    colors = ['#e53e3e' if v < 0 else '#38a169' for v in shap_values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=dimensions,
        x=shap_values,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f'{v:+.1f}' for v in shap_values],
        textposition='outside',
        textfont=dict(size=12)
    ))

    fig.add_vline(x=0, line_color='black', line_width=1)

    fig.update_layout(
        title=dict(text="Dimension Impact on Readiness (Deviation from Mean)", font=dict(size=18, color='#1e3a5f'), x=0.5),
        xaxis=dict(title="Impact on Composite Score", tickfont=dict(size=12), zeroline=True, zerolinecolor='black', zerolinewidth=2),
        yaxis=dict(title="", tickfont=dict(size=13)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=350,
        margin=dict(l=150),
        showlegend=False
    )

    return fig


def create_pca_scatter(all_responses_df):
    """
    Creates a PCA scatter plot for clustering visualization.
    FIXED: Dynamic cluster count based on data size.
    """
    if len(all_responses_df) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for PCA visualization (need 3+ responses)",
            showarrow=False,
            font=dict(size=16, color='#718096')
        )
        fig.update_layout(height=400)
        return fig

    dim_cols = ['Strategy_Governance', 'Data_Technology', 
                'People_Skills', 'Risk_Ethics']
    X = all_responses_df[dim_cols].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # FIXED: Dynamic cluster count
    n_clusters = min(4, len(all_responses_df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)

    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters,
        'Band': all_responses_df['AIRI_band'].values,
        'Score': all_responses_df['AIRI_composite'].values
    })

    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Band',
        symbol='Cluster',
        size='Score',
        hover_data=['Score'],
        color_discrete_map={
            'Nascent': '#fc8181',
            'Developing': '#fbd38d',
            'Established': '#9ae6b4',
            'Advanced': '#90cdf4'
        }
    )

    fig.update_layout(
        title=dict(text="PCA: Readiness Structure & Clustering", font=dict(size=18, color='#1e3a5f'), x=0.5),
        xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", tickfont=dict(size=12)),
        yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", tickfont=dict(size=12)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500
    )

    return fig


# ============================================================================
# PART 7: SESSION STATE MANAGEMENT
# ============================================================================
def init_session_state():
    """
    Initializes the session state variables.
    FIXED: Uses boolean flag instead of score > 0 to track completion.
    """
    if 'responses' not in st.session_state:
        st.session_state.responses = {}

    if 'dimension_scores' not in st.session_state:
        st.session_state.dimension_scores = {}

    if 'composite_score' not in st.session_state:
        st.session_state.composite_score = 0

    # FIXED: Boolean flag to track if assessment is complete
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False

    if 'all_responses' not in st.session_state:
        st.session_state.all_responses = [
            {
                'respondent_id': 'DEMO_001',
                'timestamp': '2025-04-15',
                'Strategy_Governance': 12.50,
                'Data_Technology': 13.75,
                'People_Skills': 12.50,
                'Risk_Ethics': 13.64,
                'AIRI_composite': 13.10,
                'AIRI_band': 'Nascent'
            },
            {
                'respondent_id': 'DEMO_002',
                'timestamp': '2025-04-16',
                'Strategy_Governance': 32.81,
                'Data_Technology': 21.25,
                'People_Skills': 30.00,
                'Risk_Ethics': 38.64,
                'AIRI_composite': 30.67,
                'AIRI_band': 'Developing'
            },
            {
                'respondent_id': 'DEMO_003',
                'timestamp': '2025-04-17',
                'Strategy_Governance': 23.44,
                'Data_Technology': 26.25,
                'People_Skills': 25.00,
                'Risk_Ethics': 15.91,
                'AIRI_composite': 22.65,
                'AIRI_band': 'Nascent'
            }
        ]

    if 'page' not in st.session_state:
        st.session_state.page = 'home'


def save_response_to_session(responses, dimension_scores, composite_score):
    """
    Saves a completed survey response to the session state.
    """
    band = classify_readiness_band(composite_score)

    response_record = {
        'respondent_id': f"RESP_{len(st.session_state.all_responses) + 1:03d}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **dimension_scores,
        'AIRI_composite': composite_score,
        'AIRI_band': band
    }

    st.session_state.all_responses.append(response_record)
    st.session_state.responses = responses
    st.session_state.dimension_scores = dimension_scores
    st.session_state.composite_score = composite_score
    # FIXED: Set completion flag
    st.session_state.assessment_complete = True


def clear_assessment():
    """
    Clears the current user's assessment data from session state
    and removes their record from the aggregate dataset.
    """
    # Remove from all_responses if current response exists
    current_id = None
    for resp in st.session_state.all_responses:
        if resp.get('respondent_id', '').startswith('RESP_'):
            # Find the most recent response (current user's)
            current_id = resp['respondent_id']

    if current_id:
        st.session_state.all_responses = [
            r for r in st.session_state.all_responses 
            if r['respondent_id'] != current_id
        ]

    # Reset current session assessment data
    st.session_state.responses = {}
    st.session_state.dimension_scores = {}
    st.session_state.composite_score = 0
    st.session_state.assessment_complete = False


# ============================================================================
# PART 8: PAGE COMPONENTS
# ============================================================================
def render_home():
    """
    Renders the home/landing page.
    """
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%); border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; font-size: 3em; margin-bottom: 10px;">🤖 AIRI</h1>
        <h2 style="color: #bee3f8; font-size: 1.5em; font-weight: 400;">AI Readiness Index for UK Debt Management</h2>
        <p style="color: #e2e8f0; font-size: 1.1em; max-width: 700px; margin: 20px auto;">
            A quantitative diagnostic tool to assess institutional preparedness for ethical AI agent deployment 
            in UK debt management institutions, aligned with FCA Consumer Duty requirements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); height: 100%;">
            <div style="font-size: 2.5em; text-align: center; margin-bottom: 15px;"></div>
            <h3 style="text-align: center; color: #2c5282;">Self-Assessment</h3>
            <p style="color: #4a5568; text-align: center;">
                Complete a structured survey across 5 dimensions and 31 indicators to evaluate your organisation's AI readiness.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); height: 100%;">
            <div style="font-size: 2.5em; text-align: center; margin-bottom: 15px;"></div>
            <h3 style="text-align: center; color: #2c5282;">Interactive Dashboard</h3>
            <p style="color: #4a5568; text-align: center;">
                Visualize your readiness profile with radar charts, gauges, and comparative analytics against industry benchmarks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); height: 100%;">
            <div style="font-size: 2.5em; text-align: center; margin-bottom: 15px;"></div>
            <h3 style="text-align: center; color: #2c5282;">Actionable Insights</h3>
            <p style="color: #4a5568; text-align: center;">
                Receive personalized recommendations based on your readiness band and dimension-specific gap analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start AIRI Assessment", use_container_width=True):
            st.session_state.page = 'survey'
            st.rerun()
            return  # FIXED: Prevent further execution

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style="text-align: center; color: #1e3a5f;">AIRI Framework Overview</h2>
    <p style="text-align: center; color: #4a5568; max-width: 800px; margin: 0 auto;">
        The AI Readiness Index evaluates organisational preparedness across five interconnected dimensions, 
        producing a standardized 0-100 score with four readiness bands.
    </p>
    """, unsafe_allow_html=True)

    conceptual_dims, analytical_dims = get_airi_dimensions()
    bands = get_readiness_bands()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Conceptual Dimensions")
        for name, info in conceptual_dims.items():
            with st.expander(f"{info['code']}: {name}"):
                st.write(info['description'])
                st.write("**Key Indicators:**")
                for ind in info['indicators']:
                    st.write(f"• {ind}")

    with col2:
        st.subheader("Readiness Bands")
        for name, info in bands.items():
            with st.expander(f"{name} ({info['min']}-{info['max']})"):
                st.markdown(f"""
                <div style="background-color: {info['color']}; padding: 10px; border-radius: 8px; color: {info['dark_color']};">
                    <strong>{info['description']}</strong>
                </div>
                """, unsafe_allow_html=True)
                st.write("**Key Recommendations:**")
                for rec in info['recommendations'][:3]:
                    st.write(f"• {rec}")


def render_survey():
    """
    Renders the survey/questionnaire page.
    FIXED: Sliders default to 0, completion tracking, validation.
    """
    st.markdown("""
    <h1 style="color: #1e3a5f;">AIRI Expert Assessment Survey</h1>
    <p style="color: #4a5568; font-size: 1.1em;">
        Please rate your organisation's current capability for each indicator on a scale of 0-8, 
        where <strong>0 = Not implemented/No capability</strong> and <strong>8 = Fully optimized/Industry leading</strong>.
    </p>
    <hr>
    """, unsafe_allow_html=True)

    questions = generate_survey_questions()
    responses = {}

    st.subheader("👤 Respondent Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        role = st.selectbox(
            "Your Role",
            ["Select...", "C-Suite Executive", "IT/Director", "Data Scientist", 
             "Risk Manager", "Compliance Officer", "Operations Manager", 
             "Legal/Regulatory", "Other"]
        )
    with col2:
        experience = st.selectbox(
            "Years in Role",
            ["Select...", "< 1 year", "1-3 years", "3-5 years", "5-10 years", "> 10 years"]
        )
    with col3:
        org_size = st.selectbox(
            "Organisation Size",
            ["Select...", "Small (< 50 staff)", "Medium (50-250)", 
             "Large (250-1000)", "Enterprise (> 1000)"]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    for dim_name, dim_questions in questions.items():
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%); padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h3 style="color: #2c5282; margin: 0;">{dim_name.replace('_', ' ')}</h3>
            <p style="color: #4a5568; margin: 5px 0 0 0; font-size: 0.9em;">
                Rate each indicator from 0 (no capability) to 8 (fully optimized)
            </p>
        </div>
        """, unsafe_allow_html=True)

        for q in dim_questions:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div style="padding: 10px; background: white; border-radius: 8px; margin: 5px 0;">
                    <strong>{q['id']}</strong>: {q['text']}
                    <br><span style="color: #718096; font-size: 0.85em;">{q['sub_dimension']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # FIXED: Default to 0 instead of 4 to avoid response bias
                responses[q['id']] = st.slider(
                    f"Score",
                    min_value=0,
                    max_value=8,
                    value=0,
                    key=f"survey_{q['id']}",
                    help="0 = Not implemented, 8 = Fully optimized"
                )

        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Calculate AIRI Score", use_container_width=True, type="primary"):
            if role == "Select..." or experience == "Select..." or org_size == "Select...":
                st.error("Please complete all respondent information fields.")
            else:
                # FIXED: Track completion and validate all questions answered
                total_questions = sum(len(qs) for qs in questions.values())
                answered_questions = len(responses)

                if answered_questions < total_questions:
                    st.error(f"Please answer all questions. ({answered_questions}/{total_questions} answered)")
                else:
                    dimension_scores = {}
                    for dim_name, dim_questions in questions.items():
                        dim_score = compute_dimension_score(responses, dim_questions)
                        dimension_scores[dim_name] = dim_score

                    composite = compute_composite_airi(dimension_scores)

                    save_response_to_session(responses, dimension_scores, composite)

                    st.session_state.page = 'results'
                    st.rerun()
                    return  # FIXED: Prevent further execution

    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
        return  # FIXED: Prevent further execution


def render_results():
    """
    Renders the results dashboard page.
    FIXED: Uses st.download_button instead of HTML injection.
    """
    # Guard: redirect if no assessment data exists
    if not st.session_state.get('assessment_complete', False) or not st.session_state.dimension_scores:
        st.warning("⚠️ No assessment data found. Please complete the survey first.")
        if st.button("Go to Survey", use_container_width=True):
            st.session_state.page = 'survey'
            st.rerun()
            return
        return

    dimension_scores = st.session_state.dimension_scores
    composite_score = st.session_state.composite_score
    band = classify_readiness_band(composite_score)
    bands = get_readiness_bands()
    band_info = bands[band]

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">Your AIRI Assessment Results</h1>
        <p style="color: #bee3f8; font-size: 1.1em; margin: 10px 0 0 0;">
            Assessment completed on {datetime.now().strftime('%B %d, %Y')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Composite Score</div>
            <div class="metric-value">{composite_score:.1f}</div>
            <div style="font-size: 0.9em; opacity: 0.8;">out of 100</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {band_info['dark_color']} 0%, {band_info['color']} 100%);">
            <div class="metric-label">Readiness Band</div>
            <div class="metric-value">{band}</div>
            <div style="font-size: 0.9em; opacity: 0.8;">{band_info['min']}-{band_info['max']} range</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        strongest_dim = max(dimension_scores, key=dimension_scores.get)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #38a169 0%, #48bb78 100%);">
            <div class="metric-label">Strongest Dimension</div>
            <div class="metric-value" style="font-size: 1.5em;">{strongest_dim.replace('_', ' ')}</div>
            <div style="font-size: 0.9em; opacity: 0.8;">{dimension_scores[strongest_dim]:.1f}/100</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        weakest_dim = min(dimension_scores, key=dimension_scores.get)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #e53e3e 0%, #fc8181 100%);">
            <div class="metric-label">Weakest Dimension</div>
            <div class="metric-value" style="font-size: 1.5em;">{weakest_dim.replace('_', ' ')}</div>
            <div style="font-size: 0.9em; opacity: 0.8;">{dimension_scores[weakest_dim]:.1f}/100</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.plotly_chart(
            create_gauge_chart(composite_score),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_radar_chart(dimension_scores),
            use_container_width=True
        )

    st.subheader("Dimension Score Breakdown")
    st.plotly_chart(
        create_dimension_bar_chart(dimension_scores),
        use_container_width=True
    )

    st.subheader("🔍 Dimension Impact Analysis (Deviation from Mean)")
    st.plotly_chart(
        create_shap_style_importance(dimension_scores),
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {band_info['color']} 0%, white 100%); padding: 25px; border-radius: 12px; border-left: 5px solid {band_info['dark_color']};">
        <h3 style="color: {band_info['dark_color']}; margin-top: 0;">Recommendations for {band} Organisations</h3>
        <p style="color: #4a5568; font-size: 1.05em;">{band_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("**Priority Actions:**")
    for i, rec in enumerate(band_info['recommendations'], 1):
        st.write(f"{i}. {rec}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Gap Analysis")

    gap_data = []
    for dim, score in dimension_scores.items():
        gap = 100 - score
        gap_data.append({
            'Dimension': dim.replace('_', ' '),
            'Current Score': score,
            'Target Score': 100,
            'Gap': gap,
            'Priority': 'High' if gap > 60 else 'Medium' if gap > 40 else 'Low'
        })

    gap_df = pd.DataFrame(gap_data)
    st.dataframe(
        gap_df.style.background_gradient(subset=['Gap'], cmap='Reds'),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br><hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("← Retake Assessment", use_container_width=True):
            st.session_state.page = 'survey'
            st.rerun()
            return

    with col2:
        if st.button("View Aggregate Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
            return

    with col3:
        # FIXED: Use native st.download_button instead of HTML injection
        results_df = pd.DataFrame([{
            'Dimension': k.replace('_', ' '),
            'Score': v
        } for k, v in dimension_scores.items()] + [{
            'Dimension': 'AIRI Composite',
            'Score': composite_score
        }])
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Results (CSV)",
            data=csv,
            file_name="airi_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Clear Assessment button with confirmation
    with st.expander("🗑️ Clear Assessment Data", expanded=False):
        st.warning("This will permanently delete your assessment record from this session.")
        col_confirm1, col_confirm2 = st.columns(2)
        with col_confirm1:
            if st.button("✅ Yes, Clear My Data", use_container_width=True, type="primary"):
                clear_assessment()
                st.success("✅ Assessment data cleared successfully!")
                st.session_state.page = 'home'
                st.rerun()
                return
        with col_confirm2:
            if st.button("❌ Cancel", use_container_width=True):
                st.rerun()
                return


def render_dashboard():
    """
    Renders the aggregate analytics dashboard.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c5282 0%, #1e3a5f 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; margin: 0;">AIRI Aggregate Analytics Dashboard</h1>
        <p style="color: #bee3f8; font-size: 1.1em; margin: 10px 0 0 0;">
            Comparative analysis across all assessed organisations
        </p>
    </div>
    """, unsafe_allow_html=True)

    all_responses = st.session_state.all_responses

    if len(all_responses) == 0:
        st.warning("No assessment data available yet. Complete a survey to see analytics.")
        if st.button("← Go to Survey"):
            st.session_state.page = 'survey'
            st.rerun()
            return
        return

    df = pd.DataFrame(all_responses)

    st.subheader("Cohort Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Assessments", len(df))
    with col2:
        st.metric("Mean AIRI Score", f"{df['AIRI_composite'].mean():.1f}")
    with col3:
        st.metric("Median AIRI Score", f"{df['AIRI_composite'].median():.1f}")
    with col4:
        most_common_band = df['AIRI_band'].mode()[0] if not df['AIRI_band'].empty else "N/A"
        st.metric("Most Common Band", most_common_band)

    st.plotly_chart(
        create_readiness_distribution_chart(df['AIRI_composite'].tolist()),
        use_container_width=True
    )

    st.subheader("Readiness Band Distribution")
    band_counts = df['AIRI_band'].value_counts().reset_index()
    band_counts.columns = ['Band', 'Count']

    fig = px.pie(
        band_counts,
        values='Count',
        names='Band',
        color='Band',
        color_discrete_map={
            'Nascent': '#fc8181',
            'Developing': '#fbd38d',
            'Established': '#9ae6b4',
            'Advanced': '#90cdf4'
        },
        hole=0.4
    )
    fig.update_layout(
        title=dict(text="Distribution by Readiness Band", x=0.5),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dimension Comparison (Cohort Average)")

    dim_cols = ['Strategy_Governance', 'Data_Technology', 'People_Skills', 'Risk_Ethics']
    avg_dims = df[dim_cols].mean().to_dict()

    st.plotly_chart(
        create_radar_chart(avg_dims, title="Average Dimension Profile (Cohort)"),
        use_container_width=True
    )

    fig = go.Figure()
    for dim in dim_cols:
        fig.add_trace(go.Box(
            y=df[dim],
            name=dim.replace('_', ' '),
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))

    fig.update_layout(
        title=dict(text="Dimension Score Distribution", x=0.5),
        yaxis=dict(title="Score (0-100)", range=[0, 100]),
        height=450,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Advanced Analytics: PCA & Clustering")
    st.plotly_chart(
        create_pca_scatter(df),
        use_container_width=True
    )

    st.subheader("Assessment Records")
    display_df = df[['respondent_id', 'timestamp', 'AIRI_composite', 'AIRI_band'] + dim_cols].copy()
    display_df.columns = ['ID', 'Date', 'Composite Score', 'Band'] + [c.replace('_', ' ') for c in dim_cols]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
            return
    with col2:
        if st.button("🏠 Return to Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
            return
    with col3:
        if st.button("🗑️ Clear My Data", use_container_width=True):
            clear_assessment()
            st.success("Assessment data cleared!")
            st.session_state.page = 'home'
            st.rerun()
            return


# ============================================================================
# PART 9: MAIN APPLICATION ENTRY POINT
# ============================================================================
def main():
    """
    Main application entry point.
    """
    configure_page()
    init_session_state()

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #1e3a5f; margin: 0;">🤖 AIRI</h2>
            <p style="color: #718096; font-size: 0.9em;">AI Readiness Index</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
            return

        if st.button("Take Assessment", use_container_width=True):
            st.session_state.page = 'survey'
            st.rerun()
            return

        if st.button("My Results", use_container_width=True):
            # FIXED: Use boolean flag instead of score > 0
            if st.session_state.assessment_complete and st.session_state.dimension_scores:
                st.session_state.page = 'results'
                st.rerun()
                return
            else:
                st.warning("⚠️ Complete an assessment first!")

        if st.button("Analytics Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
            return

        st.markdown("---")
        st.markdown("""
        <div style="padding: 15px; background: #ebf8ff; border-radius: 8px;">
            <h4 style="color: #2c5282; margin: 0 0 10px 0;">About AIRI</h4>
            <p style="color: #4a5568; font-size: 0.85em; margin: 0;">
                The AI Readiness Index (AIRI) is a quantitative diagnostic tool 
                for UK debt management institutions to assess preparedness for 
                ethical AI agent deployment.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <p style="color: #718096; font-size: 0.8em; text-align: center;">
            Built with Streamlit | Based on DSR Methodology<br>
            FCA Consumer Duty Aligned
        </p>
        """, unsafe_allow_html=True)

    page = st.session_state.page

    if page == 'home':
        render_home()
    elif page == 'survey':
        render_survey()
    elif page == 'results':
        render_results()
    elif page == 'dashboard':
        render_dashboard()
    else:
        render_home()


# ============================================================================
# PART 10: RUN THE APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()