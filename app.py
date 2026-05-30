
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

from scipy.optimize import differential_evolution
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Advanced Process Control",
    page_icon="⚙️",
    layout="wide"
)

# =====================================================
# ANN ARCHITECTURE
# =====================================================

class ANNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(6,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,2)

        )

    def forward(self,x):

        return self.network(x)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

   # rf_model = joblib.load("rf_model.pkl")

    xgb_model = joblib.load("xgb_model.pkl")

    scaler = joblib.load("scaler.pkl")

    ann_model = ANNModel()

    ann_model.load_state_dict(
        torch.load(
            "ann_model.pth",
            map_location="cpu"
        )
    )

    ann_model.eval()

    return xgb_model,ann_model,scaler

xgb_model,ann_model,scaler = load_models()

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    return pd.read_excel(
        "dummy_plant_data.xlsx"
    )

df = load_data()

feature_names = [

    "Reactor_Pressure",
    "Feed_Temperature",
    "Feed_Rate",
    "Reactor_Temperature",
    "Hydrogen_Flow",
    "Catalyst_Loading"

]

# =====================================================
# ENSEMBLE PREDICTION
# =====================================================

def ensemble_predict(input_vector):

    X_df = pd.DataFrame(
        [input_vector],
        columns=feature_names
    )

    

    xgb_pred = xgb_model.predict(
        X_df
    )[0]

    scaled = scaler.transform(
        X_df
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32
    )

    with torch.no_grad():

        ann_pred = ann_model(
            tensor
        ).numpy()[0]

    preds = np.array([
        
        xgb_pred,
        ann_pred
    ])

    mean_pred = preds.mean(axis=0)

    std_pred = preds.std(axis=0)

    return mean_pred,std_pred,preds

# =====================================================
# OPTIMIZER
# =====================================================

def optimize_process(

    pressure,
    feed_temp,
    feed_rate,
    target_mfi,
    target_yield

):

    def objective(x):

        reactor_temp = x[0]
        hydrogen = x[1]
        catalyst = x[2]

        input_vector = [

            pressure,
            feed_temp,
            feed_rate,

            reactor_temp,
            hydrogen,
            catalyst

        ]

        pred,std,_ = ensemble_predict(
            input_vector
        )

        loss = (

            (pred[0]-target_mfi)**2

            +

            (pred[1]-target_yield)**2

            +

            0.5*np.sum(std)

        )

        return loss

    bounds = [

        (200,260),
        (10,60),
        (0.5,3.0)

    ]

    result = differential_evolution(

        objective,

        bounds,

        maxiter=50,

        popsize=10,

        seed=42

    )

    return result

# =====================================================
# HEADER
# =====================================================

st.title(
    "⚙️ Advanced Process Control Dashboard"
)

st.markdown(
    "ANN + XGBoost Ensemble"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "Current Plant Conditions"
)

pressure = st.sidebar.number_input(

    "Reactor Pressure",

    value=22.0

)

feed_temp = st.sidebar.number_input(

    "Feed Temperature",

    value=75.0

)

feed_rate = st.sidebar.number_input(

    "Feed Rate",

    value=120.0

)

st.sidebar.markdown("---")

st.sidebar.header(
    "Target Product Properties"
)

target_mfi = st.sidebar.number_input(

    "Target MFI",

    value=35.0

)

target_yield = st.sidebar.number_input(

    "Target Yield",

    value=90.0

)

run_button = st.sidebar.button(
    "Optimize Process"
)

# =====================================================
# TABS
# =====================================================

tab1,tab2,tab3 = st.tabs(

    [

        "Historical Data",

        "Optimizer",

        "Model Insights"

    ]

)

# =====================================================
# HISTORICAL DATA
# =====================================================

with tab1:

    st.subheader(
        "Historical Plant Data"
    )

    st.dataframe(df.head(50))

    col1,col2 = st.columns(2)

    with col1:

        fig = px.scatter(

            df,

            x="Hydrogen_Flow",

            y="MFI",

            title="Hydrogen Flow vs MFI"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(

            df,

            x="Catalyst_Loading",

            y="Yield",

            title="Catalyst Loading vs Yield"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    fig3d = px.scatter_3d(

        df,

        x="Reactor_Temperature",

        y="Hydrogen_Flow",

        z="Catalyst_Loading",

        color="MFI",

        title="Operating Window"

    )

    st.plotly_chart(
        fig3d,
        use_container_width=True
    )

# =====================================================
# OPTIMIZER
# =====================================================

with tab2:

    st.subheader(
        "APC Optimizer"
    )

    if run_button:

        with st.spinner(
            "Running optimization..."
        ):

            result = optimize_process(

                pressure,

                feed_temp,

                feed_rate,

                target_mfi,

                target_yield

            )

        best_temp = result.x[0]

        best_h2 = result.x[1]

        best_cat = result.x[2]

        optimal_input = [

            pressure,

            feed_temp,

            feed_rate,

            best_temp,

            best_h2,

            best_cat

        ]

        pred,std,preds = ensemble_predict(
            optimal_input
        )

        col1,col2 = st.columns(2)

        with col1:

            st.subheader(
                "Recommended Setpoints"
            )

            st.metric(

                "Reactor Temperature",

                f"{best_temp:.2f}"

            )

            st.metric(

                "Hydrogen Flow",

                f"{best_h2:.2f}"

            )

            st.metric(

                "Catalyst Loading",

                f"{best_cat:.2f}"

            )

        with col2:

            st.subheader(
                "Predicted Quality"
            )

            st.metric(

                "Predicted MFI",

                f"{pred[0]:.2f}"

            )

            st.metric(

                "Predicted Yield",

                f"{pred[1]:.2f}"

            )

        comparison = pd.DataFrame({

            "Variable":[

                "Reactor Temperature",

                "Hydrogen Flow",

                "Catalyst Loading"

            ],

            "Recommended":[

                best_temp,

                best_h2,

                best_cat

            ]

        })

        st.dataframe(
            comparison,
            use_container_width=True
        )

        confidence = np.exp(
            -np.mean(std)
        )*100

        st.subheader(
            "Model Confidence"
        )

        st.progress(
            int(confidence)
        )

        st.write(
            f"{confidence:.1f}%"
        )

# =====================================================
# MODEL INSIGHTS
# =====================================================

with tab3:

    st.subheader(
        "Model Agreement"
    )

    sample = df.iloc[0]

    sample_input = [

        sample["Reactor_Pressure"],
        sample["Feed_Temperature"],
        sample["Feed_Rate"],
        sample["Reactor_Temperature"],
        sample["Hydrogen_Flow"],
        sample["Catalyst_Loading"]

    ]

    pred,std,preds = ensemble_predict(
        sample_input
    )

    agreement = pd.DataFrame({

        "Model":[


            "XGBoost",

            "ANN"

        ],

        "MFI":[

            preds[0][0],
            preds[1][0],
         

        ],

        "Yield":[

            preds[0][1],
            preds[1][1],
          

        ]

    })

    st.dataframe(
        agreement,
        use_container_width=True
    )

    fig1 = px.bar(

        agreement,

        x="Model",

        y="MFI",

        title="MFI Prediction Comparison"

    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.bar(

        agreement,

        x="Model",

        y="Yield",

        title="Yield Prediction Comparison"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.markdown("---")

st.caption(
    "Advanced Process Control Dashboard | Ensemble ML Optimizer"
)
