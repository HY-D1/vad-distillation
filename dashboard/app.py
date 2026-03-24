"""
Interactive Training Dashboard for VAD Distillation

Run with:
    streamlit run dashboard/app.py

Features:
- Upload and visualize training logs
- Compare multiple training runs
- Interactive prediction inspection
- Real-time monitoring (if training in progress)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.training_visualizer import TrainingVisualizer
except ImportError:
    st.error("Could not import training_visualizer. Make sure you're running from the project root.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="VAD Training Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


def load_training_log(log_path: Path) -> pd.DataFrame:
    """Load training log CSV."""
    return pd.read_csv(log_path)


def load_predictions(pred_path: Path) -> dict:
    """Load predictions NPZ file."""
    data = np.load(pred_path)
    return {
        'predictions': data['predictions'],
        'labels': data['labels'],
        'probs': data['probs'],
        'utt_ids': data.get('utt_ids', np.arange(len(data['predictions'])))
    }


def load_summary(summary_path: Path) -> dict:
    """Load summary JSON file."""
    with open(summary_path) as f:
        return json.load(f)


def plot_training_curves_plotly(df: pd.DataFrame, title: str = "Training Curves"):
    """Create interactive training curves with Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation AUC', 'Error Rates', 'F1 Score'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Loss curves
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train_loss'], name='Total Loss',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train_hard_loss'], name='Hard Loss',
                      line=dict(color='#ff7f0e', width=1.5, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['train_soft_loss'], name='Soft Loss',
                      line=dict(color='#2ca02c', width=1.5, dash='dash')),
            row=1, col=1
        )

        # Val AUC
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_auc'], name='Val AUC',
                      line=dict(color='#9467bd', width=2), mode='lines+markers'),
            row=1, col=2
        )
        best_idx = df['val_auc'].idxmax()
        fig.add_trace(
            go.Scatter(x=[df.loc[best_idx, 'epoch']], y=[df.loc[best_idx, 'val_auc']],
                      mode='markers', name='Best AUC',
                      marker=dict(color='red', size=12, symbol='star')),
            row=1, col=2
        )

        # Error rates
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_miss_rate'], name='Miss Rate',
                      line=dict(color='#e377c2', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_false_alarm_rate'], name='False Alarm Rate',
                      line=dict(color='#7f7f7f', width=2)),
            row=2, col=1
        )

        # F1 Score
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df['val_f1'], name='F1 Score',
                      line=dict(color='#d62728', width=2), mode='lines+markers'),
            row=2, col=2
        )

        fig.update_layout(
            title_text=title,
            height=600,
            showlegend=True,
            template='plotly_white'
        )

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="AUC", row=1, col=2, range=[0, 1])
        fig.update_yaxes(title_text="Rate", row=2, col=1, range=[0, 1])
        fig.update_yaxes(title_text="F1", row=2, col=2, range=[0, 1])

        return fig
    except ImportError:
        st.warning("Plotly not available. Using matplotlib instead.")
        return None


def plot_predictions_plotly(data: dict, utt_id: str):
    """Plot predictions for a specific utterance."""
    try:
        import plotly.graph_objects as go

        mask = data['utt_ids'] == utt_id
        labels = data['labels'][mask]
        probs = data['probs'][mask]
        preds = data['predictions'][mask]

        fig = go.Figure()

        # Ground truth
        fig.add_trace(go.Scatter(
            x=list(range(len(labels))),
            y=labels,
            mode='lines',
            name='Ground Truth',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.2)'
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=list(range(len(probs))),
            y=probs,
            mode='lines',
            name='Probability',
            line=dict(color='blue', width=2)
        ))

        # Threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                     annotation_text="Threshold")

        # Mark correct/incorrect
        correct = preds == labels
        incorrect = ~correct

        fig.add_trace(go.Scatter(
            x=np.where(correct)[0],
            y=probs[correct],
            mode='markers',
            name='Correct',
            marker=dict(color='green', size=8)
        ))

        fig.add_trace(go.Scatter(
            x=np.where(incorrect)[0],
            y=probs[incorrect],
            mode='markers',
            name='Incorrect',
            marker=dict(color='red', size=10, symbol='x')
        ))

        fig.update_layout(
            title=f"Predictions for {utt_id}",
            xaxis_title="Frame",
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=400,
            template='plotly_white'
        )

        return fig
    except ImportError:
        return None


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<p class="main-header">📊 VAD Training Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Interactive visualization for Voice Activity Detection training")

    # Sidebar
    st.sidebar.header("📁 Data Selection")

    # Auto-discover available logs
    outputs_dir = Path("outputs")
    available_logs = []
    if outputs_dir.exists():
        available_logs = list(outputs_dir.rglob("fold_*.csv"))

    if available_logs:
        st.sidebar.info(f"Found {len(available_logs)} training logs")

    # File upload or selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Upload Files", "Select from Outputs"]
    )

    log_df = None
    summary = None
    predictions = None

    if data_source == "Upload Files":
        uploaded_log = st.sidebar.file_uploader("Upload Training Log (CSV)", type=['csv'])
        uploaded_summary = st.sidebar.file_uploader("Upload Summary (JSON)", type=['json'])
        uploaded_predictions = st.sidebar.file_uploader("Upload Predictions (NPZ)", type=['npz'])

        if uploaded_log:
            log_df = pd.read_csv(uploaded_log)
            st.sidebar.success("Log loaded successfully!")

        if uploaded_summary:
            summary = json.load(uploaded_summary)

        if uploaded_predictions:
            # Save to temp file and load
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
                tmp.write(uploaded_predictions.read())
                predictions = load_predictions(Path(tmp.name))

    else:
        if available_logs:
            def format_log_path(x):
                try:
                    return str(x.relative_to(project_root))
                except ValueError:
                    return str(x.name)

            selected_log = st.sidebar.selectbox(
                "Select Training Log",
                options=available_logs,
                format_func=format_log_path
            )

            if selected_log:
                log_df = load_training_log(selected_log)
                st.sidebar.success(f"Loaded: {selected_log.name}")

                # Try to load corresponding summary and predictions
                log_dir = selected_log.parent
                fold_id = selected_log.stem.replace("fold_", "")

                summary_path = log_dir / f"fold_{fold_id}_summary.json"
                if summary_path.exists():
                    summary = load_summary(summary_path)
                    st.sidebar.success(f"Loaded summary")

                predictions_path = log_dir / f"fold_{fold_id}_predictions.npz"
                if predictions_path.exists():
                    predictions = load_predictions(predictions_path)
                    st.sidebar.success(f"Loaded predictions")
        else:
            st.sidebar.warning("No training logs found in outputs/ directory")

    # Main content
    if log_df is not None:
        # Summary metrics
        st.header("📈 Training Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Best Val AUC",
                value=f"{log_df['val_auc'].max():.4f}",
                delta=f"Epoch {log_df.loc[log_df['val_auc'].idxmax(), 'epoch']:.0f}"
            )

        with col2:
            st.metric(
                label="Final Val AUC",
                value=f"{log_df['val_auc'].iloc[-1]:.4f}"
            )

        with col3:
            st.metric(
                label="Best F1",
                value=f"{log_df['val_f1'].max():.4f}"
            )

        with col4:
            st.metric(
                label="Total Epochs",
                value=f"{len(log_df)}"
            )

        # Training curves
        st.header("📉 Training Curves")

        fig = plot_training_curves_plotly(log_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to matplotlib
            st.line_chart(log_df.set_index('epoch')[['train_loss', 'val_auc', 'val_f1']])

        # Detailed metrics table
        st.header("📋 Detailed Metrics")
        st.dataframe(log_df, use_container_width=True)

        # Download button
        csv = log_df.to_csv(index=False)
        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name="training_metrics.csv",
            mime="text/csv"
        )

        # Predictions visualization
        if predictions:
            st.header("🔍 Prediction Visualization")

            unique_utts = np.unique(predictions['utt_ids'])
            selected_utt = st.selectbox(
                "Select Utterance",
                options=unique_utts,
                format_func=lambda x: f"{x} ({np.sum(predictions['utt_ids'] == x)} frames)"
            )

            if selected_utt:
                pred_fig = plot_predictions_plotly(predictions, selected_utt)
                if pred_fig:
                    st.plotly_chart(pred_fig, use_container_width=True)

                # Statistics for this utterance
                mask = predictions['utt_ids'] == selected_utt
                utt_labels = predictions['labels'][mask]
                utt_probs = predictions['probs'][mask]
                utt_preds = predictions['predictions'][mask]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    accuracy = np.mean(utt_preds == utt_labels)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    speech_ratio = np.mean(utt_labels)
                    st.metric("Speech Ratio", f"{speech_ratio:.2%}")
                with col3:
                    avg_conf = np.mean(utt_probs)
                    st.metric("Avg Confidence", f"{avg_conf:.3f}")
                with col4:
                    frames = len(utt_labels)
                    st.metric("Total Frames", f"{frames}")

        # Summary information
        if summary:
            st.header("📄 Run Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Configuration")
                st.json({
                    'fold_id': summary.get('fold_id'),
                    'train_speakers': summary.get('train_speakers'),
                    'val_speaker': summary.get('val_speaker'),
                    'test_speaker': summary.get('test_speaker'),
                    'num_parameters': summary.get('num_parameters'),
                    'model_size_mb': summary.get('model_size_mb')
                })

            with col2:
                st.subheader("Test Results")
                test_metrics = summary.get('test_metrics', {})
                st.json(test_metrics)

    else:
        # Welcome screen
        st.info("👈 Please upload training data or select from available logs in the sidebar")

        st.markdown("""
        ### Getting Started

        1. **Upload your training log** (CSV file from training output)
        2. **Optionally upload**:
           - Summary JSON for fold configuration
           - Predictions NPZ for prediction visualization

        3. **Or select** from automatically discovered logs in `outputs/`

        ### Features

        - 📊 **Training Curves**: Visualize loss, AUC, F1, and error rates over epochs
        - 🔍 **Prediction Inspection**: Examine model predictions on individual utterances
        - 📋 **Metrics Table**: Full table of all logged metrics
        - 💾 **Export**: Download processed metrics as CSV
        """)

        # Example with dummy data
        st.markdown("---")
        st.subheader("Preview with Example Data")

        if st.button("Load Example Data"):
            # Create dummy data
            epochs = 20
            dummy_data = {
                'epoch': range(1, epochs + 1),
                'train_loss': [1.0 - 0.04 * i + np.random.rand() * 0.05 for i in range(epochs)],
                'train_hard_loss': [0.6 - 0.02 * i + np.random.rand() * 0.03 for i in range(epochs)],
                'train_soft_loss': [0.4 - 0.02 * i + np.random.rand() * 0.03 for i in range(epochs)],
                'val_auc': np.clip([0.5 + 0.02 * i + np.random.rand() * 0.02 for i in range(epochs)], 0, 1),
                'val_f1': np.clip([0.45 + 0.018 * i + np.random.rand() * 0.02 for i in range(epochs)], 0, 1),
                'val_miss_rate': np.clip([0.5 - 0.015 * i + np.random.rand() * 0.03 for i in range(epochs)], 0, 1),
                'val_false_alarm_rate': np.clip([0.4 - 0.01 * i + np.random.rand() * 0.02 for i in range(epochs)], 0, 1),
                'val_accuracy': np.clip([0.6 + 0.015 * i + np.random.rand() * 0.02 for i in range(epochs)], 0, 1),
                'learning_rate': [0.001 * (0.9 ** i) for i in range(epochs)],
                'time': [30 + np.random.rand() * 10 for i in range(epochs)]
            }

            log_df = pd.DataFrame(dummy_data)
            st.session_state['log_df'] = log_df
            st.rerun()


if __name__ == "__main__":
    main()
