import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="MNIST Digit Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model with caching to improve performance"""
    try:
        model = keras.models.load_model("mnist_model.keras")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'mnist_model.keras' not found. Please ensure it's in the working directory.")
        st.stop()

def preprocess_image(image, invert=True):
    """Preprocess image for MNIST model"""
    # Convert to grayscale
    image = image.convert("L")
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Invert if needed (MNIST expects white digits on black background)
    if invert:
        img_array = 1 - img_array
    
    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, image

def create_confidence_chart(prediction):
    """Create a beautiful confidence chart"""
    digits = list(range(10))
    confidences = prediction[0] * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=digits,
            y=confidences,
            marker=dict(
                color=confidences,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence %")
            ),
            text=[f'{c:.1f}%' for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores for All Digits",
        xaxis_title="Digit",
        yaxis_title="Confidence (%)",
        height=400,
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig

# Load model
model = load_model()

# Header
st.markdown("# 🧠 MNIST Digit Predictor")
st.markdown("### AI-Powered Handwritten Digit Recognition")

# Single Upload Tab
st.markdown("---")
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload a 28x28 pixel image or any size image with a single digit"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_array, processed_img = preprocess_image(image)
            
            st.success("✅ Image processed successfully!")
            
            # Prediction
            prediction = model.predict(img_array, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = prediction[0][predicted_digit] * 100
            
            # Display results
            with col2:
                st.subheader("📊 Prediction Results")
                
                # Main prediction box
                pred_html = f"""
                <div class="prediction-box">
                    <h1>{predicted_digit}</h1>
                    <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                </div>
                """
                st.markdown(pred_html, unsafe_allow_html=True)
                
                # Confidence level indicator
                if confidence > 95:
                    st.success("🎯 Very High Confidence!")
                elif confidence > 85:
                    st.info("✅ High Confidence")
                elif confidence > 70:
                    st.warning("⚠️ Moderate Confidence")
                else:
                    st.error("❌ Low Confidence - Image may be unclear")
            
            # Show processed image
            col1.subheader("🖼️ Uploaded Image")
            col1.image(uploaded_file, width=200)
            
            col2.subheader("⚙️ Processed Image (28x28)")
            col2.image(processed_img, width=200)
            
            # Confidence chart
            st.markdown("---")
            st.plotly_chart(create_confidence_chart(prediction), use_container_width=True)
            
            # Top 3 predictions
            st.markdown("---")
            st.subheader("🏆 Top 3 Predictions")
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            
            for rank, idx in enumerate(top_3_idx, 1):
                conf = prediction[0][idx] * 100
                st.metric(f"#{rank}", f"Digit {idx}", f"{conf:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")