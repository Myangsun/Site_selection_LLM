import os
import base64
import streamlit as st
import matplotlib.pyplot as plt
import geopandas as gpd
from PIL import Image
import io
import json
import time
from agent_framework import SpatialAnalysisAgent
import argparse

# Set page configuration
st.set_page_config(
    page_title="Cambridge Site Selection Assistant",
    page_icon="🏙️",
    layout="wide"
)

# Parse command line arguments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatial Analysis Web Interface")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory containing data files")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID to use")
    return parser.parse_args()


# Get arguments (when run from command line)
try:
    args = parse_args()
except SystemExit:
    # If running directly in Streamlit, use defaults
    class Args:
        data_dir = "../data"
        model = None
    args = Args()

# Configure paths
DATA_DIR = args.data_dir
MODEL_NAME = args.model or os.environ.get(
    "AGENT_MODEL_NAME") or os.environ.get("FINE_TUNED_MODEL_NAME") or "gpt-4o"

# Main title
st.title("📍 Cambridge Commercial Site Selection Assistant")

# Sidebar for information and settings
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This tool helps you find commercial parcels in Cambridge, MA based on natural language queries.
        
        **Example queries:**
        - Find commercial parcels within 500 meters of Harvard Square
        - Find retail parcels larger than 6000 square feet
        - Find parcels with no more than 2 competing restaurants within 800 meters
        - Find office parcels with at least 2 parking spaces in areas with high educational attainment
        """
    )

    st.header("Settings")
    model_display = st.text_input(
        "Model",
        value=MODEL_NAME,
        disabled=True,
        help="The model being used for analysis"
    )

    # Optional API key input
    custom_api_key = st.text_input(
        "OpenAI API Key (optional)", type="password")

    with st.expander("Advanced Settings"):
        run_visualization = st.checkbox("Generate visualization", value=True)
        timeout_seconds = st.slider("Timeout (seconds)", 10, 300, 120)

# Initialize session state if needed
if 'agent' not in st.session_state:
    # Use custom API key if provided, otherwise use environment variable
    api_key = custom_api_key if custom_api_key else os.environ.get(
        "OPENAI_API_KEY")

    # Initialize the agent
    st.session_state.agent = SpatialAnalysisAgent(
        data_dir=DATA_DIR,
        model_name=MODEL_NAME,
        openai_api_key=api_key
    )
    st.session_state.conversation_history = []
    st.session_state.result_data = {}

# Function to reset conversation


def reset_conversation():
    if 'agent' in st.session_state:
        st.session_state.agent.reset_conversation()
    st.session_state.conversation_history = []
    st.session_state.result_data = {}


# Reset button
if st.sidebar.button("Reset Conversation"):
    reset_conversation()

# Create a container for the visualization
viz_container = st.container()

# Create a form for user input
with st.form(key="query_form"):
    # User query input
    user_query = st.text_area(
        "Enter your site selection query:",
        height=100,
        placeholder="Example: Find commercial parcels within 500 meters of Harvard Square"
    )

    # Submit button
    submit_button = st.form_submit_button("Analyze")

# Process form submission
if submit_button and user_query:
    # Clear previous results
    with viz_container:
        st.empty()

    # Create a spinner while processing
    with st.spinner(f"Processing your query with model: {MODEL_NAME}..."):
        try:
            # Process the message with a timeout
            start_time = time.time()
            result = st.session_state.agent.run_conversation(user_query)
            processing_time = time.time() - start_time

            # Store the result and update conversation history
            st.session_state.result_data = result.get("data", {})
            st.session_state.conversation_history.append(
                {"role": "user", "content": user_query})
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": result["message"]})

            # Display the agent's response
            st.markdown("### Assistant Response")
            st.markdown(result["message"])

            # Display processing time
            st.caption(f"Processing time: {processing_time:.2f} seconds")

            # Display visualization if available
            if run_visualization and "visualization_path" in result.get("data", {}):
                viz_path = result["data"]["visualization_path"]
                if viz_path and os.path.exists(viz_path):
                    st.markdown("### Visualization")
                    st.image(viz_path, caption="Matching Parcels",
                             use_column_width=True)

            # Display code if available
            if "code" in result.get("data", {}):
                with st.expander("View Generated Code"):
                    st.code(result["data"]["code"], language="python")

            # Display execution output if available
            if "output" in result.get("data", {}):
                with st.expander("View Execution Output"):
                    st.text(result["data"]["output"])

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("### Conversation History")
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")


# Create custom visualizations if needed
def create_custom_visualization():
    """Create a custom visualization of the results."""
    try:
        # Check if we have result data
        if not st.session_state.result_data or "parcel_ids" not in st.session_state.result_data:
            return None

        # Load the parcels data
        parcels = gpd.read_file(os.path.join(
            DATA_DIR, 'cambridge_parcels.geojson'))

        # Filter for the result parcels
        parcel_ids = st.session_state.result_data["parcel_ids"]
        result_parcels = parcels[parcels['ml'].isin(parcel_ids)]

        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all parcels as context (with less opacity)
        parcels.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.3)

        # Plot result parcels (highlighted)
        result_parcels.plot(ax=ax, color='red', edgecolor='black', alpha=0.7)

        # Add title and labels
        plt.title(
            f"Selected Parcels ({len(result_parcels)} results)", fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)

        return buf

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


# Custom visualization button
if st.session_state.result_data and "parcel_ids" in st.session_state.result_data:
    if st.button("Create Custom Visualization"):
        with st.spinner("Creating visualization..."):
            viz_buffer = create_custom_visualization()
            if viz_buffer:
                st.image(viz_buffer, caption="Custom Visualization",
                         use_column_width=True)


# Footer
st.markdown("---")
st.caption(
    "This site selection assistant uses spatial analysis and machine learning to help find optimal commercial locations in Cambridge, MA. "
    "Data includes parcels, points of interest, census information, and consumer spending patterns."
)
