#!/usr/bin/env python3

import os
import json
import argparse
import logging
import subprocess
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import openai
        import pandas
        import geopandas
        import matplotlib
        import streamlit
        logger.info("All required dependencies are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Installing required dependencies...")
        subprocess.run([
            "pip", "install",
            "openai", "pandas", "geopandas", "matplotlib", "streamlit"
        ])
        return False


def run_agent(args):
    """
    Run the spatial analysis agent.

    Args:
        args: Command-line arguments
    """
    logger.info("Starting agent implementation")

    # Use the specified model or environment variable
    model_name = args.model or os.environ.get(
        "FINE_TUNED_MODEL_NAME")
    logger.info(f"Using model: {model_name}")

    # Export model name for interface to use
    os.environ["AGENT_MODEL_NAME"] = model_name

    if args.interface == "terminal":
        # Run terminal interface
        logger.info(f"Running terminal interface with model: {model_name}")

        try:
            from agent_framework import SpatialAnalysisAgent

            # Initialize the agent
            agent = SpatialAnalysisAgent(
                data_dir=args.data_dir,
                model_name=model_name,
                openai_api_key=args.openai_api_key
            )

            # Run conversation loop
            print(
                f"\nSpatial Analysis Agent initialized with model: {model_name}")
            print(
                "Enter queries about commercial parcels in Cambridge, MA. Type 'exit' to quit.")

            while True:
                # Get user input
                user_message = input("\nYour query: ")

                # Check for exit command
                if user_message.lower() in ['exit', 'quit', 'bye']:
                    print("Exiting. Goodbye!")
                    break

                # Process the message
                print("\nProcessing your query...")
                result = agent.run_conversation(user_message)

                # Print the response
                print("\nAgent response:")
                print(result["message"])

                # Handle visualization if available
                if "data" in result and "visualization_path" in result["data"]:
                    vis_path = result["data"]["visualization_path"]
                    if vis_path and os.path.exists(vis_path):
                        print(f"\nVisualization saved to: {vis_path}")

                        # Try to open the visualization if possible
                        try:
                            import platform
                            if platform.system() == "Darwin":  # macOS
                                subprocess.run(["open", vis_path])
                            elif platform.system() == "Windows":
                                os.startfile(vis_path)
                            else:  # Linux
                                subprocess.run(["xdg-open", vis_path])
                        except Exception as e:
                            print(
                                f"Could not open visualization automatically: {e}")

        except Exception as e:
            logger.error(f"Error running terminal interface: {e}")
            return False

    else:  # Web interface (default)
        # Run Streamlit web interface
        logger.info(f"Running web interface with model: {model_name}")

        try:
            # Check if streamlit is installed
            import streamlit

            # Run the streamlit app
            streamlit_command = f"streamlit run spatial_analysis_interface.py -- --data_dir {args.data_dir} --model {model_name}"
            subprocess.run(streamlit_command, shell=True)

        except ImportError:
            logger.error("Streamlit not installed. Trying to install...")
            subprocess.run(["pip", "install", "streamlit"])

            # Try again
            streamlit_command = f"streamlit run spatial_analysis_interface.py -- --data_dir {args.data_dir} --model {model_name}"
            subprocess.run(streamlit_command, shell=True)

        except Exception as e:
            logger.error(f"Error running web interface: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run spatial analysis agent")

    # General arguments
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory containing the data files")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key (will use environment variable if not provided)")

    # Implementation arguments
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID to use (or will use FINE_TUNED_MODEL_NAME environment variable)")
    parser.add_argument("--interface", type=str, default="web",
                        choices=["web", "terminal"], help="Type of interface to run")

    args = parser.parse_args()

    # Get API key from environment if not provided
    if not args.openai_api_key:
        args.openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Check dependencies
    check_dependencies()

    # Run agent implementation
    run_agent(args)

    logger.info("Implementation complete!")


if __name__ == "__main__":
    main()
