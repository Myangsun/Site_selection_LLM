import os
import json
import re
import tempfile
import subprocess
import logging
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import openai
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialAnalysisAgent:
    """Agent framework for spatial analysis using LLMs."""

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        openai_api_key: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the spatial analysis agent.

        Args:
            data_dir: Directory containing the geospatial datasets
            model_name: Name of the LLM model to use
            openai_api_key: OpenAI API key (optional, uses env var if not provided)
            system_prompt: Custom system prompt (optional)
        """
        self.data_dir = data_dir
        self.model_name = model_name

        # Set up OpenAI client
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"))

        # Initialize conversation history
        self.conversation_history = []

        # Set up data file paths
        self.data_files = {
            'parcels': os.path.join(data_dir, 'cambridge_parcels.geojson'),
            'poi': os.path.join(data_dir, 'cambridge_poi_processed.geojson'),
            'census': os.path.join(data_dir, 'cambridge_census_cambridge_pct.geojson'),
            'spend': os.path.join(data_dir, 'cambridge_spend_processed.csv')
        }

        # Verify data files exist
        for name, path in self.data_files.items():
            if not os.path.exists(path):
                logger.warning(f"Data file not found: {path}")

        # Set up system prompt
        default_system_prompt = """You are a commercial site selection assistant that helps users find optimal locations in Cambridge, MA. 
You translate natural language queries into Python code using GeoPandas for geospatial analysis. 
Given a query about finding parcels with specific constraints, you:
1. Analyze the constraints in the query
2. Generate Python code to find matching parcels
3. Execute the code against Cambridge geospatial data
4. Provide results and visualizations

Use your expertise in spatial analysis, zoning regulations, and commercial real estate to help users find the best locations."""

        self.system_prompt = system_prompt or default_system_prompt

        # Add system message to history
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })

        # Set up available functions
        self.available_functions = {
            "generate_spatial_analysis_code": self.generate_spatial_analysis_code,
            "visualize_results": self.visualize_results
        }

        # Keep track of the most recent results
        self.current_results = {
            "parcel_ids": [],
            "code": "",
            "visualization_path": None,
            "output": ""
        }

        # Create output directory for visualizations
        self.output_dir = os.path.join(
            tempfile.gettempdir(), "spatial_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(
            f"Initialized SpatialAnalysisAgent with model: {model_name}")
        logger.info(f"Using data directory: {data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def generate_spatial_analysis_code(self, arguments: str) -> Dict[str, Any]:
        """
        Function to generate Python code for spatial analysis.

        Args:
            arguments: JSON string with parameters (query, etc.)

        Returns:
            Dictionary with status and generated code
        """
        try:
            # Parse arguments
            args = json.loads(arguments)
            query = args.get("query", "")

            if not query:
                return {"status": "error", "error": "No query provided"}

            # If code is already provided in arguments, use it
            if "code" in args:
                code = args["code"]
                logger.info(f"Using provided code for query: {query[:50]}...")
                return {"status": "success", "code": code}

            # Generate code using the model
            code_prompt = f"""Generate Python code to answer this site selection query: "{query}"

The code should:
1. Use GeoPandas to analyze parcels, POI, and census data in Cambridge, MA
2. Filter parcels based on the criteria in the query
3. Return a sorted list of parcel IDs (ml column) that match the criteria
4. Handle spatial operations efficiently (use projected CRS epsg=26986 for accurate distance calculations)
5. Be executable without modifications

Available data files:
- Cambridge parcel data: {self.data_files['parcels']}
- POI data: {self.data_files['poi']}
- Census data: {self.data_files['census']}
- Spending data: {self.data_files['spend']}

Include code to print the final list of parcel IDs.
Only provide the Python code without explanations."""

            # Call the LLM to generate code
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in geospatial data analysis using Python."},
                    {"role": "user", "content": code_prompt}
                ],
                temperature=0.1  # Low temperature for deterministic output
            )

            # Extract code from response
            code = response.choices[0].message.content

            # Clean up the code (remove markdown backticks if present)
            code = re.sub(r'^```python\n', '', code)
            code = re.sub(r'\n```$', '', code)
            code = re.sub(r'^```\n', '', code)
            code = re.sub(r'^```python', '', code)
            code = re.sub(r'```$', '', code)

            # Store the code
            self.current_results["code"] = code

            logger.info(f"Generated code for query: {query[:50]}...")
            return {"status": "success", "code": code}

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {"status": "error", "error": str(e)}

    def execute_code(self, code: str) -> Tuple[bool, str, List[str]]:
        """
        Execute Python code and capture the output.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (success_flag, output_text, list_of_parcel_ids)
        """
        # Create a temporary directory for code execution
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Update file paths in the code to use absolute paths
                for name, path in self.data_files.items():
                    code = code.replace(f"'{name}.geojson'", f"'{path}'")
                    code = code.replace(
                        f"'{name}_processed.geojson'", f"'{path}'")
                    code = code.replace(f"'{name}_processed.csv'", f"'{path}'")

                # Standard file names replacements
                code = code.replace(
                    "'cambridge_parcels.geojson'", f"'{self.data_files['parcels']}'")
                code = code.replace(
                    "'cambridge_poi_processed.geojson'", f"'{self.data_files['poi']}'")
                code = code.replace(
                    "'cambridge_census_cambridge_pct.geojson'", f"'{self.data_files['census']}'")
                code = code.replace(
                    "'cambridge_spend_processed.csv'", f"'{self.data_files['spend']}'")

                # Make sure there's a print statement for the result_ids
                if "result_ids" in code and "print(f" not in code and "print(result_ids)" not in code:
                    code += "\nprint(result_ids)"

                # Write the code to a temporary file
                script_path = os.path.join(temp_dir, "analysis.py")
                with open(script_path, "w") as f:
                    f.write(code)

                logger.info(f"Executing code at: {script_path}")

                # Execute the code and capture output
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5-minute timeout
                )

                if result.returncode != 0:
                    logger.error(f"Code execution failed: {result.stderr}")
                    return False, result.stderr, []

                # Parse the output to extract parcel IDs
                output = result.stdout.strip()

                # Store the output
                self.current_results["output"] = output

                # Try to extract a list of IDs from the output
                try:
                    # First, look for lists in square brackets
                    match = re.search(r'\[\'(.*?)\'\]', output)
                    if match:
                        # Extract IDs from string representation of list
                        ids_str = match.group(1)
                        parcel_ids = [id.strip("' ")
                                      for id in ids_str.split("', '")]
                    else:
                        # Just split by lines and clean up
                        lines = output.strip().split('\n')
                        if len(lines) == 1 and '[' in lines[0] and ']' in lines[0]:
                            # It's a list printed on a single line
                            list_content = lines[0][lines[0].find(
                                '[')+1:lines[0].find(']')]
                            parcel_ids = [id.strip("' \"")
                                          for id in list_content.split(',')]
                        else:
                            # Multiple lines, treat each as a potential ID
                            parcel_ids = [line.strip("' \"[](),")
                                          for line in lines if line.strip()]

                    # Store the extracted parcel IDs
                    self.current_results["parcel_ids"] = parcel_ids

                    logger.info(f"Extracted {len(parcel_ids)} parcel IDs")
                    return True, output, parcel_ids

                except Exception as e:
                    logger.error(f"Failed to parse output: {e}")
                    logger.debug(f"Raw output: {output}")
                    return False, output, []

            except subprocess.TimeoutExpired:
                logger.error("Code execution timed out")
                return False, "Execution timed out", []

            except Exception as e:
                logger.error(f"Error executing code: {e}")
                return False, str(e), []

    def visualize_results(self, arguments: str) -> Dict[str, Any]:
        """
        Function to visualize spatial analysis results.

        Args:
            arguments: JSON string with parameters (parcel_ids, title, etc.)

        Returns:
            Dictionary with status and visualization path
        """
        try:
            # Parse arguments
            args = json.loads(arguments)
            parcel_ids = args.get("parcel_ids", [])

            # Use current results if no IDs provided
            if not parcel_ids:
                parcel_ids = self.current_results.get("parcel_ids", [])

            if not parcel_ids:
                return {"status": "error", "error": "No parcel IDs to visualize"}

            # Get optional parameters
            title = args.get(
                "title", f"Selected Parcels ({len(parcel_ids)} results)")
            highlight_color = args.get("highlight_color", "red")

            # Load the parcels data
            parcels = gpd.read_file(self.data_files['parcels'])

            # Filter for the result parcels
            result_parcels = parcels[parcels['ml'].isin(parcel_ids)]

            # Create the visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot all parcels as context (with less opacity)
            parcels.plot(ax=ax, color='lightgrey', edgecolor='grey', alpha=0.3)

            # Plot result parcels (highlighted)
            result_parcels.plot(ax=ax, color=highlight_color,
                                edgecolor='black', alpha=0.7)

            # Add title and labels
            plt.title(title, fontsize=16)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)

            # Add a timestamp to avoid overwriting files
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"spatial_results_{timestamp}.png")

            # Save the figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Store the visualization path
            self.current_results["visualization_path"] = output_path

            logger.info(f"Visualization saved to: {output_path}")
            return {
                "status": "success",
                "visualization_path": output_path,
                "parcel_count": len(result_parcels)
            }

        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {"status": "error", "error": str(e)}

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return a response.

        Args:
            user_message: The user's message

        Returns:
            Dictionary with response message and data
        """
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            # Call the model for response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                functions=[
                    {
                        "name": "generate_spatial_analysis_code",
                        "description": "Generate Python code for geospatial analysis based on user query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The natural language query about site selection"
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "visualize_results",
                        "description": "Visualize spatial analysis results",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "parcel_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of parcel IDs to visualize"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Title for the visualization"
                                },
                                "highlight_color": {
                                    "type": "string",
                                    "description": "Color to highlight selected parcels"
                                }
                            }
                        }
                    }
                ],
                function_call="auto"
            )

            # Process the model's response
            assistant_message = response.choices[0].message

            # Clear current results for a new query
            self.current_results = {
                "parcel_ids": [],
                "code": "",
                "visualization_path": None,
                "output": ""
            }

            # Check if a function call was requested
            if hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                # Extract function call details
                function_name = assistant_message.function_call.name
                function_args = assistant_message.function_call.arguments

                # Add the assistant message to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_name,
                        "arguments": function_args
                    }
                })

                # Call the appropriate function
                if function_name in self.available_functions:
                    function_to_call = self.available_functions[function_name]
                    function_response = function_to_call(function_args)

                    # Add function response to conversation history
                    self.conversation_history.append({
                        "role": "function",
                        "name": function_name,
                        "content": json.dumps(function_response)
                    })

                    # Generate response based on function output
                    if function_name == "generate_spatial_analysis_code" and function_response.get("status") == "success":
                        # Execute the generated code
                        success, output, parcel_ids = self.execute_code(
                            function_response["code"])

                        # Create visualization if code execution was successful
                        if success and parcel_ids:
                            viz_args = json.dumps({"parcel_ids": parcel_ids})
                            viz_response = self.visualize_results(viz_args)

                            # Create a combined response with code execution and visualization results
                            combined_response = {
                                "code_execution": {"success": success, "output": output, "parcel_ids": parcel_ids},
                                "visualization": viz_response
                            }

                            # Add combined response to conversation history
                            self.conversation_history.append({
                                "role": "function",
                                "name": "execute_and_visualize",
                                "content": json.dumps(combined_response)
                            })
                        else:
                            # Just add code execution response
                            code_response = {
                                "success": success,
                                "output": output,
                                "parcel_ids": parcel_ids
                            }

                            self.conversation_history.append({
                                "role": "function",
                                "name": "execute_code",
                                "content": json.dumps(code_response)
                            })

                    # Get final assistant response after function calls
                    final_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.conversation_history
                    )

                    # Extract final message
                    final_message = final_response.choices[0].message.content

                    # Add to conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_message
                    })

                    # Return final result
                    return {
                        "message": final_message,
                        "data": self.current_results
                    }
                else:
                    error_message = f"Function {function_name} not implemented"
                    logger.error(error_message)
                    return {"message": error_message, "data": {}}
            else:
                # No function call, just a regular response
                content = assistant_message.content

                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content
                })

                return {"message": content, "data": {}}

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"message": f"Error processing your request: {str(e)}", "data": {}}

    def run_conversation(self, user_message: str) -> Dict[str, Any]:
        """
        Run a full conversation turn with the user message.

        Args:
            user_message: The user's message

        Returns:
            Dictionary with response message and data
        """
        return self.process_message(user_message)

    def reset_conversation(self):
        """Reset the conversation history, keeping only the system prompt."""
        self.conversation_history = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        logger.info("Conversation history reset")
