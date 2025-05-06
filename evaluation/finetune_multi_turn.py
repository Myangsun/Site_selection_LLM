import os
import json
import logging
import random
import openai
import time
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpatialFineTuningHandler:
    """Handler for fine-tuning models on spatial analysis tasks."""

    def __init__(self, data_dir: str, openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")):
        """
        Initialize the fine-tuning handler.

        Args:
            data_dir: Directory containing the training data
            openai_api_key: OpenAI API key for API access
        """
        self.data_dir = data_dir

        # Set up OpenAI client
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("OpenAI API key not provided")

        # Load test samples if available
        self.samples = self.load_samples()

        logger.info(
            f"Fine-tuning handler initialized with data dir: {data_dir}")
        logger.info(f"Loaded {len(self.samples)} samples for fine-tuning")

    def load_samples(self) -> List[Dict]:
        """
        Load samples from the data directory.

        Returns:
            List of dictionaries with Query, Code, Answer keys
        """
        try:
            sample_path = os.path.join(self.data_dir, 'spatial_samples.json')
            if os.path.exists(sample_path):
                with open(sample_path, 'r', encoding='utf-8-sig') as f:
                    samples = json.load(f)
                logger.info(
                    f"Loaded {len(samples)} samples from {sample_path}")
                return samples
            else:
                logger.warning(f"Samples file not found at {sample_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading samples: {e}")
            return []

    def prepare_standard_format(self,
                                output_dir: str,
                                validation_split: float = 0.2) -> Dict[str, str]:
        """
        Prepare data for fine-tuning in standard format.

        Args:
            output_dir: Directory to save prepared data
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with paths to training and validation files
        """
        if not self.samples:
            logger.error("No samples available for preparing fine-tuning data")
            return {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create a copy of samples and shuffle
        all_samples = self.samples.copy()
        random.shuffle(all_samples)

        # Split into training and validation sets
        if validation_split > 0:
            split_index = int(len(all_samples) * (1 - validation_split))
            training_samples = all_samples[:split_index]
            validation_samples = all_samples[split_index:]

            logger.info(f"Split dataset: {len(training_samples)} training samples, "
                        f"{len(validation_samples)} validation samples")
        else:
            training_samples = all_samples
            validation_samples = []

        # Prepare training data
        training_data = []
        for sample in training_samples:
            # Create simple prompt-completion format
            entry = {
                "messages": [
                    {"role": "system", "content": "You are an expert in geospatial data analysis using Python."},
                    {"role": "user",
                        "content": f"Generate Python code to answer this query: {sample['Query']}"},
                    {"role": "assistant", "content": sample['Code']}
                ]
            }
            training_data.append(entry)

        # Save training data
        training_file = os.path.join(output_dir, "training_data.jsonl")
        with open(training_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')

        logger.info(f"Training data saved to {training_file}")

        # Prepare validation data if we have a split
        validation_file = None
        if validation_samples:
            validation_data = []
            for sample in validation_samples:
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are an expert in geospatial data analysis using Python."},
                        {"role": "user",
                            "content": f"Generate Python code to answer this query: {sample['Query']}"},
                        {"role": "assistant", "content": sample['Code']}
                    ]
                }
                validation_data.append(entry)

            validation_file = os.path.join(output_dir, "validation_data.jsonl")
            with open(validation_file, 'w') as f:
                for item in validation_data:
                    f.write(json.dumps(item) + '\n')

            logger.info(f"Validation data saved to {validation_file}")

        return {
            "training_file": training_file,
            "validation_file": validation_file
        }

    def prepare_multi_turn_format(self,
                                  output_dir: str,
                                  validation_split: float = 0.2,
                                  system_prompt: Optional[str] = None) -> Dict[str, str]:
        """
        Prepare data for fine-tuning with multi-turn conversations and function calls.

        Args:
            output_dir: Directory to save prepared data
            validation_split: Fraction of data to use for validation
            system_prompt: Custom system prompt to use

        Returns:
            Dictionary with paths to training and validation files
        """
        if not self.samples:
            logger.error("No samples available for preparing fine-tuning data")
            return {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create a copy of samples and shuffle
        all_samples = self.samples.copy()
        random.shuffle(all_samples)

        # Split into training and validation sets
        if validation_split > 0:
            split_index = int(len(all_samples) * (1 - validation_split))
            training_samples = all_samples[:split_index]
            validation_samples = all_samples[split_index:]

            logger.info(f"Split dataset: {len(training_samples)} training samples, "
                        f"{len(validation_samples)} validation samples")
        else:
            training_samples = all_samples
            validation_samples = []

        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = "You are a commercial site selection assistant that generates Python code to analyze geospatial data."

        # Prepare training data
        training_data = []
        for sample in training_samples:
            # Create multi-turn conversation format
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Find {sample['Query']}"},
                    {"role": "assistant",
                        "content": "I'll generate Python code to find parcels matching your criteria."},
                    {"role": "user", "content": "Please proceed with the code."},
                    {"role": "assistant",
                        "content": f"Here's the Python code to find parcels matching your criteria:\n\n{sample['Code']}"},
                    {"role": "user", "content": "What parcels match these criteria?"},
                    {"role": "assistant",
                        "content": f"The matching parcels are: {sample['Answer']}"}
                ]
            }
            training_data.append(entry)

        # Save training data
        training_file = os.path.join(
            output_dir, "training_data_multi_turn.jsonl")
        with open(training_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')

        logger.info(f"Multi-turn training data saved to {training_file}")

        # Prepare validation data if we have a split
        validation_file = None
        if validation_samples:
            validation_data = []
            for sample in validation_samples:
                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Find {sample['Query']}"},
                        {"role": "assistant",
                            "content": "I'll generate Python code to find parcels matching your criteria."},
                        {"role": "user", "content": "Please proceed with the code."},
                        {"role": "assistant",
                            "content": f"Here's the Python code to find parcels matching your criteria:\n\n{sample['Code']}"},
                        {"role": "user", "content": "What parcels match these criteria?"},
                        {"role": "assistant",
                            "content": f"The matching parcels are: {sample['Answer']}"}
                    ]
                }
                validation_data.append(entry)

            validation_file = os.path.join(
                output_dir, "validation_data_multi_turn.jsonl")
            with open(validation_file, 'w') as f:
                for item in validation_data:
                    f.write(json.dumps(item) + '\n')

            logger.info(
                f"Multi-turn validation data saved to {validation_file}")

        return {
            "training_file": training_file,
            "validation_file": validation_file
        }

    def prepare_function_calling_format(self,
                                        output_dir: str,
                                        validation_split: float = 0.2) -> Dict[str, str]:
        """
        Prepare data for fine-tuning with function calling format.

        Args:
            output_dir: Directory to save prepared data
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary with paths to training and validation files
        """
        if not self.samples:
            logger.error("No samples available for preparing fine-tuning data")
            return {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create a copy of samples and shuffle
        all_samples = self.samples.copy()
        random.shuffle(all_samples)

        # Split into training and validation sets
        if validation_split > 0:
            split_index = int(len(all_samples) * (1 - validation_split))
            training_samples = all_samples[:split_index]
            validation_samples = all_samples[split_index:]

            logger.info(f"Split dataset: {len(training_samples)} training samples, "
                        f"{len(validation_samples)} validation samples")
        else:
            training_samples = all_samples
            validation_samples = []

        # Prepare training data
        training_data = []
        for sample in training_samples:
            # Create function calling format
            entry = {
                "messages": [
                    {"role": "system", "content": "You are a commercial site selection assistant that generates Python code to analyze geospatial data."},
                    {"role": "user", "content": f"Find {sample['Query']}"},
                    {"role": "assistant", "function_call": {
                        "name": "generate_spatial_analysis_code",
                        "arguments": json.dumps({"query": sample['Query'], "code": sample['Code']})
                    }},
                    {"role": "function", "name": "generate_spatial_analysis_code", "content": json.dumps(
                        {"status": "success", "code": sample['Code']})},
                    {"role": "user", "content": "Show me the results."},
                    {"role": "assistant",
                        "content": f"Here are the matching parcels: {sample['Answer']}"}
                ]
            }
            training_data.append(entry)

        # Save training data
        training_file = os.path.join(
            output_dir, "training_data_function_calling.jsonl")
        with open(training_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')

        logger.info(f"Function calling training data saved to {training_file}")

        # Prepare validation data if we have a split
        validation_file = None
        if validation_samples:
            validation_data = []
            for sample in validation_samples:
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are a commercial site selection assistant that generates Python code to analyze geospatial data."},
                        {"role": "user", "content": f"Find {sample['Query']}"},
                        {"role": "assistant", "function_call": {
                            "name": "generate_spatial_analysis_code",
                            "arguments": json.dumps({"query": sample['Query'], "code": sample['Code']})
                        }},
                        {"role": "function", "name": "generate_spatial_analysis_code", "content": json.dumps(
                            {"status": "success", "code": sample['Code']})},
                        {"role": "user", "content": "Show me the results."},
                        {"role": "assistant",
                            "content": f"Here are the matching parcels: {sample['Answer']}"}
                    ]
                }
                validation_data.append(entry)

            validation_file = os.path.join(
                output_dir, "validation_data_function_calling.jsonl")
            with open(validation_file, 'w') as f:
                for item in validation_data:
                    f.write(json.dumps(item) + '\n')

            logger.info(
                f"Function calling validation data saved to {validation_file}")

        return {
            "training_file": training_file,
            "validation_file": validation_file
        }

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to OpenAI for fine-tuning.

        Args:
            file_path: Path to the file to upload

        Returns:
            File ID from OpenAI
        """
        try:
            logger.info(f"Uploading file: {file_path}")

            response = self.client.files.create(
                file=open(file_path, "rb"),
                purpose="fine-tune"
            )

            file_id = response.id
            logger.info(f"File uploaded with ID: {file_id}")

            return file_id

        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def start_fine_tuning(self,
                          training_file_id: str,
                          validation_file_id: Optional[str] = None,
                          model: str = "gpt-4o-2024-08-06",
                          suffix: str = "spatial-agent",
                          n_epochs: Optional[int] = 3,
                          batch_size: Optional[int] = 16,
                          learning_rate_multiplier: Optional[float] = 0.08) -> str:
        """
        Start a fine-tuning job with OpenAI.

        Args:
            training_file_id: ID of the training file
            validation_file_id: ID of the validation file (optional)
            model: Base model to fine-tune
            suffix: Suffix to add to the model name
            n_epochs: Number of epochs to train for (optional)
            batch_size: Batch size for training (optional)
            learning_rate_multiplier: Learning rate multiplier (optional)

        Returns:
            Job ID from OpenAI
        """
        try:
            logger.info(f"Starting fine-tuning job with model: {model}")

            # Prepare hyperparameters if specified
            hyperparameters = {}
            if n_epochs is not None:
                hyperparameters["n_epochs"] = n_epochs
            if batch_size is not None:
                hyperparameters["batch_size"] = batch_size
            if learning_rate_multiplier is not None:
                hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

            # Create fine-tuning job parameters
            job_params = {
                "training_file": training_file_id,
                "model": model,
                "suffix": suffix
            }

            # Add optional parameters
            if validation_file_id:
                job_params["validation_file"] = validation_file_id

            if hyperparameters:
                job_params["hyperparameters"] = hyperparameters

            # Create fine-tuning job
            response = self.client.fine_tuning.jobs.create(**job_params)

            job_id = response.id
            logger.info(f"Fine-tuning job created with ID: {job_id}")

            return job_id

        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {e}")
            raise

    def check_fine_tuning_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a fine-tuning job.

        Args:
            job_id: ID of the fine-tuning job

        Returns:
            Dictionary with status information
        """
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)

            # Convert to dictionary for easier handling
            status_dict = {
                "id": response.id,
                "status": response.status,
                "fine_tuned_model": getattr(response, "fine_tuned_model", None),
                "created_at": response.created_at,
                "finished_at": getattr(response, "finished_at", None),
                "training_file": response.training_file,
                "validation_file": getattr(response, "validation_file", None),
                "result_files": getattr(response, "result_files", [])
            }

            logger.info(f"Job {job_id} status: {status_dict['status']}")

            return status_dict

        except Exception as e:
            logger.error(f"Error checking fine-tuning status: {e}")
            return {"status": "error", "error": str(e)}

    def wait_for_completion(self, job_id: str, check_interval: int = 60, max_checks: int = 60) -> Dict[str, Any]:
        """
        Wait for a fine-tuning job to complete.

        Args:
            job_id: ID of the fine-tuning job
            check_interval: Seconds between status checks
            max_checks: Maximum number of checks to perform

        Returns:
            Dictionary with final status information
        """
        logger.info(f"Waiting for fine-tuning job {job_id} to complete...")

        checks = 0
        while checks < max_checks:
            status = self.check_fine_tuning_status(job_id)

            if status["status"] in ["succeeded", "failed", "cancelled"]:
                logger.info(f"Job {job_id} {status['status']}")
                return status

            logger.info(
                f"Job {job_id} status: {status['status']}. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)
            checks += 1

        logger.warning(
            f"Max checks reached. Job {job_id} is still in progress.")
        return {"status": "timeout", "job_id": job_id}

    def run_fine_tuning_pipeline(self,
                                 output_dir: str,
                                 format_type: str = "standard",
                                 model: str = "gpt-4o-2024-08-06",
                                 suffix: Optional[str] = None,
                                 validation_split: float = 0.2,
                                 wait_for_completion: bool = True) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline.

        Args:
            output_dir: Directory to save intermediate files
            format_type: Format type for fine-tuning data ("standard", "multi_turn", or "function_calling")
            model: Base model to fine-tune
            suffix: Suffix to add to the model name
            validation_split: Fraction of data to use for validation
            wait_for_completion: Whether to wait for job completion

        Returns:
            Dictionary with pipeline results
        """
        try:
            # Generate suffix if not provided
            if suffix is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                suffix = f"spatial-agent-{timestamp}"

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # 1. Prepare data based on format type
            logger.info(
                f"Preparing fine-tuning data in {format_type} format...")

            if format_type == "multi_turn":
                data_files = self.prepare_multi_turn_format(
                    output_dir=output_dir,
                    validation_split=validation_split
                )
            elif format_type == "function_calling":
                data_files = self.prepare_function_calling_format(
                    output_dir=output_dir,
                    validation_split=validation_split
                )
            else:  # standard format
                data_files = self.prepare_standard_format(
                    output_dir=output_dir,
                    validation_split=validation_split
                )

            if not data_files.get("training_file"):
                raise ValueError("Failed to prepare training data")

            # 2. Upload files
            logger.info("Uploading training file...")
            training_file_id = self.upload_file(data_files["training_file"])

            validation_file_id = None
            if data_files.get("validation_file"):
                logger.info("Uploading validation file...")
                validation_file_id = self.upload_file(
                    data_files["validation_file"])

            # Wait for files to be processed
            logger.info("Waiting for files to be processed...")
            time.sleep(60)  # Wait a minute for file processing

            # 3. Start fine-tuning job
            logger.info("Starting fine-tuning job...")
            job_id = self.start_fine_tuning(
                training_file_id=training_file_id,
                validation_file_id=validation_file_id,
                model=model,
                suffix=suffix
            )

            # 4. Wait for completion if requested
            if wait_for_completion:
                logger.info("Waiting for fine-tuning to complete...")
                final_status = self.wait_for_completion(job_id)

                if final_status["status"] == "succeeded":
                    logger.info(
                        f"Fine-tuning completed successfully. Model: {final_status.get('fine_tuned_model')}")
                else:
                    logger.warning(
                        f"Fine-tuning did not complete successfully. Status: {final_status['status']}")

                return {
                    "job_id": job_id,
                    "status": final_status,
                    "data_files": data_files,
                    "training_file_id": training_file_id,
                    "validation_file_id": validation_file_id
                }
            else:
                return {
                    "job_id": job_id,
                    "status": {"status": "in_progress"},
                    "data_files": data_files,
                    "training_file_id": training_file_id,
                    "validation_file_id": validation_file_id
                }

        except Exception as e:
            logger.error(f"Error in fine-tuning pipeline: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
