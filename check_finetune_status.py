import os
import json
import logging
import argparse
import openai
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_fine_tuning_status(job_id):
    """Check the status of a fine-tuning job."""
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Retrieve the job status
        response = client.fine_tuning.jobs.retrieve(job_id)

        # Create status dictionary
        status_dict = {
            "id": response.id,
            "status": response.status,
            "fine_tuned_model": getattr(response, "fine_tuned_model", None),
            "created_at": response.created_at,
            "finished_at": getattr(response, "finished_at", None)
        }

        logger.info(f"Job {job_id} status: {status_dict['status']}")

        return status_dict

    except Exception as e:
        logger.error(f"Error checking fine-tuning status: {e}")
        return {"status": "error", "error": str(e)}


def update_fine_tuning_result(results_dir, job_id, status_dict):
    """Update the fine-tuning result file with current status."""
    try:
        # Get the existing file
        result_path = os.path.join(results_dir, "fine_tuning_result.json")

        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result = json.load(f)
        else:
            result = {"job_id": job_id}

        # Update the status
        result["status"] = status_dict

        # Save updated result
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Updated fine-tuning result saved to {result_path}")

        return result

    except Exception as e:
        logger.error(f"Error updating fine-tuning result: {e}")
        return None


def wait_for_completion(job_id, results_dir, check_interval=300, max_wait_time=None):
    """Wait for a fine-tuning job to complete."""
    logger.info(f"Waiting for fine-tuning job {job_id} to complete...")

    start_time = time.time()
    while True:
        status_dict = check_fine_tuning_status(job_id)
        update_fine_tuning_result(results_dir, job_id, status_dict)

        if status_dict["status"] in ["succeeded", "failed", "cancelled"]:
            logger.info(
                f"Job {job_id} completed with status: {status_dict['status']}")
            return status_dict

        # Check if we've exceeded the maximum wait time
        if max_wait_time and (time.time() - start_time) > max_wait_time:
            logger.warning(
                f"Exceeded maximum wait time. Job {job_id} is still in progress.")
            return status_dict

        wait_time = check_interval
        logger.info(
            f"Job {job_id} is still in progress. Checking again in {wait_time} seconds...")
        time.sleep(wait_time)


def main():
    parser = argparse.ArgumentParser(
        description="Check fine-tuning job status")

    parser.add_argument("--job_id", type=str, required=True,
                        help="Fine-tuning job ID to check")
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory to store results")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job completion")
    parser.add_argument("--check_interval", type=int, default=300,
                        help="Seconds between status checks (default: 5 minutes)")
    parser.add_argument("--max_wait_time", type=int, default=None,
                        help="Maximum seconds to wait (default: unlimited)")

    args = parser.parse_args()

    if args.wait:
        status = wait_for_completion(
            args.job_id,
            args.results_dir,
            args.check_interval,
            args.max_wait_time
        )
    else:
        status = check_fine_tuning_status(args.job_id)
        update_fine_tuning_result(args.results_dir, args.job_id, status)

    # Print status for convenience
    print(json.dumps(status, indent=2))

    # Print instructions if model is ready
    if status["status"] == "succeeded" and status["fine_tuned_model"]:
        print("\nModel is ready!")
        print(f"To use the model, run:")
        print(f"export FINE_TUNED_MODEL_NAME=\"{status['fine_tuned_model']}\"")
        print(f"python evaluate_methods.py --methods fine-tuned --skip-finetune")


if __name__ == "__main__":
    main()
