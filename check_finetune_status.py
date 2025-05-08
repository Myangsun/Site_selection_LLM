import os
import json
import logging
import argparse
import openai
import time
import re
import matplotlib.pyplot as plt

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


# Updated helper function to plot fine-tune loss using the new API
def plot_finetune_loss(job_id):
    """
    Fetch fine-tune events for job_id, extract training loss, and display a plot.
    """
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Retrieve all events
        events = []
        try:
            # Set a higher limit to get more events (default might be too low)
            for event in client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id,
                limit=100  # Increase this if needed
            ):
                events.append(event)

            logger.info(f"Retrieved {len(events)} events for job {job_id}")

            # If no events were found, try retrieving the job to check its status
            if len(events) == 0:
                job = client.fine_tuning.jobs.retrieve(job_id)
                logger.info(f"Job status: {job.status}")

                # If the job is not complete, we might not have loss data yet
                if job.status not in ["succeeded", "failed", "cancelled"]:
                    logger.warning(
                        f"Job {job_id} is still in progress (status: {job.status}). Loss data may not be available yet.")

        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return

        # Parse out step and loss
        steps = []
        losses = []
        val_steps = []
        val_losses = []

        for ev in events:
            msg = ev.message

            # Check for training loss - match the exact format from the logs
            # Format: "Step X/Y: training loss=Z, validation loss=W, full validation loss=V"
            m_train = re.search(
                r"Step\s+(\d+)/\d+:.*?training loss=\s*([0-9.]+)", msg)
            if m_train:
                step_num = int(m_train.group(1))
                train_loss = float(m_train.group(2))
                logger.info(
                    f"Found training loss: Step {step_num}, Loss {train_loss}")
                steps.append(step_num)
                losses.append(train_loss)

            # Also check for validation loss if available
            m_val = re.search(
                r"Step\s+(\d+)/\d+:.*?validation loss=\s*([0-9.]+)", msg)
            if m_val:
                step_num = int(m_val.group(1))
                val_loss = float(m_val.group(2))
                logger.info(
                    f"Found validation loss: Step {step_num}, Loss {val_loss}")
                val_steps.append(step_num)
                val_losses.append(val_loss)

        # Plot
        if steps and losses:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, 'b-', label='Training Loss')

            # Add validation loss if available
            if val_steps and val_losses:
                plt.plot(val_steps, val_losses, 'r-', label='Validation Loss')
                plt.legend()

            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"Fine-tune Loss Curve for {job_id}")
            plt.grid(True, linestyle='--', alpha=0.7)

            output_file = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), f"finetune_loss_{job_id}.png")
            plt.savefig(output_file)
            plt.show()
            logger.info(f"Loss plot saved to {output_file}")
        else:
            logger.warning("No training loss events found to plot.")

            # Debug info about the events
            if events:
                logger.info(
                    "Found events, but no loss data was extracted. Event messages:")
                for i, ev in enumerate(events[:5]):  # Show first 5 events
                    logger.info(f"Event {i}: {ev.message}")

    except Exception as e:
        logger.error(f"Error plotting fine-tune loss: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Check fine-tuning job status")

    parser.add_argument("--job_id", type=str, required=True,
                        help="Fine-tuning job ID to check")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to store results")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job completion")
    parser.add_argument("--check_interval", type=int, default=300,
                        help="Seconds between status checks (default: 5 minutes)")
    parser.add_argument("--max_wait_time", type=int, default=None,
                        help="Maximum seconds to wait (default: unlimited)")

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

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
        # Plot the fine-tune loss curve
        try:
            plot_finetune_loss(args.job_id)
        except Exception as e:
            logger.error(f"Error plotting fine-tune loss: {e}")


if __name__ == "__main__":
    main()
