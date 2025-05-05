# Site_selection_LLM

This project uses LLMs to analyze natural language queries about parcels in Cambridge, MA and generate Python code for spatial analysis. It consists of two main components:

1. **Method Evaluation**: Compare zero-shot, few-shot, and fine-tuning approaches
2. **Agent Implementation**: Build an interactive agent using the best-performing method

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Required packages:
  ```
  pip install openai pandas geopandas matplotlib seaborn streamlit tqdm shapely
  ```

### Setup

1. Clone the repository
2. Set up your environment:
   ```
   export OPENAI_API_KEY="your-api-key"
   ```
3. Place the data files in the `data` directory

## Method Evaluation

Run the evaluation script to compare different approaches:

```bash
cd evaluation
python evaluate_methods.py
```

This will:

- Run zero-shot, few-shot, and fine-tuned evaluations
- Analyze and compare results
- Determine the best-performing method
- Save the results to the `results` directory

### Evaluation Options

```bash
# Run specific methods only
python evaluate_methods.py --methods zero-shot few-shot

# Use a subset of samples for faster testing
python evaluate_methods.py --sample_size 5

# Specify data and output directories
python evaluate_methods.py --data_dir ../data --output_dir ../results
```

The fine-tuning process will automatically detect whether to use the multi-turn conversation format based on the dataset.

## Agent Implementation

After evaluation, run the agent with the best-performing model:

```bash
cd implementation
python run_agent.py --model "ft:gpt-4o-mini:org:your-model-id"
```

OR

```bash
export FINE_TUNED_MODEL_NAME="ft:gpt-4o-mini:org:your-model-id"
python run_agent.py
```

This will:

- Load the best model information from evaluation results
- Start an interactive interface for spatial analysis
- Allow natural language queries about Cambridge parcels
- Generate visualizations of the results

### Implementation Options

```bash
# Specify a model to use
python run_agent.py --model "gpt-4o"

# Use terminal interface instead of web interface
python run_agent.py --interface terminal

# Specify data and results directories
python run_agent.py --data_dir ../data --results_dir ../results
```

## Example Queries

The agent can handle queries like:

- "Find commercial parcels within 500 meters of Harvard Square"
- "Find parcels larger than 6000 square feet that are zoned for retail use"
- "Find parcels with no more than 2 competing restaurants within 800 meters"
- "Find the top 20 parcels with the highest consumer spending in surrounding areas"

## Evaluation Metrics

The evaluation process measures:

- **Exact Match**: Binary accuracy (1 if all parcels match, 0 otherwise)
- **Result Match Rate**: Percentage of correctly identified parcels
- **Precision**: Ratio of correctly identified parcels to total generated parcels
- **Recall**: Ratio of correctly identified parcels to ground truth parcels
- **F1 Score**: Harmonic mean of precision and recall

## Agent Framework

The agent implementation uses:

- **Multi-turn Conversations**: Natural dialog flow with the user
- **Function Calling**: Generate and execute Python code
- **Visualization**: Create maps showing the matching parcels
- **Web or Terminal Interface**: Choose the interface that fits your needs

## Notes

- Fine-tuning requires OpenAI API access and may incur costs
- Visualization requires matplotlib and the ability to save files locally
- The web interface requires Streamlit to be installed
