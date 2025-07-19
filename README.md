## CDeeply Experiments
Experiments on Cdeeply tool.


### Development Setup
1. Create and activate a virtual environment:
    ```sh
   python3 -m venv venv

   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Running Application
3. Start the application
    ```sh
    python3 predictions-task.py \
    --mode columns \
    --targets 3,5 \
    --max_layers 2 \
    --max_neurons 32 \
    --activation tanh \
    --has_bias \
    --quantize

    ```

### Folder Structure
```sh
project/
├── predictions-task.py         # Main script to train and evaluate the model
├── run_benchmarks.py           # Automation script for running experiments
├── benchmark_results.csv       # Collected metrics from all benchmark runs
├── benchmark_dashboard.ipynb   # Jupyter notebook to visualize results
└── README.md                   # Project overview and usage instructions
```