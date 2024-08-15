# Text Summarization using Python

## Setup and Usage

1. **Create and Activate a Virtual Environment**: Navigate to your project directory and run the following commands:

    ```bash
    # Create a virtual environment
    python -m venv summarizerenv

    # Activate the virtual environment
    # On Windows
    summarizerenv\Scripts\activate

    # On macOS/Linux
    source summarizerenv/bin/activate
    ```

2. **Install Dependencies**: With the virtual environment activated, install the required packages using:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Your Input File**: Create your input text file and place it in the project root directory.

4. **Configure the Project**: Edit `config.yaml` to customize the summarization process. You can:
    - **Choose Algorithms**: Select from NLTK, SUMY, BERT Extractive, or BART Abstractive.
    - **Specify Input File**: Define the name of your input text file.
    - **Configure Word Lists**: Set lists for bonus, stigma, and null words to guide the summarization.
    - **Select Specific Algorithms or Models**: Choose specific SUMY algorithms or BART models.
    - **Set Summary Lengths**: Define summary lengths as a percentage of the source text or a specific number of sentences.
    - **Optional Length Constraints**: Optionally set minimum and maximum summary lengths.

5. **Run the Main Script**: Execute the script to generate a summary based on your configuration:

    ```bash
    python main.py
    ```

