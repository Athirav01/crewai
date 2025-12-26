#!/usr/bin/env python
import os
from dotenv import load_dotenv

# Load the .env file in your CrewAI project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Now environment variables are available
from langfuse import get_client
from openinference.instrumentation.crewai import CrewAIInstrumentor

lf = get_client()
CrewAIInstrumentor().instrument(skip_dep_check=True)




import sys
import warnings

from datetime import datetime

from test_llm.crew import TestLlm

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    inputs = {
        'topic': 'Machine Learning Model Evaluation',
        'current_year': str(datetime.now().year)
    }

    try:
        # Wrap the kickoff in a Langfuse observation
        with lf.start_as_current_observation(as_type="span", name="TestLlm-run"):
            TestLlm().crew().kickoff(inputs=inputs)
        lf.flush()  # send telemetry to Langfuse
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")



def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        TestLlm().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        TestLlm().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "Machine Learning Model Evaluation",
        "current_year": str(datetime.now().year)
    }

    try:
        TestLlm().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = TestLlm().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
