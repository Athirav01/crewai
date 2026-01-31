#!/usr/bin/env python
import os
import sys
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Langfuse instrumentation
from langfuse import get_client
from openinference.instrumentation.crewai import CrewAIInstrumentor

lf = get_client()
CrewAIInstrumentor().instrument(skip_dep_check=True)

from crewai_hierarchical.crew import TestLlm

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    inputs = {
        "topic": "Machine Learning Model Evaluation",
        "current_year": str(datetime.now().year)
    }

    try:
        with lf.start_as_current_observation(
            as_type="span",
            name="TestLlm-run"
        ):
            TestLlm().crew().kickoff(inputs=inputs)
        lf.flush()
    except Exception as e:
        raise Exception(f"Error running crew: {e}")


def train():
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        TestLlm().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"Error during training: {e}")


def replay():
    try:
        TestLlm().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"Error during replay: {e}")


def test():
    inputs = {
        "topic": "Machine Learning Model Evaluation",
        "current_year": str(datetime.now().year)
    }

    try:
        TestLlm().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"Error during testing: {e}")


def run_with_trigger():
    import json

    if len(sys.argv) < 2:
        raise Exception("Trigger payload missing")

    trigger_payload = json.loads(sys.argv[1])

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        return TestLlm().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"Error running with trigger: {e}")
