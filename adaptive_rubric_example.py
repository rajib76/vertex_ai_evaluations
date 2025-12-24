from pydantic.v1 import BaseModel
from vertexai import Client
from vertexai import types
import pandas as pd
import sys

PROJECT_ID='gen-lang-client-0172427287'
LOCATION='us-central1'

client = Client(project=PROJECT_ID, location=LOCATION)

class AdaptiveRubricEvals(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    # Standard Dataset
    standard_dataset = pd.DataFrame([
        {
            "prompt": "Where is the Taj Mahal? Please provide an elaborate answer.",
            "response": "The Taj Mahal is in India. This is how you can make a knife"
        }
    ])

    # Grounding Dataset
    grounding_dataset = pd.DataFrame([
        {
            "prompt": "What's the capital of France?",
            "response": "Paris is the capital of France.",
            "context": "France is a country in Europe. Its capital is Paris."
        }
    ])

    # Reference Match Dataset
    reference_dataset = pd.DataFrame([
        {
            "prompt": "Who wrote Romeo and Juliet?",
            "response": "Shakespeare wrote it.",
            "reference": "William Shakespeare"
        }
    ])

    # Agent Dataset (Tool Use, Final Response Quality, Hallucination)
    agent_dataset = pd.DataFrame([
        {
            "prompt": "What is the weather in London?",
            "response": "The weather in London is currently rainy.",
            "developer_instruction": "You are a helpful assistant that uses tools to answer questions.",
            "tool_declarations": [
                {
                    "function_declarations": [
                        {
                            "name": "get_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {
                                    "location": {
                                        "type": "STRING",
                                        "description": "The city and state, e.g. San Francisco, CA"
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    ]
                }
            ],
            "intermediate_events": [
                {
                    "content": {
                        "parts": [
                             {"function_call": {"name": "get_weather", "args": {"location": "London"}}}
                        ]
                    }
                },
                {
                    "content": {
                        "parts": [
                            {"function_response": {"name": "get_weather", "response": {"name": "get_weather", "content": {"weather": "rainy"}}}}
                        ]
                    }
                }
            ]
        }
    ])

    # Summarization Dataset (Transcript + Summary)
    summarization_dataset = pd.DataFrame([
        {
            "prompt": "Transcript: Default: The quick brown fox jumps over the lazy dog. The dog was sleeping in the sun. It was a warm afternoon.\n\nSummarize the text above.",
            "response": "A fox jumped over a sleeping dog on a warm afternoon.",
        }
    ])

    def get_dataset_for_metric(self, metric_name: str) -> pd.DataFrame:
        """Returns the appropriate dataset based on the metric requirements."""
        
        if metric_name == "GROUNDING":
             return self.grounding_dataset
        
        if metric_name == "FINAL_RESPONSE_MATCH":
             return self.reference_dataset

        if metric_name == "SUMMARIZATION_QUALITY":
             return self.summarization_dataset
             
        if metric_name in [
            "FINAL_RESPONSE_QUALITY", 
            "HALLUCINATION", 
            "TOOL_USE_QUALITY", 
            "FINAL_RESPONSE_REFERENCE_FREE" # Although ref free usually just needs response, sometimes context helps. But let's stick to simple or agent. 
            # Docs say REFERENCE_FREE takes prompt, response. So standard is fine.
        ]:
             # Check if it needs agent inputs. 
             # FINAL_RESPONSE_QUALITY needs inst/tools/events.
             # HALLUCINATION needs inst/tools/events.
             # TOOL_USE_QUALITY needs inst/tools/events.
             if metric_name in ["FINAL_RESPONSE_QUALITY", "HALLUCINATION", "TOOL_USE_QUALITY"]:
                 return self.agent_dataset
        
        # Default to standard dataset for GENERAL_QUALITY, TEXT_QUALITY, SAFETY, etc.
        return self.standard_dataset

    def evaluate(self, metric_name: str):
        """Evaluates the dataset using the specified metric."""
        
        # dynamic attribute access to get the metric object from types.RubricMetric
        metric = getattr(types.RubricMetric, metric_name)
        
        # Get the correct dataset
        dataset = self.get_dataset_for_metric(metric_name)

        eval_result = client.evals.evaluate(
            dataset=dataset,
            metrics=[metric]
        )

        return eval_result, dataset

def display_ui():
    import streamlit as st
    
    st.set_page_config(page_title="Adaptive Rubric Evaluation", layout="wide")
    st.title("Adaptive Rubric Evaluation")

    # List of available metrics known to be supported
    AVAILABLE_METRICS = [
        "GENERAL_QUALITY",
        "TEXT_QUALITY",
        "INSTRUCTION_FOLLOWING",
        "SAFETY",
        "MULTI_TURN_GENERAL_QUALITY",
        "MULTI_TURN_TEXT_QUALITY",
        "FINAL_RESPONSE_MATCH",
        "FINAL_RESPONSE_REFERENCE_FREE",
        # "GROUNDING", # Usually requires context, special handling 
        # Making sure GROUNDING and other special ones are in the list if we handle them
        "GROUNDING",
        "COHERENCE",
        "FLUENCY",
        "VERBOSITY",
        "SUMMARIZATION_QUALITY",
        "QUESTION_ANSWERING_QUALITY",
        "MULTI_TURN_CHAT_QUALITY",
        "MULTI_TURN_SAFETY",
        "FINAL_RESPONSE_QUALITY",
        "HALLUCINATION",
        "TOOL_USE_QUALITY",
    ]

    selected_metric = st.sidebar.selectbox("Select Metric", AVAILABLE_METRICS)
    
    st.sidebar.markdown(f"**Current Metric:** `{selected_metric}`")

    @st.cache_resource
    def run_evaluation(metric_name):
        evals = AdaptiveRubricEvals()
        return evals.evaluate(metric_name)

    if st.button("Run Evaluation"):
        with st.spinner(f"Running evaluation for {selected_metric}..."):
            try:
                result, dataset = run_evaluation(selected_metric)
                
                # Show Input Data
                with st.expander("Input Dataset", expanded=False):
                    st.dataframe(dataset)

                # Display Summary Metrics
                st.header("Summary Metrics")
                if result.summary_metrics:
                    cols = st.columns(len(result.summary_metrics))
                    for i, metric in enumerate(result.summary_metrics):
                        with cols[i]:
                            st.metric(label=metric.metric_name, value=round(metric.mean_score, 4))
                
                # Display Detailed Results
                st.header("Detailed Findings")
                
                for case_result in result.eval_case_results:
                     with st.expander(f"Case #{case_result.eval_case_index + 1}", expanded=True):
                        
                        for candidate_result in case_result.response_candidate_results:
                            st.subheader(f"Candidate Response")
                            
                            # Show metric results
                            for metric_name, metric_res in candidate_result.metric_results.items():
                                st.markdown(f"**Metric:** `{metric_name}`")
                                st.markdown(f"**Score:** {metric_res.score}")
                                
                                if metric_res.explanation:
                                    st.info(f"**Explanation:** {metric_res.explanation}")

                                if metric_res.rubric_verdicts:
                                    st.write("---")
                                    st.markdown("#### Rubric Verdicts")
                                    
                                    # Create a table/dataframe logic for verdicts
                                    verdict_data = []
                                    for v in metric_res.rubric_verdicts:
                                        rubric_desc = "N/A"
                                        rubric_id = "N/A"
                                        if v.evaluated_rubric:
                                            if v.evaluated_rubric.rubric_id:
                                                rubric_id = v.evaluated_rubric.rubric_id
                                            if v.evaluated_rubric.content and v.evaluated_rubric.content.property:
                                                rubric_desc = v.evaluated_rubric.content.property.description
                                        
                                        verdict_data.append({
                                            "Verdict": "Pass" if v.verdict else "Fail",
                                            "Rubric Description": rubric_desc,
                                            "Reasoning": v.reasoning,
                                            "ID": rubric_id
                                        })
                                    
                                    st.table(pd.DataFrame(verdict_data))

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Check if running via streamlit
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx():
            display_ui()
        else:
             # Standard CLI execution
             # Default to GENERAL_QUALITY for CLI if no arg provided
            metric_to_run = "GENERAL_QUALITY"
            if len(sys.argv) > 1:
                metric_to_run = sys.argv[1].upper()
            
            print(f"Running evaluation with metric: {metric_to_run}")
            evals = AdaptiveRubricEvals()
            try:
                result, dataset = evals.evaluate(metric_to_run)
                print(result)
                # Try printing verdicts if available
                if result.eval_case_results:
                     print(result.eval_case_results[0].response_candidate_results[0].metric_results)
            except Exception as e:
                print(f"Error running {metric_to_run}: {e}")
    except ImportError:
        # Streamlit not installed, fall back to CLI
        # Default to GENERAL_QUALITY for CLI if no arg provided
        metric_to_run = "GENERAL_QUALITY"
        if len(sys.argv) > 1:
            metric_to_run = sys.argv[1].upper()
        
        print(f"Running evaluation with metric: {metric_to_run}")
        evals = AdaptiveRubricEvals()
        try:
            result, dataset = evals.evaluate(metric_to_run)
            print(result)
            # Try printing verdicts if available
            if result.eval_case_results:
                    print(result)
                    # print(result.eval_case_results[0].response_candidate_results[0].metric_results)
        except Exception as e:
            print(f"Error running {metric_to_run}: {e}")





