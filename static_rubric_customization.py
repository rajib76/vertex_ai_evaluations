from pydantic.v1 import BaseModel
from vertexai import Client
from vertexai import types
from vertexai._genai.types import MetricPromptBuilder, LLMMetric
import pandas as pd
import sys

PROJECT_ID='gen-lang-client-0172427287'
LOCATION='us-central1'

client = Client(project=PROJECT_ID, location=LOCATION)


class StaticRubricEvals(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    # Standard Dataset
    standard_dataset = pd.DataFrame([
        {
            "prompt": "Where does Rajib Live.",
            "response": "Rajib has 27 years of experience.",
            "baseline_model_response": "Rajib lives in San Jose.",
            "context" : "Rajib is an architect with expertise in data and engineering space. He has 27 years of experience and lives in San Jose"
        }
    ])

    # Define a custom metric to evaluate relevance
    relevance_metric = types.LLMMetric(
        name='context_relevance',
        prompt_template=MetricPromptBuilder(
            instruction="Evaluate the response against the context and calculate the relevance score.",
            criteria={
                "accuracy": "must be grounded to the facts in the context.",
                "comprehensiveness": "answer must be comprehensive and detailed",
            },
            rating_scores={
                "3": "Excellent: accurate and grounded to the context, comprehensive and detailed.",
                "2": "Good: accurate and grounded to the context, but some of the information may be missing",
                "1": "Poor: inaccurate and not grounded to the context",
            }
        )
    )

    def evaluate(self, metric_name: str = "context_relevance"):
        """Evaluates the dataset using the specified metric."""
        
        if metric_name == "context_relevance":
             metric = self.relevance_metric
        else:
             metric = getattr(types.RubricMetric, metric_name)
             
        # In this static example, we primarily use standard_dataset
        dataset = self.standard_dataset

        eval_result = client.evals.evaluate(
            dataset=dataset,
            metrics=[metric]
        )
        return eval_result, dataset

def display_ui():
    import streamlit as st
    
    st.set_page_config(page_title="Static Rubric Customization", layout="wide")
    st.title("Static Rubric Customization")

    # List of available metrics
    AVAILABLE_METRICS = [
        "context_relevance"
    ]

    selected_metric = st.sidebar.selectbox("Select Metric", AVAILABLE_METRICS)
    
    st.sidebar.markdown(f"**Current Metric:** `{selected_metric}`")

    @st.cache_resource
    def run_evaluation(metric_name):
        evals = StaticRubricEvals()
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
                
                # Handle Pairwise Summary vs Pointwise Summary
                if hasattr(result, 'summary_metrics') and result.summary_metrics: # Pointwise
                     cols = st.columns(len(result.summary_metrics))
                     for i, metric in enumerate(result.summary_metrics):
                            with cols[i]:
                                st.metric(label=metric.metric_name, value=round(metric.mean_score, 4))
                else: 
                     # Pairwise summary is usually not in summary_metrics structured the same way or might need calc
                     # Vertex AI PairwiseMetric usually populates summary_metrics with win rates if available
                     # But let's check what is available on result object or just rely on detailed table
                     pass


                # Display Detailed Results
                st.header("Detailed Findings")
                
                # If result has eval_case_results (generic)
                if hasattr(result, 'eval_case_results') and result.eval_case_results:
                     for case_result in result.eval_case_results:
                         with st.expander(f"Case #{case_result.eval_case_index + 1}", expanded=True):
                            
                            for candidate_result in case_result.response_candidate_results:
                                st.subheader(f"Candidate Response")
                                
                                # Check metric results
                                if candidate_result.metric_results:
                                    for metric_name, metric_res in candidate_result.metric_results.items():
                                        st.markdown(f"**Metric:** `{metric_name}`")
                                        
                                        # Pointwise Score
                                        if hasattr(metric_res, 'score'):
                                            st.markdown(f"**Score:** {metric_res.score}")
                                        
                                        # Pairwise Choice
                                        if hasattr(metric_res, 'pairwise_choice'):
                                            st.markdown(f"**Pairwise Choice:** {metric_res.pairwise_choice}")

                                        if hasattr(metric_res, 'explanation') and metric_res.explanation:
                                            st.info(f"**Explanation:** {metric_res.explanation}")

                                        # Rubric Verdicts (Pointwise)
                                        if hasattr(metric_res, 'rubric_verdicts') and metric_res.rubric_verdicts:
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
                
                # PairwiseMetric might return results differently depending on SDK version?
                # The generic logic above using eval_case_results should handle both if attributes exist.
                # However, Pairwise results might also be accessible via metrics_table on the result object.
                if hasattr(result, 'metrics_table'):
                    st.subheader("Metrics Table")
                    st.dataframe(result.metrics_table)

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
            metric_to_run = "context_relevance"
            if len(sys.argv) > 1:
                metric_to_run = sys.argv[1]
            
            print(f"Running evaluation with metric: {metric_to_run}")
            evals = StaticRubricEvals()
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
        metric_to_run = "context_relevance"
        if len(sys.argv) > 1:
            metric_to_run = sys.argv[1]
        
        print(f"Running evaluation with metric: {metric_to_run}")
        evals = StaticRubricEvals()
        try:
            result, dataset = evals.evaluate(metric_to_run)
            print(result)
            if result.eval_case_results:
                     print(result.eval_case_results[0].response_candidate_results[0].metric_results)
        except Exception as e:
            print(f"Error running {metric_to_run}: {e}")