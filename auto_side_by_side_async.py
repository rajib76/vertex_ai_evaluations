import os

from google.cloud import aiplatform

PROJECT_ID = "gen-lang-client-0172427287"
LOCATION = "us-central1"

# ... your AUTORATER_PROMPT_PARAMETERS + parameters ...

AUTORATER_PROMPT_PARAMETERS= {
    'inference_instruction': {
        'column': 'question'
    },
    'inference_context': {
        'column': 'context'
    }
}
parameters = {
    'evaluation_dataset': "gs://auto_side_by_side/s_x_s_dataset.jsonl",
    'id_columns': ['id'],
    'task': 'question_answering@latest',
    'autorater_prompt_parameters': AUTORATER_PROMPT_PARAMETERS,
    'response_column_a': 'model_a_response',
    'response_column_b': 'model_b_response',
    # 'model_a': 'MODEL_A',
    # 'model_a_prompt_parameters': MODEL_A_PROMPT_PARAMETERS,
    # 'model_b': 'MODEL_B',
    # 'model_b_prompt_parameters': MODEL_B_PROMPT_PARAMETERS,
    'judgments_format': 'jsonl',
    # 'bigquery_destination_prefix':
    # BIGQUERY_DESTINATION_PREFIX,
}

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket="gs://auto_side_by_side")

display_name= f'auto_side_by_side_qna_pipe-22'
job = aiplatform.PipelineJob(
    display_name=display_name,
    pipeline_root=os.path.join("gs://auto_side_by_side", display_name),
    template_path="https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/autosxs-template/default",
    parameter_values=parameters,
    enable_caching=False,
)

# Run async (method unblocks). :contentReference[oaicite:5]{index=5}
job.run()
# print(f"Started: {job.resource_name}")
#
# terminal = {
#     PipelineState.PIPELINE_STATE_SUCCEEDED,
#     PipelineState.PIPELINE_STATE_FAILED,
#     PipelineState.PIPELINE_STATE_CANCELLED,
# }
#
# poll_s = 30
# while True:
#     # Re-fetch latest server-side state
#     latest = aiplatform.PipelineJob.get(job.resource_name)
#     state = latest.state
#     state_name = state.name if hasattr(state, "name") else str(state)
#
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] state={state_name}")
#
#     # Terminal states per PipelineState enum. :contentReference[oaicite:6]{index=6}
#     if state in terminal or state_name in {s.name for s in terminal}:
#         job = latest
#         break
#
#     time.sleep(poll_s)
#
# if (job.state.name if hasattr(job.state, "name") else str(job.state)) != PipelineState.PIPELINE_STATE_SUCCEEDED.name:
#     raise RuntimeError(f"Pipeline ended in {job.state}. Error: {getattr(job, 'error', None)}")
#
# print("Pipeline succeeded.")
