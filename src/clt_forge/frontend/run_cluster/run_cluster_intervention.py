# #!/usr/bin/env python3

# import sys
# import os
# import json

# # Add the src directory to Python path
# sys.path.insert(0, '/lustre/home/fdraye/projects/featflow/src')
# #
# from circuit_tracing_visual_interface.attribution.intervention import run_cluster_intervention

# def main():
#     # Get parameters from command line arguments or environment variables
#     if len(sys.argv) < 2:
#         print("Usage: python run_cluster_intervention.py <args_json_file>")
#         sys.exit(1)
    
#     args_file = sys.argv[1]  # Path to JSON file with arguments
    
#     # Read arguments from JSON file
#     try:
#         with open(args_file, 'r') as f:
#             args_data = json.load(f)
        
#         cluster_features = args_data["cluster_features"]
#         manual_features = args_data["manual_features"]
#         input_string = args_data["input_string"]
#         cluster_intervention_value = float(args_data["cluster_intervention_value"])
#         freeze_attention = args_data.get("freeze_attention", True)  # Default to True for backwards compatibility
        
#     except Exception as e:
#         print(f"Error reading arguments file {args_file}: {e}")
#         sys.exit(1)
    
#     # Get config from environment variables (matching frontend config)
#     clt_checkpoint = os.getenv("CLT_CHECKPOINT", "/home/fdraye/projects/featflow/checkpoints/gpt2_multilingual_20/zhb8w33x/final_17478656")
#     model_name = os.getenv("MODEL_NAME", "CausalNLP/gpt2-hf_multilingual-20")
#     model_class_name = os.getenv("MODEL_CLASS_NAME", "HookedTransformer")
    
#     print(f"Running intervention on {len(cluster_features)} cluster features and {len(manual_features)} manual features...")
#     print(f"Input: '{input_string}'")
#     print(f"Cluster intervention value: {cluster_intervention_value}")
#     print(f"Freeze attention: {freeze_attention}")
    
#     try:
#         result = run_cluster_intervention(
#             cluster_features=cluster_features,
#             manual_features=manual_features,
#             clt_checkpoint=clt_checkpoint,
#             input_string=input_string,
#             cluster_intervention_value=cluster_intervention_value,
#             model_name=model_name,
#             top_tokens_count=4,
#             device="cuda",
#             freeze_attention=freeze_attention
#         )
        
#         print("\nIntervention Results:")
#         print(f"Top tokens: {result['tokens']}")
#         print(f"Probabilities: {[f'{p:.4f}' for p in result['probabilities']]}")
        
#         # Save results to file in the run_cluster folder
#         output_file = "/lustre/home/fdraye/projects/featflow/src/featflow/frontend/run_cluster/cluster_intervention_results.json"
#         with open(output_file, 'w') as f:
#             json_result = {
#                 'tokens': result['tokens'],
#                 'probabilities': [float(p) for p in result['probabilities']],
#                 'baseline_probabilities': [float(p) for p in result.get('baseline_probabilities', [])],
#                 'probability_differences': [float(p) for p in result.get('probability_differences', [])],
#                 'cluster_features': cluster_features,
#                 'manual_features': manual_features,
#                 'cluster_feature_count': result.get('cluster_feature_count', len(cluster_features)),
#                 'manual_feature_count': result.get('manual_feature_count', len(manual_features)),
#                 'input_string': input_string,
#                 'cluster_intervention_value': cluster_intervention_value
#             }
#             json.dump(json_result, f, indent=2)
        
#         print(f"\nResults saved to {output_file}")
        
#     except Exception as e:
#         print(f"Error during intervention: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
