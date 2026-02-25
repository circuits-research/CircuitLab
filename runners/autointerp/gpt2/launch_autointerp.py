# # # Fix filelock issue before any other imports
# # import filelock
# # filelock.FileLock = filelock.SoftFileLock

# # Set all environment variables FIRST, before ANY imports

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# # os.environ["TOKENIZERS_PARALLELISM"] = "false"
# # os.environ["VLLM_USE_MODELSCOPE"] = "0"
# # os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
# # os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
# # os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
# # os.environ["FILELOCK_BACKEND"] = "SoftFileLock"

# import sys
# import torch
# from circuitlab.config.autointerp_config import AutoInterpConfig  
# from circuitlab.autointerp.pipeline import AutoInterp

# # Check CUDA before any other imports
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA device count: {torch.cuda.device_count()}")

# def main():
#     """Main function to run autointerp with proper multiprocessing support."""
#     d_in = 768
#     expansion_factor = 32
#     d_latent = expansion_factor * d_in

#     layer = int(sys.argv[1])
#     job_id = int(sys.argv[2]) 
#     total_jobs = int(sys.argv[3])
#     chunk_id = int(sys.argv[4])

#     clt_path = "/fast/fdraye/data/featflow/cache/checkpoints/gpt2/d1s3fw30/middle_22137856"

#     autointerp_cfg = { 
#         "device": "cpu",
#         "model_name": "gpt2",
#         "clt_path": clt_path,
#         "latent_cache_path": "/fast/fdraye/data/featflow/cache/gpt2", # where to load
#         "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
#         "context_size": 16,
#         "total_autointerp_tokens": 12*(16*4096), #2*(60*4096)
#         "vllm_model": "google/gemma-3-12b-it",
#         "train_batch_size_tokens": 4096,
#         "n_batches_in_buffer": 32,
#         "store_batch_size_prompts": 32, 
#         "d_in": 768, 
#         "n_chunks": 12 #36
#     }

#     print("Creating AutoInterpConfig...", flush=True)
#     autointerp_config = AutoInterpConfig(**autointerp_cfg)
#     print("Initializing AutoInterp...", flush=True)
#     autointerp = AutoInterp(autointerp_config)
#     print("AutoInterp initialized successfully!", flush=True)

#     # Calculate feature index list for this job
#     features_per_job = d_latent // total_jobs
#     start_idx = job_id * features_per_job 
#     end_idx = start_idx + features_per_job
#     index_list = list(range(start_idx, end_idx))
    
#     print(f"Job {job_id}/{total_jobs}: Processing layer {layer}, features {start_idx}-{end_idx-1} ({len(index_list)} features)", flush=True)

#     # # # chunk_list = list(range(2))
#     # chunk_list = [chunk_id]
#     # print("Generate Cache", flush=True)
#     # autointerp.run(chunk_list)

#     # index_list = list(range(10))

#     # # Once you have run the cache, you can run the next three functions with queue 96 or less. If you have cuda memory issues when running the explanation, run it seperatly. 
#     # print("Running prompt generation", flush=True)
#     # autointerp.generate_prompts_for_layer(layer=layer, top_k=100, index_list=index_list)

#     # index_list = list(range(10))
 
#     # print("Running explanation generation", flush=True)
#     # autointerp.generate_explanations_from_prompts(layer=layer, index_list=index_list)
    
#     print("Running dictionary generation", flush=True)
#     autointerp.generate_feature_dictionaries(layer=layer, index_list=index_list)

#     # autointerp.save_feature_frequency()
                                             
# if __name__ == "__main__":
#     main()
