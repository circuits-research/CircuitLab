import torch
import os
from circuitlab.attribution.attribution import AttributionRunner
import circuitlab
STORAGE_ROOT = Path(circuitlab.__file__).resolve().parents[2] / "storage" # symlink to scratch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# should run on a single GPU

def main():
    MODEL = "gpt2"
    clt_checkpoint = STORAGE_ROOT / "checkpoints" / MODEL / "d1s3fw30/middle_22137856"

    test_strings = [
        'The opposite of "large" is "'
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # where to save the attribution_graph.pt
    folder_name = str(STORAGE_ROOT / "attribution" / MODEL)

    print("Creating Attribution Runner")
    runner = AttributionRunner(
        clt_checkpoint=clt_checkpoint,
        model_name=MODEL,
        device=device,
        debug=True,
    )

    for i, test_string in enumerate(test_strings, 1):
        print(f"\n Processing test string {i}: '{test_string}'")

        try:
            _ = runner.run(
                input_string=test_string,
                folder_name=folder_name,
                max_n_logits=1,
                desired_logit_prob=0.95,
                max_feature_nodes=1_000,
                batch_size=256,
                feature_threshold=0.85,
                edge_threshold=0.98,
                offload=None,
                run_interventions=True,
            )

        except Exception as e:
            print(f"Attribution computation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
