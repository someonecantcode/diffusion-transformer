from datasets import load_dataset

data = load_dataset("stanford-vision-lab/gpic", split="val", data_files={"val": "val/*.tar"})
data = data.filter(
    lambda ex: ex["jpg"] is not None
               and ex["json"]["caption"] is not None
)
data.save_to_disk("./filtered")