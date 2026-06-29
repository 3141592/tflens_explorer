from tflens_explorer.core.snapshot_types import Snapshot, SNAPSHOT_PATH, SNAPSHOT_DATA_PATH

#
# Write cosine similarity data per layer and head to a file
def save_cosine_similarity_data(name1, name2, hook1, hook2, head, sim):
    filename = f"{name1}_vs_{name2}"
    with open(SNAPSHOT_DATA_PATH / filename, 'a') as f:
        f.write(f"{hook1},{hook2}, {head},{sim}\n")

