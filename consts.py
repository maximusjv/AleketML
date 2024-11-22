CLASSES_TO_NUM = {"background": 0, "healthy": 1, "necrosed": 2}
NUM_TO_CLASSES = {0: "background", 1: "healthy", 2: "necrosed"}
VALIDATION_METRICS = ["AP", "Recall", "Precision", "F1", "ACD", "AAD"]
LOSSES_NAMES = [
    "loss",
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
]
PRIMARY_VALIDATION_METRIC = "AAD"
def collate_fn(batch):
    """Collates data samples into batches for the dataloader."""
    return tuple(zip(*batch))
