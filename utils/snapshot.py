import torch


def save_snapshot(file, epoch, last_score, best_score, global_step, **kwargs):
    data = {
        "state_dict": dict(kwargs),
        "training_meta": {
            "epoch": epoch,
            "last_score": last_score,
            "best_score": best_score,
            "global_step": global_step
        }
    }
    torch.save(data, file)


def resume_from_snapshot(model, snapshot):
    snapshot = torch.load(snapshot, map_location="cpu")

    state_dict = snapshot["state_dict"]["model"]

    model.load_state_dict(state_dict, False)

    return snapshot