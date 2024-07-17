def get_val_folder(n_fold, max_fold, fast_run):

    if fast_run:
        val_fold = 2
        # print(f"Validation folder: {val_fold}")
        return val_fold

    val_fold = (n_fold + 1) % max_fold

    if val_fold == 0:
        val_fold = 1

    # print(f"Validation folder: {val_fold}")
    return val_fold
