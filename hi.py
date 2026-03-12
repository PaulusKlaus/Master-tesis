from itertools import combinations_with_replacement

augmentations = ['gaussian', 'normal', 'scale', 'randomstrech', 'randomcrop', 'fft']

pairs = list(combinations_with_replacement(augmentations, 2))

print(len(pairs))  # 21
print(pairs)


def augmentation_test(paths):
    all_df = []

    for path in paths:
        df = parse_training_log(path)
        df["model"] = path.split("/")[-2]  # keep experiment id
        all_df.append(df)

    all_df = pd.concat(all_df, ignore_index=True)

    # Option A (recommended for SSL): treat (aug_1, aug_2) same as (aug_2, aug_1)
    # -> sort the pair so it's order-invariant
    all_df[["aug_a", "aug_b"]] = np.sort(all_df[["aug_1", "aug_2"]].values, axis=1)

    summary_aug = (
        all_df
        .groupby(["aug_a", "aug_b"], as_index=False)
        .agg(
            n_runs=("test_acc", "count"),
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            mean_loss=("test_loss", "mean"),
            std_loss=("test_loss", "std"),
        )
        .sort_values("mean_acc", ascending=False)
    )

    print(summary_aug.to_string(index=False))
