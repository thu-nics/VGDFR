def similarity_threhshold_schedule(
    image_reshape, ssim_results1, ssim_results2, ssim_results3, similarity_threshold, enable_4x4merge=False
):
    merge2x4_inds = []
    merge4x4_inds = []
    latent_remain_inds = [_ for _ in range(len(image_reshape) // 4 + 1)]
    video_remain_inds = [_ for _ in range(len(image_reshape))]
    merge_plan = []
    ind = 1

    while ind < len(image_reshape) - 8 - 1:  # gurantee the last frame cannot be merged
        # 4x4 merge
        if enable_4x4merge and ind < len(image_reshape) - 16 - 1:
            min_ssim = min(
                [
                    min(
                        ssim_results1[ind + offset],
                        ssim_results2[ind + offset],
                        ssim_results3[ind + offset],
                        ssim_results1[ind + offset + 1],
                        ssim_results2[ind + offset + 1],
                        ssim_results1[ind + offset + 2],
                    )
                    for offset in [0, 4, 8, 12]
                ]
            )
            # print(f"4x4 merge min_ssim: {min_ssim}")
            if min_ssim > similarity_threshold:
                merge4x4_inds.append(ind)
                for offset in [0, 4, 8, 12]:
                    merge_plan.append([ind + offset, ind + offset + 1, ind + offset + 2, ind + offset + 3])
                    video_remain_inds.remove(ind + offset + 1)
                    video_remain_inds.remove(ind + offset + 2)
                    video_remain_inds.remove(ind + offset + 3)
                for offset in [4, 8, 12]:
                    latent_remain_inds.remove(1 + (ind + offset) // 4)
                ind += 16
                continue
        # 2x4 merge
        min_ssim = min([ssim_results1[ind + offset] for offset in [0, 2, 4, 6]])
        # print(f"2x4 merge min_ssim: {min_ssim}")
        if min_ssim > similarity_threshold:
            merge2x4_inds.append(ind)
            for offset in [0, 2, 4, 6]:
                merge_plan.append([ind + offset, ind + offset + 1])
                video_remain_inds.remove(ind + offset + 1)
            for offset in [4]:
                latent_remain_inds.remove(1 + (ind + offset) // 4)
        ind += 8
    return merge_plan, merge2x4_inds, merge4x4_inds, latent_remain_inds, video_remain_inds


def keep_token_ratio_schedule(image_reshape, ssim_results1, ssim_results2, ssim_results3, keep_token_ratio):
    # binary search
    # keep_token_ratio is len(video_remain_inds)/len(image_reshape)
    left_similarity_threshold = 0.01
    right_similarity_threshold = 0.99
    while left_similarity_threshold + 0.001 < right_similarity_threshold:
        similarity_threshold = (left_similarity_threshold + right_similarity_threshold) / 2
        # print(f"Test similarity_threshold: {similarity_threshold}")
        merge_plan, merge2x4_inds, merge4x4_inds, latent_remain_inds, video_remain_inds = (
            similarity_threhshold_schedule(
                image_reshape,
                ssim_results1,
                ssim_results2,
                ssim_results3,
                similarity_threshold,
                enable_4x4merge=similarity_threshold < 0.5,
            )
        )
        if len(video_remain_inds) / len(image_reshape) < keep_token_ratio:
            left_similarity_threshold = similarity_threshold
            print(
                f"Now status: {len(video_remain_inds) / len(image_reshape)} < {keep_token_ratio}, left_similarity_threshold: {left_similarity_threshold}"
            )
        else:
            right_similarity_threshold = similarity_threshold
            print(
                f"Now status: {len(video_remain_inds) / len(image_reshape)} > {keep_token_ratio}, right_similarity_threshold: {right_similarity_threshold}"
            )
    # print("left_similarity_threshold: ", left_similarity_threshold)
    # print("right_similarity_threshold: ", right_similarity_threshold)
    similarity_threshold = (left_similarity_threshold + right_similarity_threshold) / 2
    if similarity_threshold < 0.5:
        enable_4x4merge = True
        print("Enable 4x4 merge because similarity_threshold < 0.5")
    else:
        enable_4x4merge = False
    merge_plan, merge2x4_inds, merge4x4_inds, latent_remain_inds, video_remain_inds = similarity_threhshold_schedule(
        image_reshape,
        ssim_results1,
        ssim_results2,
        ssim_results3,
        similarity_threshold,
        enable_4x4merge=enable_4x4merge,
    )
    return merge_plan, merge2x4_inds, merge4x4_inds, latent_remain_inds, video_remain_inds
