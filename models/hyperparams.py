
dataset_hyperparams = {
    "FFHQ64": {
        "patch_size": [2, 2],
        "depths": [2, 2, 8],
        "widths": [128 * 3, 256 * 3, 512 * 3],
        "joint": [True, True, True,],
        "self_attns": [
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "global", "d_head": 64},
        ],
    },
    "FFHQ256": {
        "patch_size": [4, 4],
        "depths": [2, 2, 4, 2],
        "widths": [64 * 3, 128 * 3, 256 * 3, 512 * 3],
        "joint": [True, True, True, False],
        "self_attns": [
            {"type": "neighborhood", "d_head": 64, "kernel_size": 5},
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "global", "d_head": 64},
            {"type": "global", "d_head": 64},
        ],
    },
    "ImageNet64": {
        "patch_size": [2, 2],
        "depths": [2, 2, 8],
        "widths": [128 * 3, 256 * 3, 512 * 3],
        "joint": [True, True, True,],
        "self_attns": [
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "global", "d_head": 64},
        ],
    },
}