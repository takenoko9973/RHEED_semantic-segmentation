import torch


def merge_predictions_by_priority(
    logits: torch.Tensor,
    background_index: int = 0,
    label_priority: list[int] | None = None,
) -> torch.Tensor:
    """複数クラスのロジット (背景含む) から優先順位に従って単一クラスにマージする

    Args:
        logits (torch.Tensor): 形状 (B, C, H, W)、Cはクラス数 (背景含む)
        background_index (int): 背景クラスのインデックス (デフォルト0)
        label_priority (list[int] | None): 優先順位の高いラベル番号のリスト
            None の場合は背景以外のクラス番号を昇順で使用

    Returns:
        torch.Tensor: 形状 (B, H, W)、マージ後のラベルマップ

    """
    B, C, H, W = logits.shape  # noqa: N806

    if label_priority is None:
        # 背景以外のラベルを昇順に並べる
        label_priority = [i for i in range(C) if i != background_index]

    bg = logits[:, background_index, :, :]
    merged_pred = torch.full((B, H, W), background_index, dtype=torch.long, device=logits.device)

    for label in label_priority:
        score = logits[:, label, :, :] - bg
        mask = (merged_pred == background_index) & (score > 0)
        merged_pred[mask] = label

    return merged_pred
