import torch

def compute_iou(coord1, coord2): # Implementation for tensor
    bbox1 = coord_to_bbox(coord1) # N * S * S * 4
    bbox2 = coord_to_bbox(coord2) # N * S * S * 4

    # Compute intersection
    x1 = torch.max(bbox1[..., 0:1], bbox2[..., 0:1]) # N * S * S * 2 -> max -> N * S * S * 1
    y1 = torch.max(bbox1[..., 1:2], bbox2[..., 1:2]) # N * S * S * 2 -> max -> N * S * S * 1
    x2 = torch.min(bbox1[..., 2:3], bbox2[..., 2:3]) # N * S * S * 2 -> max -> N * S * S * 1
    y2 = torch.min(bbox1[..., 3:4], bbox2[..., 3:4]) # N * S * S * 2 -> max -> N * S * S * 1

    intersection = (x2 - x1) * (y2 - y1)

    # Compute union
    union = (bbox1[..., 2:3] - bbox1[..., 0:1]) * (bbox1[..., 3:4] - bbox1[..., 1:2]) + (bbox2[..., 2:3] - bbox2[..., 0:1]) * (bbox2[..., 3:4] - bbox2[..., 1:2]) - intersection
    return intersection / union


def coord_to_bbox(coord): # Implementation for tensor
    assert coord.shape[-1] == 4 # N * S * S * 4
    bbox = []
    bbox.append(coord[..., 0:1] - coord[..., 2:3] / 2)
    bbox.append(coord[..., 1:2] - coord[..., 3:4] / 2)
    bbox.append(coord[..., 0:1] + coord[..., 2:3] / 2)
    bbox.append(coord[..., 1:2] + coord[..., 3:4] / 2)

    return torch.cat(bbox, dim=-1)