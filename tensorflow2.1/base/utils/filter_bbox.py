def get_area(bbox):
    """
    bbox: [x1, y1, x2, y2], upper-left corner and lower-right corner.
    """
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        return (bbox[2] - bbox[0])*(bbox[3]-bbox[1])*1.0
    else:
        return 0.0

def get_overlap(bbox_1, bbox_2):
    """
    bbox: [x1, y1, x2, y2], upper-left corner and lower-right corner.
    """
    left_x = max(bbox_1[0], bbox_2[0])
    up_y = max(bbox_1[1], bbox_2[1])
    right_x = min(bbox_1[2], bbox_2[2])
    low_y = min(bbox_1[3], bbox_2[3])
    area_overlap = get_area([left_x, up_y, right_x, low_y])
    return area_overlap

def filter_bbox_by_overlap(bboxes, overlap_thres=0.5, self_cover_thres=0.5):
    """
    bbox: [x1, y1, x2, y2]
    DO NOT ALTER BBOXES.
    """
    bbox_num = len(bboxes)
    if bbox_num < 2:
        return [bid for bid in range(bbox_num)]
    ## descend sort by area.
    bboxes_with_id = [(bid, bbox) for bid, bbox in enumerate(bboxes)]
    bboxes_with_id = sorted(bboxes_with_id, key = lambda bbox:get_area(bbox[1]), reverse=True)
    bboxes_sorted, ids_sorted = [], []
    for bid, bbox in bboxes_with_id:
        bboxes_sorted.append(bbox)
        ids_sorted.append(bid)
    ## filter by overlap
    keep_bbox = [True for _ in range(bbox_num)]
    for bid, bbox in enumerate(bboxes_sorted):
        if keep_bbox[bid]:
            area_bid = get_area(bbox)
            for pbid in range(bid+1, bbox_num):
                area_pbid = get_area(bboxes_sorted[pbid])
                area_overlap = get_overlap(bbox, bboxes_sorted[pbid])
                area_total = area_bid + area_pbid - area_overlap
                if area_overlap/area_total >= overlap_thres or area_overlap/area_pbid > self_cover_thres:
                    keep_bbox[pbid] = False
    # return id
    filtered_bids = [ids_sorted[bid] for bid in range(bbox_num) if keep_bbox[bid]]
    return filtered_bids

def get_filtered(bboxes, *meta_data, overlap_thres=0.5, self_cover_thres=0.5):
    filtered_bids = filter_bbox_by_overlap(bboxes, overlap_thres, self_cover_thres)
    results = []
    bboxes_filtered = [bboxes[bid] for bid in filtered_bids]
    results.append(bboxes_filtered)
    for meta in meta_data:
        meta_filtered = [meta[bid] for bid in filtered_bids]
        results.append(meta_filtered)
    return results