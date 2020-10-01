#!/usr/bin/env python3


class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
    	margin
    	type_of_triplets: options are "all", "hard", or "semihard".
    		"all" means all triplets that violate the margin
    		"hard" is a subset of "all", but the negative is closer to the anchor than the positive
    		"semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """
    def __init__(self, margin, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.add_to_recordable_attributes(list_of_names=[
            "avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"
        ],
                                          is_stat=True)
        self.type_of_triplets = type_of_triplets

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        import pdb
        pdb.set_trace()  ## DEBUG ##
        anchor_idx, positive_idx, negative_idx = get_random_triplet_indices(
            labels, ref_labels, t_per_anchor=1)
        anchors, positives, negatives = embeddings[anchor_idx], ref_emb[
            positive_idx], ref_emb[negative_idx]
        ap_dist = cosine_distance_torch(anchors, positives)
        an_dist = cosine_distance_torch(anchors, negatives)
        triplet_margin = an_dist - ap_dist
        self.pos_pair_dist = torch.mean(ap_dist).item()
        self.neg_pair_dist = torch.mean(an_dist).item()
        self.avg_triplet_margin = torch.mean(triplet_margin).item()
        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= an_dist <= ap_dist
            elif self.type_of_triplets == "semihard":
                threshold_condition &= an_dist > ap_dist
        return anchor_idx[threshold_condition], positive_idx[
            threshold_condition], negative_idx[threshold_condition]


if __name__ == "__main__":
    criterion = TripletMarginMiner(margin=0.3, type_of_triplets="semihard")
    feats = torch.randn((11, 128))
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])

    criterion(feats, labels)
