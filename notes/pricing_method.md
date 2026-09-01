# how the pricing verdict is made

The verdict is built from closed sales, not opinions. A comparable sale (a "comp") is a home in the same zip code with the same number of bedrooms that closed within the last twelve months. The median of those sale prices is the anchor: half the comps sold for more, half for less, and a single outlier cannot drag it.

A list price within five percent of the anchor is called fairly priced in a balanced market. The band widens to seven percent in a sellers' market (under four months of supply) because buyers are paying premiums, and narrows to three percent in a buyers' market (over six months of supply) because they are not.

A second vote comes from a statistical model trained on the same sales: a hedonic regression that prices the bundle of attributes (size, beds, baths, age, lot, condition, zip) plus a small neural network that can bend that line. When the two disagree by more than ten percent the model's vote is discounted.

Price per square foot is used as a size check, not a third vote. Small homes carry a naturally high price per square foot, so the check is compared against comps of similar size (within thirty percent of the subject) and it lowers confidence when it disagrees rather than moving the price.

Confidence combines how much evidence there is (comp count, model availability), how well the signals agree, and how far the combined figure sits from the five-percent line. Below a confidence of 0.55 no call is issued; the file is routed to a human analyst with the numbers attached.
