So...

The issue with metric is, that some of our models return, for example,
a list of `top-k` items, and asking them to return score for every label is impossible.

Some other models return score for every label.

Some other models return `rank` for every positive label only.

So for performance reasons I decided to implement each metric for those three
scenarios
* When we have `top-k` predictions
* When we have `scores` for every label
* When we have `ranks` for positive labels.

Feel free to prpose how to handle this in a nicer way.
