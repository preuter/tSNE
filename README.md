# tSNE
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.

This is an unofficial, direct port from https://github.com/karpathy/tsnejs/blob/master/tsne.js for use in PHP projects. It builds off research published at http://lvdmaaten.github.io/tsne/ This PHP was written to be compatible in PHP 4 and above, at the time of writing this, is known to work in PHP 5.5.9


## Research Paper
The algorithm was originally described in this paper:

    L.J.P. van der Maaten and G.E. Hinton.
    Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research
    9(Nov):2579-2605, 2008.

You can find the PDF [here](http://jmlr.csail.mit.edu/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).


## Typical Use Case
### From data values
    $samples = array( array(1, 0.1, 0.2), array(0.1, 1, 0.3), array(0.2, 0.1, 1));
    $tsne = new tSNE(array('dim' => 2));
    $tsne->initDataRaw($samples);
    for ($k=0; $k < 500; $k++) {
      // every time you call this, the solution gets better.
      $tsne->step();
    }
    $Y = $tsne->getSolution();
    print_r($Y);

### From distance (perhaps a correlation, or some similarity metric)
Simply change `$tsne->initDataRaw($samples);` to `$tsne->initDataDist($samples);` in the above fragment. Useful when you have cosine similarity or another distance metric, that you wish to compute spatial similarity maps from.

## License

MIT - As used by https://github.com/karpathy/tsnejs/

In keeping with L.J.P. van der Maaten's original wishes, "You are free to use, modify, or redistribute this software in any way you want, but only for non-commercial purposes. The use of the software is at your own risk; the authors are not responsible for any damage as a result from errors in the software." -- http://lvdmaaten.github.io/tsne/

