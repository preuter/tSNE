# tSNE
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.

This is a port from http://lvdmaaten.github.io/tsne/ for use in PHP projects. The PHP was written to be compatible in PHP 4 and above, at the time of writing this, is known to work in PHP 5.5.9

# Typical Use Case
## From data values
    $samples = array( array(1,0.1,0.2), array(0.1,1,0.3), array(0.2,0.1,1));
    $tsne = new tSNE(array('dim'=>2));
    $tsne->initDataRaw($samples);
    for( $k=0; $k<500; $k++) {
      // every time you call this, the solution gets better.
      $tsne->step();
    }
    $Y = $tsne->getSolution();
    print_r($Y);

## From distance (perhaps a correlation, or some similarity metric)
Simply change `$tsne->initDataRaw($samples);` to `$tsne->initDataDist($samples);` in the above fragment. Useful when you have cosine similarity or another distance metric, that you wish to compute spatial similarity maps from.
