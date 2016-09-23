<?php
 /**
  * t-Distributed Stochastic Neighbor Embedding (t-SNE)
  * @link http://lvdmaaten.github.io/tsne/
  *
  * t-SNE is a technique for dimensionality reduction that is particularly
  * well suited for the visualization of high-dimensional datasets.
  *
  *
  * @file tSNE.php
  * @date 2015-10-13 12:47 PDT
  * @author Paul Reuter
  * @version 1.0.0
  *
  * @modifications <pre>
  * 1.0.0 - 2015-10-13 - Created from template: phpclass
  * </pre>
  */



/**
 * t-Distributed Stochastic Neighbor Embedding (t-SNE)
 * @package tSNE
 */
class tSNE {
  /**
   * Number of nearest neighbors.
   * @access public
   * @type uint
   */
  var $perplexity = 30;
  /**
   * Number of output dimensions.
   * @access public
   * @type uint
   */
  var $dim = 2;
  /**
   * Learning rate.
   * @access public
   * @type float
   */
  var $epsilon = 10;



  // BEGIN PUBLIC API



  /**
   * t-Distributed Stochastic Neighbor Embedding (t-SNE)
   *
   * @access public
   * @return new tSNE object
   */
  function tSNE($opts=null) { 
    if( !is_array($opts) ) {
      $opts = array();
    }

    foreach( array('perplexity','dim','epsilon') as $k_opt ) {
      if( isset($opts[$k_opt]) ) {
        $this->$k_opt = $opts[$k_opt];
      }
    }

    $this->iter = 0;
    return $this;
  } // END: constructor tSNE


  /**
   * Take a set of N-dimensional points, create matrix P using Gaussian kernel.
   *
   * @access public
   * @param array $X array of n-dimensional points.
   * @return bool true on success, false on failure.
   */
  function initDataRaw($X) {
    $N = count($X);
    $D = count($X[0]);
    $this->assert($N>0, " X is empty? You must have some data!");
    $this->assert($D>0, " X[0] is empty? Where is the data?");
    $dists = $this->xtod($X); // convert X to distances using gaussian kernel
    $this->P = $this->d2p($dists, $this->perplexity, 1e-4);
    $this->N = $N; // back up the size of the dataset
    $this->initSolution(); // refresh this
    return (!empty($P)) ? true : false;
  } // END: function initDataRaw($X)


  /**
   * Taks a given distance matrix and creates matrix P from them.
   *
   * @param array $D an array of arrays, and should be symmetric (Dij == Dji).
   * @return bool true if successful, false on failure.
   */
  function initDataDist($D) {
    $N = count($D);
    $this->assert($N>0, " X is empty? You must have some data!");
    // convert D to a (fast) typed array version
    $dists = $this->zeros($N*$N);
    for($i=0; $i<$N; $i++) {
      for($j=$i+1; $j<$N; $j++) {
        $d = $D[$i][$j];
        $dists[$i*$N+$j] = $d;
        $dists[$j*$N+$i] = $d;
      }
    }
    $this->P = $this->d2p($dists, $this->perplexity, 1e-4);
    $this->N = $N;
    $this->initSolution();
    return (!empty($P)) ? true : false;
  } // END: function initDataDist($D)


  /**
   * @access public
   */
  function initSolution() {
    // the solution
    $this->Y = $this->randn2d($this->N, $this->dim);
    // step gains to accelerate progress in unchanging directions
    $this->gains = $this->randn2d($this->N, $this->dim, 1.0);
    // momentum accumulator
    $this->ystep = $this->randn2d($this->N, $this->dim, 0.0);
    $this->iter = 0;
    return true;
  } // END: function initSolution()


  /**
   * @access public
   */
  function getSolution() {
    return (isset($this->Y)) ? $this->Y : false;
  } // END: function getSolution()


  /**
   * @access public
   */
  function step() {
    $this->iter += 1;
    $N = $this->N;

    $cg = $this->costGrad($this->Y); // evaluate gradient
    $cost = $cg['cost'];
    $grad = $cg['grad'];

    // perform gradient step
    $ymean = $this->zeros($this->dim);
    for($i=0; $i<$N; $i++) {
      for($d=0, $D=$this->dim; $d<$D; $d++) {
        $gid = $grad[$i][$d];
        $sid = $this->ystep[$i][$d];
        $gainid = $this->gains[$i][$d];

        // compute gain update
        $samesign = ($this->sign($gid)==$this->sign($sid));
        $newgain = ($samesign) ? $gainid * 0.8 : $gainid + 0.2;
        if( $newgain < 0.01 ) {
          $newgain = 0.01; // clamp
        }
        $this->gains[$i][$d] = $newgain; // store for next turn

        // compute momentum step direction
        $momval = ($this->iter) < 250 ? 0.5 : 0.8;
        $newsid = $momval * $sid - $this->epsilon * $newgain * $grad[$i][$d];
        $this->ystep[$i][$d] = $newsid; // remember the step we took

        // step!
        $this->Y[$i][$d] += $newsid;

        $ymean[$d] += $this->Y[$i][$d]; // accumulate mean to re-center later
      }
    }

    // reproject Y to be zero mean
    for($i=0; $i<$N; $i++) {
      for($d=0,$D=$this->dim; $d<$D; $d++) {
        $this->Y[$i][$d] -= $ymean[$d]/$N;
      }
    }

    return $cost;
  } // END: function step()


  /**
   * @access public
   */
  function debugGrad() {
    $N = $this->N;

    $cg = $this->costGrad($this->Y); // evaluate gradient
    $cost = $cg['cost'];
    $grad = $cg['grad'];

    $e = 1e-5;
    for($i=0; $i<$N; $i++) {
      for($d=0, $D=$this->dim; $d<$D; $d++) {
        $yold = $this->Y[$i][$d];

        $this->Y[$i][$d] = $yold + $e;
        $cg0 = $this->costGrad($this->Y);

        $this->Y[$i][$d] = $yold - $e;
        $cg1 = $this->costGrad($this->Y);

        $analytic = $grad[$i][$d];
        $numerical = ($cg0['cost'] - $cg1['cost']) / (2*$e);
        error_log("debug: $i, $d: analytic: $analytic, numeric: $numerical");

        $this->Y[$i][$d] = $yold;
      }
    }
  } // END: function debugGrad()


  /**
   * Return cost and gradient, given an arrangement
   *
   * @access public
   */
  function costGrad($Y) {
    $N = $this->N;
    $dim = $this->dim; // dimension of output space (dft=2)
    $P = $this->P;

    $pmul = ($this->iter < 100) ? 4 : 1; // trick to help with local optima

    // compute current Q distribution, unnormalized first
    $NN = $N*$N;
    $Qu = $this->zeros($NN);
    $qsum = 0.0;
    for($i=0; $i<$N; $i++) {
      for($j=$i+1; $j<$N; $j++) {
        $dsum = 0.0;
        for($d=0; $d<$dim; $d++) {
          $dhere = $Y[$i][$d] - $Y[$j][$d];
          $dsum += $dhere*$dhere;
        }
        $qu = 1.0 / (1.0 + $dsum); // Student t-distribution
        $Qu[$i*$N+$j] = $qu;
        $Qu[$j*$N+$i] = $qu;
        $qsum += $qu + $qu;
      }
    }
    // normalize Q distribution to sum to 1
    $Q = $this->zeros($NN);
    for($q=0; $q<$NN; $q++) {
      $Q[$q] = max($Qu[$q] / $qsum, 1e-100);
    }

    $cost = 0.0;
    $grad = array();
    for($i=0; $i<$N; $i++) {
      // init grad for point i.
      $gsum = array_fill(0, $dim, 0.0);
      for($j=0; $j<$N; $j++) {
        $cost += -$P[$i*$N+$j] * log($Q[$i*$N+$j]); // accumulate cost
        $premult = 4 * ($pmul * $P[$i*$N+$j] - $Q[$i*$N+$j]) * $Qu[$i*$N+$j];
        for($d=0; $d<$dim; $d++) {
          $gsum[$d] += $premult * ($Y[$i][$d] - $Y[$j][$d]);
        }
      }
      $grad[] = $gsum;
    }

    return array('cost' => $cost, 'grad' => $grad);
  } // END: function costGrad()



  // END PUBLIC API
  // BEGIN helper routines.


  /**
   * @access private
   */
  function assert($condition, $message="Assertion failed", $isFatal=true) {
    if( !$condition ) {
      trigger_error($message, ($isFatal) ? E_USER_ERROR : E_USER_WARNING);
      return false;
    }
    return true;
  } // END: function assert($condition, $message)


  /**
   * @access private
   */
  function getopt($opt, $field, $dftVal) {
    return (isset($opt[$field])) ? $opt[$field] : $dftVal;
  } // END: function getopt($opt, $field, $dftVal)


  /**
   * @access private
   */
  function gaussRandom() {
    if( !isset($this->_return_v) ) {
      $this->_return_v = false;
      $this->_v_val = 0.0;
    }
    if( $this->_return_v ) {
      $this->_return_v = false;
      return $this->_v_val;
    }
    $u = 2*mt_rand()/mt_getrandmax() - 1;
    $v = 2*mt_rand()/mt_getrandmax() - 1;
    $r = $u*$u + $v*$v;
    if( $r==0 || $r>1 ) {
      return $this->gaussRandom();
    }
    $c = sqrt(-2*log($r)/$r);
    $this->_v_val = $v*$c; // cached for next time fn called.
    $this->_return_v = true;
    return $u*$c;
  } // END: function gaussRandom()


  /**
   * @access private
   */
  function randn($mu, $std) {
    return $mu + $this->gaussRandom()*$std;
  } // END: function randn($mu, $std)


  /**
   * Utility that returns N-dimensional array filled with random numbers,
   * or with the value $s, if provided.
   *
   * @access private
   */
  function randn2d($n, $d, $s=null) {
    $x = array();
    for($i=0; $i<$n; $i++) {
      $xhere = array();
      for($j=0; $j<$d; $j++) {
        $xhere[] = ($s===null) ? $this->randn(0.0, 1e-4) : $s;
      }
      $x[] = $xhere;
    }
    return $x;
  } // END: function randn2d($n, $d, $s=null)


  /**
   * @access private
   */
  function zeros($n=0) {
    if( (int)$n<=0 ) {
      return array();
    }
    return array_fill(0,$n,0);
  } // END: function zeros($n=0)


  /**
   * Compute L2 distance between two vectors
   *
   * @access private
   */
  function L2($x1, $x2) {
    $d = 0;
    for($i=0, $D=count($x1); $i<$D; $i++) {
      $diff = $x1[$i]-$x2[$i];
      $d += $diff*$diff;
    }
    return $d;
  } // END: function L2($x1, $x2)


  /**
   * Compute pairwise distance in all vectors in X
   *
   * @access private
   */
  function xtod($X) {
    $N = count($X);
    // allocate contiguous array
    $dist = $this->zeros($N*$N);
    for($i=0; $i<$N; $i++) {
      for($j=$i+1; $j<$N; $j++) {
        $d = $this->L2($X[$i], $X[$j]);
        $dist[$i*$N+$j] = $d;
        $dist[$j*$N+$i] = $d;
      }
    }
    return $dist;
  } // END: function xtod($X)


  /**
   * Compute (p_{i|j} + p_{j|i})/(2n)
   *
   * @access private
   */
  function d2p($D, $perplexity, $tol) {
    // D should be square (unrolled array), so N (and Nf) should be an integer.
    $Nf = sqrt(count($D));
    $N = intVal($Nf);
    $this->assert(!abs($Nf-$N), "D should have square number of elements.");
    $Htarget = log($perplexity); // target entropy of distribution
    $P = $this->zeros($N*$N); // temporary probability matrix

    $prow = $this->zeros($N);
    for($i=0; $i<$N; $i++) {
      $betamin = -INF;
      $betamax = INF;
      $beta = 1; // initial value of precision
      $done = false;
      $maxtries = 50;

      // perform binary search to find a suitable precision beta
      // so that the entropy of the distribution is appropriate
      $num = 0;
      while(!$done) {
        // compute entropy and kernel row with beta precision
        $psum = 0.0;
        for($j=0; $j<$N; $j++) {
          $pj = exp(-$D[$i*$N+$j] * $beta);
          if( $i===$j ) {
            $pj = 0; // we don't care about diagonals
          }
          $prow[$j] = $pj;
          $psum += $pj;
        }
        // normalize p and compute entropy
        $Hhere = 0.0;
        for($j=0; $j<$N; $j++) {
          $pj = ($psum==0) ? 0 : $prow[$j] / $psum;
          $prow[$j] = $pj;
          if( $prow[$j] > 1e-7 ) {
            $Hhere -= $pj * log($pj);
          }
        }

        // adjust beta based on result
        if( $Hhere > $Htarget ) {
          $betabin = $beta; // move up the bounds
          if( is_infinite($betamax) ) {
            $beta = $beta * 2;
          } else {
            $beta = ($beta + $betamax) / 2;
          }
        } else {
          // converse case. make distribution less peaky
          $betamax = $beta;
          if( is_infinite($betamin) ) {
            $beta = $beta / 2;
          } else {
            $beta = ($beta + $betamin) / 2;
          }
        }

        // stopping conditions: too many tries or got a good precision
        $num++;
        if( abs($Hhere - $Htarget) < $tol
        || $num >= $maxtries ) {
          $done = true;
        }
      }

      // copy over the final prow to P at row i
      for($j=0; $j<$N; $j++) {
        $P[$i*$N+$j] = $prow[$j];
      }
    } // end loop over example i

    // symmetrize P and normalize it to sum to 1 over all ij
    $Pout = $this->zeros($N*$N);
    $N2 = $N*2;
    for($i=0; $i<$N; $i++) {
      for($j=0; $j<$N; $j++) {
        $Pout[$i*$N+$j] = max(($P[$i*$N+$j] + $P[$j*$N+$i])/$N2, 1e-100);
      }
    }

    return $Pout;
  } // END: function d2p($D, $perplexity, $tol)


  /**
   * @access private
   */
  function sign($x) {
    return ($x>0) ? 1 : (($x<0) ? -1 : 0);
  } // END: function sign($x)



} // END: class tSNE


/*
// DEBUG:
$dists = array( array(1,0.1,0.2), array(0.1,1,0.3), array(0.2,0.1,1));
$tsne = new tSNE(array('epsilon'=>10));
$tsne->initDataDist($dists);
for( $k=0; $k<500; $k++) {
  // every time you call this, the solution gets better.
  $tsne->step();
}
$Y = $tsne->getSolution();
print_r($Y);
*/

// EOF -- tSNE.php
?>
