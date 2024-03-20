// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;

//-----------------------------------------------------------------------------
//' @title Analyse an xts object
//' @description Get start and end points where a given threshold is
//' satisfied.
//' @param x The xts object to be analyzed.
//' @param threshold The minimum threshold.
//' @param window The search window.
//' @return A 2-column matrix with start and end indices.
//' @export
// [[Rcpp::export(analyser)]]
arma::mat analyser(arma::vec x, double threshold, int window) {
  //-----------------------------------------------------------------------------
  
  // get the size of the vector to setup the result matrix
  int n = x.size();
  // initially there is no event
  bool event = FALSE;
  
  // create instance of type armadillo matrix with length n and columns 2
  // column 1 will contain the start point of an event
  // column 2 will contain the end point of an event
  // each row represents an event
  arma::mat event_mat(n, 2, arma::fill::zeros);
  event_mat.fill(NA_REAL);
  
  // initially there is no event
  int event_idx = 0;
  
  // run through the vector
  for(int i = 0; i < n; i++) {
    
    // if the value excceeds the treshold and no previous event started, the event starts
    if ((x(i) > threshold) && (event == FALSE)) {
      // the event starts
      event = TRUE;
      // fill the row event_idx in the column 0 with the index of xts
      event_mat(event_idx, 0) = i;
      // next index
      i += 1;
    }
    
    // while the end of the vector - window is not reached and event is still TRUE
    while ( (i < (n - window) ) && (event == TRUE) ) {
      
      // find the maximum value within i:i+window
      // if it's below the threshold, the event ends
      if (arma::max(x.subvec(i, i + window)) <= threshold) {
        // event ends
        event = FALSE;
        // fill the result matrix with the event endpoint
        event_mat(event_idx, 1) = i;
        // next event ID
        event_idx += 1;
      }
      // next index
      i += 1;
      
    }
    
  }
  
  // after all, we resize the original matrix to
  event_mat.resize(std::max(event_idx,1), 2);
  
  // return 2 column matrix with event start and end points...
  return(event_mat);
  
}