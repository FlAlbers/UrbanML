// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;


double run_sum_max(arma::vec x, int n){
  
  int sz = x.size();
  
  arma::vec res(sz,arma::fill::zeros);
  
  for(int i = 0; i < (sz-n+1); i++){
    
    res[i+n-1] = arma::sum(x.subvec(i, i + n - 1));
    
  }
  // pad the first n-1 elements with NA
  return res.max();
}

//-----------------------------------------------------------------------------
//' @title Analyse an xts object
//' @description Get start and end points where a given threshold and event sum 
//' within a given time is satisfied.
//' @param x The xts object to be analyzed.
//' @param value_threshold The minimum threshold for individual timestep.
//' @param event_threshold The threshold for relevant event within sep-time
//' @param n_max_event_time The search window / separation time per event. 
//' @return A 2-column matrix with start and end indices.
//' @export
// [[Rcpp::export(rainfall_events)]]
arma::mat rainfall_events(arma::vec x,
                          double value_threshold=0.01, 
                          double event_threshold=0.5, 
                          int n_max_event_time=48,
                          int n_roll_max_event=12) {
  //---------------------------------------------------------------------------
  
  // get the size of the vector to setup the result matrix
  int n = x.size();
  int m=0; 
  // initially there is no event
  bool event = FALSE;
  // current sum within separation time
  double sum_event_follow;
  
  // create instance of type armadillo matrix with length n and columns 2
  // column 1 will contain the start point of an event
  // column 2 will contain the end point of an event
  // each row represents an event
  arma::mat event_mat(n, 6, arma::fill::zeros);
  event_mat.fill(NA_REAL);
  
  // initially there is no event
  int event_idx = 0;
  
  // run through the vector
  for(int i = 0; i < n; i++) {
    
    if(!event && (x(i) <= value_threshold)) {
      // no event 
      
    } else if(event && (x(i) > value_threshold)) {
      //event is ongoing
      
    } else {
      // check if event sum in seaparation time > event_threshold
      m =std::min(n-1,i + n_max_event_time);
      sum_event_follow = arma::sum(x.subvec(i, m));
      
      if(!event && (sum_event_follow > event_threshold)) {
        // start event
        event = TRUE;
        event_mat(event_idx,0) =i;
        
      } else if(event && (sum_event_follow <= event_threshold)) {
        // finish event and store end index
        event=FALSE;
        event_mat(event_idx,1) =i;
        event_idx++;
        
      } 
      
    }
    
  }
  
  // after all, we resize the original matrix to
  event_mat.resize(std::max(event_idx,1), 6);
  
  // now get simple event statistic
  // sum, mean, max, rollmax in n
  arma::vec event_vals;
  
  for(int i = 0; i < event_idx; i++) {
    event_vals=x.subvec(event_mat(i,0), event_mat(i,1));
    event_mat(i,2)=arma::sum(event_vals);
    event_mat(i,3)=arma::mean(event_vals);
    event_mat(i,4)=arma::max(event_vals);
    event_mat(i,5)=run_sum_max(event_vals,n_roll_max_event);

  }
  
  
  // return 2 column matrix with event start and end points...
  return(event_mat);
 
}

