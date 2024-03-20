## helper function

#' Split an xts-object into a list of xts-objects by eventpoints
#'
#' @param x An xts-object 
#' @param m A two-column matrix containing eventpoints
#'
#' @return A list of xts-objects
#' @export
split_by_ep <- function(x, m) {
  
  apply(m, 1, function(ep) {
    if (any(is.na(ep))) return(NULL)
    x[ep[1]:ep[2]]
  })
  
}
