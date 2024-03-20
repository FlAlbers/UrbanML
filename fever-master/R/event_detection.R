library(tidyverse)

#' Detect rainfall events from given timereries. values are interpreted as mm/dt
#'
#' @param x The xts object to be analyzed.
#' @param value_threshold The minimum threshold for individual timestep.
#' @param event_threshold The threshold for relevant event within sep-time
#' @param n_max_event_time The search window / separation time per event. 
#' @param n_roll_max_event # timesteps for max sum over n 
#'
#' @return tibble with event details
#' @export
#'
#' @examples
event_detection <- function(x,
                            value_threshold = 0.01,
                            event_threshold = 0.5,
                            n_max_event_time = 48,
                            n_roll_max_event = 12) {
  
  events <-
    rainfall_events(x,
                    value_threshold,
                    event_threshold,
                    n_max_event_time,
                    n_roll_max_event)

    # combine with dates
    event_tbl <- tibble(start=zoo::index(x[events[,1]+1]),
                     end=zoo::index(x[events[,2]+1]),
                     hN_mm=events[,3],
                     iN_mean=events[,4],
                     iN_max=events[,5],
                     hN_max_60=events[,6]) %>%
      mutate(DN_h=as.numeric(difftime(end,start,tz = xts::tzone(x),units = "min")/60))
  
  
  
}



